from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from .runtime import _log


def save_parquet(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    _log().info("Saved parquet", extra={"extra": {"path": str(path), "rows": len(df)}})
    return path


def read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def export_neo4j_csv(
    *,
    entities: pd.DataFrame,
    edges: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, str]:
    """Export Neo4j-friendly CSVs + a minimal Cypher LOAD snippet."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = out_dir / "nodes.csv"
    rels_path = out_dir / "rels.csv"
    cypher_path = out_dir / "load.cypher"

    # Neo4j expects headers like :ID, :LABEL, etc.
    nodes = entities.copy()
    nodes[":ID"] = nodes["entity_id"]
    nodes[":LABEL"] = nodes["entity_type"]
    nodes["name"] = nodes["name"].astype(str)
    nodes["description"] = nodes.get("description", "").astype(str)
    nodes[[":ID", ":LABEL", "name", "description"]].to_csv(nodes_path, index=False)

    rels = edges.copy()
    rels[":START_ID"] = rels["src"]
    rels[":END_ID"] = rels["dst"]
    rels[":TYPE"] = rels["rel"]
    rels["weight"] = rels.get("weight", 1.0).astype(float)
    rels[[":START_ID", ":END_ID", ":TYPE", "weight"]].to_csv(rels_path, index=False)

    cypher = f"""// Place files into Neo4j import directory, then run:

LOAD CSV WITH HEADERS FROM 'file:///nodes.csv' AS row
MERGE (n {{id: row.`:ID`}})
SET n:Entity
SET n += {{
  entity_type: row.`:LABEL`,
  name: row.name,
  description: row.description
}};

LOAD CSV WITH HEADERS FROM 'file:///rels.csv' AS row
MATCH (a {{id: row.`:START_ID`}}), (b {{id: row.`:END_ID`}})
MERGE (a)-[r:REL {{type: row.`:TYPE`}}]->(b)
SET r.weight = toFloat(row.weight);
""".strip()
    cypher_path.write_text(cypher)

    _log().info("Exported Neo4j CSV", extra={"extra": {"nodes": str(nodes_path), "rels": str(rels_path)}})
    return {"nodes_csv": str(nodes_path), "rels_csv": str(rels_path), "cypher": str(cypher_path)}


def export_rdf_turtle(
    *,
    entities: pd.DataFrame,
    edges: pd.DataFrame,
    out_path: Path,
    base_iri: str = "http://example.org/histo/",
    entity_type_map: Optional[Dict[str, str]] = None,
    rel_type_map: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    """Export a very simple RDF graph in Turtle.

    Uses rdflib if installed; otherwise returns None.
    """
    try:
        from rdflib import Graph, Literal, Namespace, RDF, URIRef  # type: ignore
    except Exception as e:
        _log().warning("rdflib not installed; skipping RDF export", extra={"extra": {"error": str(e)}})
        return None

    H = Namespace(base_iri)
    entity_type_map = entity_type_map or {
        # Align with minimal semantic scaffold (without breaking parquet schema)
        "ITEM": "PATCH",
        "CLUSTER": "FEATURE",
    }
    rel_type_map = rel_type_map or {
        # Align cluster membership with HAS_FEATURE for semantic readiness
        "IN_CLUSTER": "HAS_FEATURE",
    }
    g = Graph()

    # Entities
    for _, r in entities.iterrows():
        uri = URIRef(base_iri + r["entity_id"])
        et = str(r["entity_type"])
        g.add((uri, RDF.type, H[entity_type_map.get(et, et)]))
        g.add((uri, H.name, Literal(str(r.get("name", "")))))
        if "description" in r and pd.notna(r["description"]):
            g.add((uri, H.description, Literal(str(r.get("description", "")))))

    # Edges
    for _, r in edges.iterrows():
        s = URIRef(base_iri + r["src"])
        o = URIRef(base_iri + r["dst"])
        rel = str(r["rel"])
        pred = H[rel_type_map.get(rel, rel)]
        g.add((s, pred, o))
        if "weight" in r and pd.notna(r["weight"]):
            g.add((s, H["weight"], Literal(float(r["weight"]))))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    g.serialize(destination=str(out_path), format="turtle")
    _log().info("Exported RDF Turtle", extra={"extra": {"path": str(out_path), "triples": len(g)}})
    return str(out_path)
