from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .runtime import _log


REL_TYPES = {
    "HAS_LABEL",
    "IN_CLUSTER",
    "SIMILAR_TO",
    "MENTIONS",
    "CO_OCCURS",
    "DERIVED_FROM",
    "PART_OF",
    "HAS_TAXON_LABEL",
    "EVIDENCE_FOR",
}


def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def _eid(prefix: str, key: str) -> str:
    return f"{prefix}:{_sha1(key)[:16]}"


def build_kg_tables(
    *,
    items: pd.DataFrame,
    clusters: pd.DataFrame,
    fused: pd.DataFrame,
    k_sim: int = 5,
    min_sim: float = 0.25,
    label_name_map: Optional[Dict[str, str]] = None,
    cluster_name_map: Optional[Dict[int, str]] = None,
    embedding_version: str = "",
    pipeline_version: str = "",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build entities/edges/provenance tables for a simple KG.

    Nodes:
      - item nodes (each patch)
      - label nodes (tissue class)
      - cluster nodes
      - dataset/source nodes

    Edges:
      - item --HAS_LABEL--> label
      - item --IN_CLUSTER--> cluster
      - item --PART_OF--> dataset
      - item --SIMILAR_TO--> item (kNN in fused space)

    Provenance:
      - one row per edge with evidence and confidence.
    """
    # Entities
    entities: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    prov: List[Dict[str, Any]] = []

    # Dataset/source nodes
    sources = sorted(items["source"].dropna().unique().tolist())
    source_eids = {s: _eid("dataset", s) for s in sources}
    for s in sources:
        entities.append(
            {
                "entity_id": source_eids[s],
                "entity_type": "DATASET",
                "name": s,
                "description": f"Histopathology dataset source: {s}",
            }
        )

    # Label nodes (optionally map short codes -> human-friendly names)
    label_name_map = label_name_map or {}
    labels = sorted(items["label"].dropna().unique().tolist())
    label_eids = {l: _eid("label", str(l)) for l in labels}
    for l in labels:
        # Keep the original code for traceability/search (e.g., "TUM")
        pretty = str(label_name_map.get(str(l), str(l)))
        name = f"{pretty} ({l})" if pretty != str(l) else str(l)
        entities.append(
            {
                "entity_id": label_eids[l],
                "entity_type": "LABEL",
                "name": name,
                "description": f"Tissue class label code={l}",
            }
        )

    # Cluster nodes (optionally use meaningful names from clusters.parquet or provided mapping)
    if cluster_name_map is None:
        if "cluster_name" in clusters.columns:
            try:
                tmp = (
                    clusters[["cluster_id", "cluster_name"]]
                    .dropna(subset=["cluster_name"])
                    .drop_duplicates(subset=["cluster_id"])
                )
                cluster_name_map = {int(k): str(v) for k, v in zip(tmp["cluster_id"].tolist(), tmp["cluster_name"].tolist())}
            except Exception:
                cluster_name_map = {}
        else:
            cluster_name_map = {}

    clus = clusters["cluster_id"].dropna().unique().tolist()
    cluster_eids = {int(c): _eid("cluster", str(int(c))) for c in clus}
    for c in sorted(cluster_eids.keys()):
        cname = cluster_name_map.get(int(c), "")
        display_name = f"C{c}: {cname}" if cname else f"C{c}"
        entities.append(
            {
                "entity_id": cluster_eids[c],
                "entity_type": "CLUSTER",
                "name": display_name,
                "description": f"Embedding cluster id={c}",
            }
        )

    # Item nodes
    item_eids = {iid: _eid("item", iid) for iid in items["item_id"].tolist()}
    for _, r in items.iterrows():
        entities.append(
            {
                "entity_id": item_eids[r["item_id"]],
                "entity_type": "ITEM",
                "name": r["item_id"],
                "description": r.get("text", ""),
            }
        )

    # Helper to add edge + provenance
    def add_edge(
        src_e: str,
        dst_e: str,
        rel: str,
        *,
        weight: float,
        evidence: str,
        confidence: float,
        source_item_id: str,
        edge_attrs: Optional[Dict[str, Any]] = None,
        prov_attrs: Optional[Dict[str, Any]] = None,
    ):
        edge_key = f"{src_e}|{rel}|{dst_e}|{evidence}"
        edge_id = _eid("edge", edge_key)
        prov_id = _eid("prov", edge_key)
        e_row: Dict[str, Any] = {
            "edge_id": edge_id,
            "src": src_e,
            "dst": dst_e,
            "rel": rel,
            "weight": float(weight),
            "provenance_id": prov_id,
        }
        if edge_attrs:
            e_row.update(edge_attrs)
        edges.append(e_row)

        p_row: Dict[str, Any] = {
            "provenance_id": prov_id,
            "source_item_id": source_item_id,
            "evidence_type": "computed" if rel == "SIMILAR_TO" else "field",
            "evidence": evidence,
            "confidence": float(confidence),
        }
        if prov_attrs:
            p_row.update(prov_attrs)
        prov.append(p_row)

    # Item->Label, Item->Cluster, Item->Dataset
    cl_map = clusters.set_index("item_id")["cluster_id"].to_dict()
    for _, r in items.iterrows():
        iid = r["item_id"]
        src = item_eids[iid]
        # label
        lab = r.get("label")
        if isinstance(lab, str) and lab in label_eids:
            add_edge(src, label_eids[lab], "HAS_LABEL", weight=1.0, evidence=f"items.label={lab}", confidence=1.0, source_item_id=iid)
        # cluster
        cid = cl_map.get(iid, None)
        if cid is not None and int(cid) in cluster_eids:
            add_edge(src, cluster_eids[int(cid)], "IN_CLUSTER", weight=1.0, evidence=f"cluster_id={int(cid)}", confidence=1.0, source_item_id=iid)
        # dataset
        s = r.get("source")
        if isinstance(s, str) and s in source_eids:
            add_edge(src, source_eids[s], "PART_OF", weight=1.0, evidence=f"items.source={s}", confidence=1.0, source_item_id=iid)

    # Similarity edges in fused embedding space (kNN)
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore

        X = np.array(fused["vector"].tolist(), dtype=np.float32)
        nn = NearestNeighbors(n_neighbors=min(int(k_sim) + 1, len(fused)), metric="cosine")
        nn.fit(X)
        dists, idx = nn.kneighbors(X, return_distance=True)

        item_ids = fused["item_id"].tolist()
        for i, (row_d, row_idx) in enumerate(zip(dists, idx)):
            src_item = item_ids[i]
            src_e = item_eids[src_item]
            for rank, (dist, j) in enumerate(zip(row_d[1:], row_idx[1:]), start=1):  # skip self
                sim = float(1.0 - dist)
                if sim < float(min_sim):
                    continue
                dst_item = item_ids[int(j)]
                dst_e = item_eids[dst_item]
                add_edge(
                    src_e,
                    dst_e,
                    "SIMILAR_TO",
                    weight=sim,
                    evidence=f"cosine_sim={sim:.4f}",
                    confidence=min(1.0, max(0.0, sim)),
                    source_item_id=src_item,
                    edge_attrs={
                        "rank": int(rank),
                        "distance": float(dist),
                        "similarity": float(sim),
                        "embedding_version": str(embedding_version),
                        "pipeline_version": str(pipeline_version),
                    },
                    prov_attrs={
                        "rank": int(rank),
                        "distance": float(dist),
                        "similarity": float(sim),
                        "embedding_version": str(embedding_version),
                        "pipeline_version": str(pipeline_version),
                    },
                )
    except Exception as e:
        _log().warning("Similarity edges skipped", extra={"extra": {"error": str(e)}})

    ent_df = pd.DataFrame(entities).drop_duplicates(subset=["entity_id"]).reset_index(drop=True)
    edge_df = pd.DataFrame(edges).drop_duplicates(subset=["edge_id"]).reset_index(drop=True)
    prov_df = pd.DataFrame(prov).drop_duplicates(subset=["provenance_id"]).reset_index(drop=True)

    _log().info(
        "KG tables built",
        extra={"extra": {"entities": len(ent_df), "edges": len(edge_df), "provenance": len(prov_df)}},
    )
    return ent_df, edge_df, prov_df
