from __future__ import annotations

"""Semantic layer helpers (minimal ontology scaffold + mapping spec).

This is intentionally small: the workshop uses simple parquet-first tables, but we
want an easy migration path to a semantic KG (RDF/OWL) later.
"""

from pathlib import Path
from typing import Optional


def write_minimal_ontology_turtle(
    *,
    out_path: Path,
    base_iri: str = "http://example.org/histo/",
) -> str:
    """Write a minimal Turtle ontology scaffold.

    Includes:
      - Classes: PATCH, SLIDE, LABEL, FEATURE
      - Properties: HAS_LABEL, SIMILAR_TO, HAS_FEATURE, FROM_SLIDE

    We keep it schema-light on purpose; downstream tools can extend as needed.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ttl = f"""@prefix : <{base_iri}> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .

### Classes

:PATCH a rdfs:Class ; rdfs:label "Patch" .
:SLIDE a rdfs:Class ; rdfs:label "Slide" .
:LABEL a rdfs:Class ; rdfs:label "Label" .
:FEATURE a rdfs:Class ; rdfs:label "Feature" .

### Object properties

:HAS_LABEL a rdf:Property ; rdfs:label "has label" ; rdfs:domain :PATCH ; rdfs:range :LABEL .
:SIMILAR_TO a rdf:Property ; rdfs:label "similar to" ; rdfs:domain :PATCH ; rdfs:range :PATCH .
:HAS_FEATURE a rdf:Property ; rdfs:label "has feature" ; rdfs:domain :PATCH ; rdfs:range :FEATURE .
:FROM_SLIDE a rdf:Property ; rdfs:label "from slide" ; rdfs:domain :PATCH ; rdfs:range :SLIDE .

### Data properties (lightweight)

:name a rdf:Property ; rdfs:label "name" .
:description a rdf:Property ; rdfs:label "description" .
:weight a rdf:Property ; rdfs:label "weight" .
"""
    out_path.write_text(ttl)
    return str(out_path)


def write_mapping_spec_md(*, out_path: Path) -> str:
    """Write column→node/edge mapping notes for future semantic migration."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    md = """# Semantic mapping spec (Parquet → Semantic KG)

This project currently stores graph data in **parquet-first** tables:

- `exports/kg/entities.parquet`
- `exports/kg/edges.parquet`
- `exports/kg/provenance.parquet`

The semantic layer is a future-facing compatibility target (e.g., for an RDF/OWL
or Mantis-style KG). This document records how current columns map to ontology
elements.

## Entities → Nodes

| entities.parquet column | semantic node field | notes |
|---|---|---|
| `entity_id` | IRI suffix | exported as `{base_iri}{entity_id}` |
| `entity_type` | rdf:type | mapped to a class (e.g., ITEM→PATCH, LABEL→LABEL) |
| `name` | :name | human-readable label |
| `description` | :description | optional |

## Edges → Triples

| edges.parquet column | semantic edge field | notes |
|---|---|---|
| `src` | subject | IRI of source entity |
| `rel` | predicate | mapped to an ontology property (e.g., HAS_LABEL, SIMILAR_TO) |
| `dst` | object | IRI of destination entity |
| `weight` | :weight | for SIMILAR_TO similarity strength |
| `rank` (optional) | provenance | neighbor rank for SIMILAR_TO |
| `distance` (optional) | provenance | cosine distance |
| `embedding_version` (optional) | provenance | embedding pipeline identifier |
| `pipeline_version` (optional) | provenance | workshop/pipeline version identifier |

## Provenance

`exports/kg/provenance.parquet` is a lightweight evidence table keyed by
`provenance_id`. In RDF, this could be represented using reified statements or a
named graph. For demo purposes we keep it as parquet.

## Notes

- A *SLIDE* node type is included in the minimal ontology, but not all datasets
  provide slide-level metadata. When available (e.g., WSI ID), add:
  `PATCH --FROM_SLIDE--> SLIDE`.
- A *FEATURE* node type is included for future interpretability features (e.g.,
  stain-normalization stats, morphology features, model attribution).
"""
    out_path.write_text(md)
    return str(out_path)
