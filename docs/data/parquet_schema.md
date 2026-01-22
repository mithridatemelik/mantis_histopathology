# parquet schemas (parquet-first data contracts)

This project is **parquet-first**: every notebook stage writes its outputs as parquet, and downstream stages **only read parquet artifacts**.

Below are the expected schemas (columns). Some stages may add extra audit columns, but downstream dependencies assume at least these.

---

## exports/artifact_manifest.parquet

| column | type | meaning |
|---|---|---|
| stage | string | stage name (e.g., `stage_03_clustering`) |
| artifact | string | artifact name (e.g., `items_with_clusters`) |
| path | string | absolute/relative output path |
| schema_version | string | schema version tag |
| created_at_utc | string | ISO timestamp |
| inputs_json | string | JSON list of input fingerprints |
| rows | int | row count (if known) |
| cols | int | column count (if known) |
| notes | string | optional notes |

---

## stage 00 — download

### exports/stage_00_download/raw_index.parquet

| column | type | meaning |
|---|---|---|
| dataset_key | string | logical dataset key (e.g., `CRC_VAL_HE_7K`) |
| provider | string | provider (`zenodo`, `hf`, `medmnist`, etc.) |
| split | string | split label (`train`, `val`, `test`, etc.) |
| label | string | class folder / label code |
| image_path | string | staged image path |
| width | int | image width |
| height | int | image height |
| mpp | float | microns-per-pixel (if known; default 0.5) |

---

## stage 01 — prepare

### exports/stage_01_prepare/items.parquet

Canonical items table.

| column | type |
|---|---|
| item_id | string |
| source | string |
| split | string |
| label | string |
| text | string |
| image_path | string |
| width | int |
| height | int |
| mpp | float |

---

## stage 02 — embeddings

### exports/stage_02_embeddings/items_with_embeddings.parquet

| column | type | notes |
|---|---|---|
| (all `items.parquet` columns) |  |  |
| model_id | string | fused embedding model id |
| dim | int | vector dimension |
| vector | object | list[float] stored in parquet |

---

## stage 03 — clustering

### exports/stage_03_clustering/items_with_clusters.parquet

| column | type |
|---|---|
| (all `items_with_embeddings` columns) |  |
| cluster_id | int |
| x | float | optional 2D coordinate |
| y | float | optional 2D coordinate |

### exports/stage_03_clustering/cluster_centroids.parquet

| column | type |
|---|---|
| cluster_id | int |
| n_items | int |
| vector | object (list[float]) |

---



### exports/stage_03_clustering/cluster_label_summary.parquet (optional)

A compact per-cluster label breakdown (top labels) for quick diagnostics.

| column | type |
|---|---|
| cluster_id | int |
| n_items | int |
| top_labels | object (list[dict]) |
## stage 04 — agent 1 cluster cleanup

### exports/stage_04_agent1_cleanup/clusters_semantic.parquet

| column | type |
|---|---|
| cluster_id | int |
| cluster_name | string |
| cluster_description | string |
| cluster_keywords | object (list[str]) |
| n_items | int |
| dominant_labels | object (list[dict]) |

### exports/stage_04_agent1_cleanup/agent1_memory.parquet

| column | type |
|---|---|
| cluster_id | int |
| signature | string |
| model | string |
| temperature | float |
| prompt | string |
| response_json | string |
| cluster_name | string |
| cluster_description | string |
| cluster_keywords | string (json list) |
| created_at_utc | string |

### exports/stage_04_agent1_cleanup/items_after_agent1.parquet

Same as `items_with_clusters.parquet`, plus optional semantic columns such as `cluster_name`.

---

## stage 05 — agent 2 cluster linking

### exports/stage_05_agent2_linking/cluster_links.parquet

| column | type |
|---|---|
| src_cluster_id | int |
| dst_cluster_id | int |
| similarity | float |
| relationship | string |
| confidence | float |
| rationale | string |

### exports/stage_05_agent2_linking/agent2_memory.parquet
Same concept as agent 1, but keyed by `(src_cluster_id, dst_cluster_id)`.

### exports/stage_05_agent2_linking/mantis_export.parquet

A Mantis-ready flat table. Minimum fields:

| column | type |
|---|---|
| id | string |
| text | string |
| vector | object (list[float]) |
| vector_str | string (json) |
| cluster_id | int |
| cluster_name | string |
| metadata | string (json) |

A CSV version is also written: `mantis_export.csv`.



### exports/stage_05_agent2_linking/mantis_export_by_dataset/ (optional, enabled by default)

Per-dataset Mantis exports (one parquet + csv per `source` value), to support multi-dataset runs without recomputing upstream artifacts.

Files:
- `mantis_export_<dataset>.parquet`
- `mantis_export_<dataset>.csv`
---

## stage 06 — knowledge graph

### exports/stage_06_knowledge_graph/kg_nodes.parquet

| column | type |
|---|---|
| node_id | string |
| node_type | string |
| name | string |
| description | string |
| attributes_json | string |

### exports/stage_06_knowledge_graph/kg_edges.parquet

| column | type |
|---|---|
| edge_id | string |
| src | string |
| dst | string |
| rel | string |
| weight | float |
| attributes_json | string |



### exports/stage_06_knowledge_graph/kg_provenance.parquet

A minimal provenance table (evidence per edge). This is intentionally simple but makes demos/debugging easier.

| column | type |
|---|---|
| provenance_id | string |
| edge_id | string |
| evidence_json | string (json) |
### exports/stage_06_knowledge_graph/kg_summary.parquet

A single-row table with basic counts and diagnostics.


### exports/stage_06_knowledge_graph/neo4j/ (optional)

Neo4j-friendly CSV exports:
- `nodes.csv`
- `rels.csv`
- `load.cypher`

### exports/stage_06_knowledge_graph/kg.ttl (optional)

RDF Turtle export (requires `rdflib`).


### exports/stage_06_knowledge_graph/eda/ (optional)

Not part of downstream contracts; convenience diagnostics:
- `node_type_counts.parquet`
- `rel_counts.parquet`
- `kg_stats.json`
- `degree_hist.png`
- `parquet_audit.parquet`
