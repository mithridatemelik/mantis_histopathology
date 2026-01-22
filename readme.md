# agentic histocartography colab workflow (dag + parquet-first + knowledge graph)

This repository provides a **Google Colab–ready**, **parquet-first**, **DAG-structured** notebook workflow for:

- downloading / staging histopathology patch datasets
- building embeddings and clustering patches
- running a **live (non-mocked) agentic pipeline**:
  - **agent 1**: semantic cluster cleanup (names/descriptions/keywords)
  - **agent 2**: post-cleanup cluster linking (relationships)
- exporting **Mantis-ready** tabular outputs
- building a **knowledge graph** from the agentic outputs

The agentic design follows the reference diagram (see `docs/agentic/agentic_workflow_diagram.png`):

**user → prompt → agent 1 (memory + reasoning + chat) → agent 2 (tools + database + human-in-the-loop) → output**

---

## quickstart (colab)

1. Upload this repository to Google Drive (or clone it from GitHub)  
2. Open and run notebooks **in order**:

By default, `pipeline_config.yaml` is set up for **multi-dataset** runs using `data.dataset_keys`. You can switch to a single dataset by editing that config.

| stage | notebook | outputs (parquet-first) |
|---|---|---|
| 00 | `notebooks/00_download_data.ipynb` | `exports/stage_00_download/raw_index.parquet` |
| 01 | `notebooks/01_prepare_dataset.ipynb` | `exports/stage_01_prepare/items.parquet` |
| 02 | `notebooks/02_compute_embeddings.ipynb` | `exports/stage_02_embeddings/items_with_embeddings.parquet` |
| 03 | `notebooks/03_cluster_embeddings.ipynb` | `exports/stage_03_clustering/items_with_clusters.parquet`, `cluster_centroids.parquet` |
| 04 | `notebooks/04_agent1_cluster_cleanup.ipynb` | `exports/stage_04_agent1_cleanup/clusters_semantic.parquet`, `agent1_memory.parquet` |
| 05 | `notebooks/05_agent2_cluster_linking.ipynb` | `exports/stage_05_agent2_linking/cluster_links.parquet`, `mantis_export.parquet`, `mantis_export.csv` |
| 06 | `notebooks/06_build_knowledge_graph.ipynb` | `exports/stage_06_knowledge_graph/kg_nodes.parquet`, `kg_edges.parquet`, `kg_visualization.html` |

✅ Each notebook does **one job**, writes **parquet artifacts**, and downstream notebooks **only read parquet outputs** (no re-upload / no re-download).

---

## secrets (required for agentic steps)

**Never hardcode tokens**.

### required
- `OPENAI_API_KEY` (required for stages 04–06)

### optional
- `HF_TOKEN` (only if downloading private Hugging Face artifacts)
- `MANTIS_TOKEN` (only if uploading to a Mantis server via API)

See:
- `docs/security/secrets_and_tokens.md`
- `docs/colab/google_colab_run_guide.md`

---

## artifact manifest (resume-friendly)

Every stage appends a record to:

- `exports/artifact_manifest.parquet`

This acts as a lightweight **artifact registry** with:
- stage name
- output paths
- schema version
- input fingerprints
- timestamps

---

## documentation

- `docs/colab/google_colab_run_guide.md`
- `docs/security/secrets_and_tokens.md`
- `docs/agentic/agentic_pipeline.md`
- `docs/data/parquet_schema.md`
- `docs/repo_audit/repo_audit.md`
- `docs/notebooks/*` (per-notebook I/O contracts)

---

## repository hygiene

- filenames are **lowercase** and **meaningful**
- unused legacy notebooks/artifacts are moved under `deprecated/`
- `__pycache__`, `*.pyc`, `.DS_Store` are removed

---

## license / dataset notes

This repo includes helper code for public datasets (e.g., Zenodo CRC patches, MedMNIST, Hugging Face datasets).
Always review dataset licenses before redistribution.
