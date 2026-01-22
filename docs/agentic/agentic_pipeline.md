# agentic pipeline

This project implements a **real (non-mocked) agentic workflow** for semantic cluster building, cluster linking, and knowledge graph construction.

Reference diagram:

- `docs/agentic/agentic_workflow_diagram.png`

## architecture mapping (diagram → implementation)

**user → prompt → agent 1 (memory + reasoning + chat) → agent 2 (tools + database + human-in-the-loop) → output**

### user → prompt
- implemented inside notebooks as the "run configuration" cells:
  - dataset keys
  - safe mode / sampling controls
  - model choice
  - thresholds

### agent 1: memory + reasoning + chat (cluster cleanup)
Notebook: `notebooks/04_agent1_cluster_cleanup.ipynb`

Responsibilities:
- read clustering results
- build per-cluster evidence summaries
- call OpenAI Chat Completions to produce:
  - `cluster_name` (unique, short)
  - `cluster_description`
  - `cluster_keywords`

Memory:
- persisted to parquet:
  - `exports/stage_04_agent1_cleanup/agent1_memory.parquet`

This allows:
- resuming without re-calling the LLM
- caching per-cluster outputs
- reproducible reruns

Output:
- `exports/stage_04_agent1_cleanup/clusters_semantic.parquet`
- `exports/stage_04_agent1_cleanup/items_after_agent1.parquet`

### agent 2: tools + database + human-in-the-loop (cluster linking)
Notebook: `notebooks/05_agent2_cluster_linking.ipynb`

Tools:
- similarity shortlist from centroid cosine similarity
- optional DuckDB querying over parquet artifacts

LLM step:
- OpenAI call classifies relationship:
  - `same_as`, `subtype_of`, `overlaps_with`, `related_to`, `unrelated`
- returns:
  - relationship
  - confidence
  - short rationale

Human-in-the-loop:
- export a review CSV
- re-import reviewed decisions to override model outputs

Memory:
- persisted to parquet:
  - `exports/stage_05_agent2_linking/agent2_memory.parquet`

Output:
- `exports/stage_05_agent2_linking/cluster_links.parquet`
- `exports/stage_05_agent2_linking/mantis_export.parquet`
- `exports/stage_05_agent2_linking/mantis_export.csv`

### output: knowledge graph construction
Notebook: `notebooks/06_build_knowledge_graph.ipynb`

Inputs:
- semantic clusters from agent 1
- cluster links from agent 2
- (optional) items table for item→cluster membership

Outputs (parquet-first):
- `kg_nodes.parquet`
- `kg_edges.parquet`
- `kg_summary.parquet`

Optional:
- `kg_visualization.html` (pyvis) for quick inspection in Colab

---

## why parquet memory (not json)

Parquet:
- is queryable
- is schema-friendly
- integrates with the pipeline's parquet-first contracts
- supports resuming without additional glue code

---

## models

The notebooks default to a lightweight chat model (configurable in-notebook).
No model name is hardcoded into library code; you can choose based on availability and cost.
