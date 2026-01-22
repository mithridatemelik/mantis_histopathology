# 04_agent1_cluster_cleanup.ipynb

## purpose
Run **agent 1** to produce semantic cluster names/descriptions/keywords.

## inputs (reads)
- `exports/stage_03_clustering/items_with_clusters.parquet`
- `exports/stage_03_clustering/cluster_centroids.parquet` (optional)

## outputs (writes)
Directory: `exports/stage_04_agent1_cleanup/`

- `clusters_semantic.parquet`
- `agent1_memory.parquet`
- `items_after_agent1.parquet`

## live LLM requirement
Requires `OPENAI_API_KEY` (via env or Colab Secrets). No mocking.
