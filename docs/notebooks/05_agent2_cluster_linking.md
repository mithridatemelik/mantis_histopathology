# 05_agent2_cluster_linking.ipynb

This is the **"0.5" notebook** to test.

## purpose
Run **agent 2** to link clusters post-cleanup (tools + database + optional human-in-the-loop review).

## inputs (reads)
- `exports/stage_04_agent1_cleanup/clusters_semantic.parquet`
- `exports/stage_03_clustering/cluster_centroids.parquet`
- `exports/stage_04_agent1_cleanup/items_after_agent1.parquet`

## outputs (writes)
Directory: `exports/stage_05_agent2_linking/`

Core parquet:
- `cluster_links.parquet`
- `agent2_memory.parquet`
- `items_after_agent2.parquet`
- `mantis_export.parquet` + `mantis_export.csv`

Multi-dataset convenience (enabled by default):
- `mantis_export_by_dataset/mantis_export_<dataset>.parquet`
- `mantis_export_by_dataset/mantis_export_<dataset>.csv`

Optional diagnostics:
- `eda/relationship_counts.parquet`
- `eda/relationship_counts.png`

## tools + database + human-in-the-loop
- similarity shortlist is computed from centroid vectors
- optional DuckDB querying over parquet artifacts
- optional review CSV export/import for human overrides
