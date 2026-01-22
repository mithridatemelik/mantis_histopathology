# 03_cluster_embeddings.ipynb

## purpose
Cluster fused embeddings and compute 2D cartography coordinates.

## inputs (reads)
- `exports/stage_02_embeddings/items_with_embeddings.parquet`

## outputs (writes)
Directory: `exports/stage_03_clustering/`

Core parquet:
- `items_with_clusters.parquet`
- `cluster_centroids.parquet`

Optional diagnostics:
- `cluster_label_summary.parquet`
- `eda/*.json` (clustering EDA catalog items)
- `plots/cartography_by_label.png`
- `plots/cartography_by_cluster.png`

Optional sweep artifacts (when HDBSCAN deps are available and SAFE_MODE is off):
- `hdbscan_sweep/`
- `best_run.json`
