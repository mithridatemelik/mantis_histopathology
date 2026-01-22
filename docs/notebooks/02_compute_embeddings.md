# 02_compute_embeddings.ipynb

## purpose
Compute per-item embeddings and write a fused vector for downstream clustering.

Modalities (best-effort):
- image embeddings (torchvision ResNet-50, with safe fallback)
- morphology/QC features + embeddings (optional, best-effort)
- text embeddings (optional; default off to avoid label leakage)

## inputs (reads)
- `exports/stage_01_prepare/items.parquet`

## outputs (writes)
Directory: `exports/stage_02_embeddings/`

Core parquet:
- `items_with_embeddings.parquet`
- `fused_embeddings.parquet`

Optional modality artifacts:
- `image_embeddings.parquet`
- `morph_features.parquet`
- `morph_embeddings.parquet`
- `text_embeddings.parquet`

Optional diagnostics:
- `eda/*.json` (EDA catalog results for embeddings/morphology when inputs exist)
- `qa/knn_label_consistency.json`
- `plots/fused_embedding_norms.png`
