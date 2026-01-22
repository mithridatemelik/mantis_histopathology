# 01_prepare_dataset.ipynb

## purpose
Read the staged `raw_index.parquet` and produce the canonical **items table** with stable IDs.

## inputs (reads)
- `exports/stage_00_download/raw_index.parquet`

## outputs (writes)
Directory: `exports/stage_01_prepare/`

Core parquet:
- `items.parquet`

Optional diagnostics:
- `eda/*.json` (EDA catalog results that apply to the items table)
- `plots/label_distribution.png`

## notes
This stage is intentionally lightweight and provides the stable `item_id` contract that all downstream stages depend on.
