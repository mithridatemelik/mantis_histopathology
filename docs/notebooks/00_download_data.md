# 00_download_data.ipynb

## purpose
Download / sync raw histopathology patch datasets (Zenodo, Hugging Face, etc.) and stage them into a local folder.

## inputs
- `pipeline_config.yaml` (dataset selection + safe mode)
- optional: `HF_TOKEN` (private HF artifacts)

## outputs (writes)
Directory: `exports/stage_00_download/`

- `raw_index.parquet`  
  Parquet file inventory / metadata index that downstream notebooks read (no re-download).

## idempotency
If `raw_index.parquet` exists, the notebook skips downloading unless `FORCE_REBUILD=True`.
