# RUNBOOK — Histopathology Cartography (Colab)

## Quick start
1) Open **Colab** → **File → Open notebook → Google Drive**
2) Navigate to your folder:
   `/MyDrive/mit/histopathology_202502.../`
3) Run notebooks in order:
   - `00_HistoCartography_Setup_and_Paths.ipynb`
   - `01_HistoCartography_Data_Ingestion_Staging.ipynb`
   - `02_HistoCartography_Embeddings_EDA_Clustering.ipynb`
   - `03_HistoCartography_Knowledge_Graph_Exports.ipynb`

---

## SAFE_MODE (default)
SAFE_MODE avoids heavy compute:
- sample extraction from CRC zip
- smaller embedding batch sizes
- fewer clustering/EDA computations

To confirm SAFE_MODE:
- open `pipeline_config.yaml` and ensure:
```yaml
project:
  safe_mode: true
```

---

## REAL_DATA mode
To process more data:
```yaml
project:
  safe_mode: false
data:
  max_items_full: null   # all extracted images
```

If you want to use the huge CRC-100K zip:
```yaml
data:
  dataset_key: NCT_CRC_HE_100K
  download:
    allow_large: true
```

---

## Resume after crash / restart
Everything is Drive-first.

Resume strategy:
1) Restart runtime (if needed)
2) Re-run the first 2 cells of `00_HistoCartography_Setup_and_Paths.ipynb`
3) Continue running notebooks; they will reuse checkpoints:
   - `checkpoints/_STATE.json` stores last-success cell and artifacts.
   - each artifact has `*.manifest.json`.

---

## Where to look when something fails
- Logs: `logs/run.jsonl`
- Last successful cell: `checkpoints/_STATE.json` → `last_success`
- EDA outputs: `exports/eda/*.json`

---

## Debug knobs
In `pipeline_config.yaml`:
```yaml
project:
  debug_level: 3
  safe_mode: true
```

Debug level meanings:
- 0: errors only
- 1: standard progress
- 2: verbose
- 3: debug details

---

## Path issues (most common)
If `PROJECT_ROOT` is wrong, set:

```python
import os
os.environ["HISTO_PROJECT_ROOT"] = "/content/drive/MyDrive/mit/histopathology_202502..."
```

and re-run the setup cell.

### Drive mount failed

If you see `ValueError: mount failed` from `drive.mount(...)`:

```python
from google.colab import drive
drive.flush_and_unmount()
drive.mount("/content/drive", force_remount=True, timeout_ms=300000)
```

If it still fails:
- Runtime ▸ Restart runtime, then try again.
- Try an Incognito window / allow third‑party cookies for `colab.research.google.com`.

