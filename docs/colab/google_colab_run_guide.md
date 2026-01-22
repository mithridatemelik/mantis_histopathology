# google colab run guide

This project is designed to run **end-to-end** in Google Colab.

## 1) get the repo into colab

### option a: upload to google drive (simplest)
1. In Drive, create a folder, e.g. `MyDrive/projects/mantis_histopathology/`
2. Upload the repo contents into that folder.
3. In Colab, open a notebook from that folder.

### option b: clone from github
In a Colab cell:

```bash
!git clone <your-repo-url> /content/mantis_histopathology
```

If you clone into `/content`, artifacts will be lost when the runtime resets. Prefer Drive for persistence.

---

## 2) mount google drive (recommended)

Most notebooks start with a "Colab setup" cell that runs:

```python
from google.colab import drive
drive.mount("/content/drive")
```

After mount, your Drive files live under:

- `/content/drive/MyDrive/...`

---

## 3) set the project root (important)

The notebooks try to auto-detect the repo root, but you can pin it using:

```python
import os
os.environ["HISTO_PROJECT_ROOT"] = "/content/drive/MyDrive/projects/mantis_histopathology"
```

The helper `histo_cartography.paths.resolve_project_root()` looks for:

- `pipeline_config.yaml`
- `label_taxonomy.yaml`

in order to find the correct root folder.

---

## 4) configure secrets safely

Open:

**Runtime → Secrets → Add a new secret**

Add:

- `OPENAI_API_KEY` (required for agentic stages 04–06)

Optional:

- `HF_TOKEN`
- `MANTIS_TOKEN`

The notebooks will read secrets from:
- environment variables (`os.environ`)
- or Colab secrets via `google.colab.userdata.get(...)`

✅ The notebooks do **not** print tokens.

See `docs/security/secrets_and_tokens.md`.

---

## 4.5) choose datasets (single vs multi-dataset)

The default configuration (`pipeline_config.yaml`) enables **multi-dataset** staging via:

- `data.dataset_keys: [...]`

To run a single dataset instead, edit `pipeline_config.yaml`:

- set `data.dataset_keys` to a single entry, or
- comment it out and set `data.dataset_key`.

⚠️ Some dataset keys are very large and require `data.download.allow_large: true`.

## 5) run the dag in order

Run notebooks in this exact order:

1. `00_download_data.ipynb`
2. `01_prepare_dataset.ipynb`
3. `02_compute_embeddings.ipynb`
4. `03_cluster_embeddings.ipynb`
5. `04_agent1_cluster_cleanup.ipynb`
6. `05_agent2_cluster_linking.ipynb`
7. `06_build_knowledge_graph.ipynb`

### parquet-first contracts
Each notebook writes outputs to:

- `exports/stage_XX_<name>/...`

Downstream notebooks **only read** these parquet outputs.  
This means you can restart later stages without re-uploading or re-downloading.

---

## 6) idempotency / skipping work

Each stage checks whether expected parquet outputs exist.

- If outputs exist, it **skips** the expensive step.
- If you want to recompute, set:

```python
FORCE_REBUILD = True
```

inside that notebook.

---

## 7) quick diagnostics

If a notebook fails:

- confirm `HISTO_PROJECT_ROOT` is correct
- confirm the stage output directories exist
- confirm parquet files are readable

You can also inspect:

- `exports/artifact_manifest.parquet`

which logs every artifact written.

---

## 8) gpu is optional

The pipeline can run on CPU.

If GPU is available, stage 02 may use it for image embeddings (torchvision ResNet-50).
GPU is not required.
