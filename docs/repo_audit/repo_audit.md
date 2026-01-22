# repo audit (refactor + deprecations)

This document records what was changed during the **DAG + parquet-first + agentic** refactor.

## goals of the audit

- remove or quarantine unused / legacy artifacts
- ensure no secrets are present in the repo
- enforce lowercase, meaningful naming
- ensure notebooks form a clear parquet-first DAG

---

## security scan

A scan was performed for common secret patterns (OpenAI `sk-...`, Hugging Face `hf_...`, `Bearer ...`, and `api_key=...` style strings).

- ✅ no hardcoded tokens were retained
- ✅ notebooks do not print tokens
- ✅ secrets are expected only via env vars or Colab Secrets

See: `docs/security/secrets_and_tokens.md`.

---

## dag refactor

New primary notebooks live under `notebooks/`:

- `00_download_data.ipynb`
- `01_prepare_dataset.ipynb`
- `02_compute_embeddings.ipynb`
- `03_cluster_embeddings.ipynb`
- `04_agent1_cluster_cleanup.ipynb`
- `05_agent2_cluster_linking.ipynb`
- `06_build_knowledge_graph.ipynb`

Each notebook:
- does **one job**
- writes outputs to `exports/stage_XX_*`
- appends to `exports/artifact_manifest.parquet`
- reads only upstream parquet outputs (no re-upload / re-download)

---

## deprecated items

Legacy files that are no longer part of the primary workflow were moved under `deprecated/`.

### legacy notebooks
Moved to: `deprecated/notebooks/`

- `legacy_00_setup_and_paths.ipynb`
- `legacy_01_data_ingestion_staging.ipynb`
- `legacy_02_embeddings_eda_clustering.ipynb`
- `legacy_03_knowledge_graph_exports.ipynb`
- `legacy_04_mantis_csv_export.ipynb`
- `legacy_04k_agentic_mantis_export.ipynb`
- `legacy_mantis_colab_deploy_safe.ipynb`

Reason:
- notebooks were not tied together as a parquet-first DAG
- intermediate artifacts were not consistently referenced downstream
- naming was inconsistent / hard to maintain

### legacy docs / reports
Moved to: `deprecated/docs/`

- `legacy_readme_original.md`
- `legacy_readme_revamped_mantis.md`
- `legacy_runbook.md`
- `legacy_health_report.json`
- `legacy_health_report.md`

### config
Moved to: `deprecated/configs/`

- `eda_catalog.yaml`

Reason:
- not required by the new DAG; kept only for reference.

### large legacy exports
Large CSV exports were not included in the refactor output zip to keep the repo lightweight.
Small samples are provided in `examples/`:

- `examples/legacy_mantis_export_agentic_sample.csv`
- `examples/legacy_mantis_workshop_ready_sample.csv`

---

## removed noise

Removed repository noise files:

- `__pycache__/`
- `*.pyc`
- `.DS_Store`

---

## naming rules

- new files are **lowercase**
- notebook names use `00_...` stage prefixes
- no `v3_fix_patch_final` style naming remains in primary paths

### legacy scripts
Moved to: `deprecated/scripts/`

- `legacy_regenerate_exports.py`

Reason:
- relied on legacy checkpoint paths; not used by the new parquet-first DAG.


## eda catalog

- Promoted `configs/eda_catalog.yaml` to an active config (previously in `deprecated/configs/`).
- Stage notebooks 01–03 now optionally run EDA and write `eda/*.json` + quick plots inside their stage export folders.
