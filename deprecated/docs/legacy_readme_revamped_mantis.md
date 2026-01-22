# Revamped Mantis Export (Colab)

This project includes a **revamped** notebook:

- `04_HistoCartography_Mantis_CSV_Export.ipynb`

It is designed to produce a **healthy Mantis knowledge graph**:

- **7-level supervised ontology** (`cluster_l1..cluster_l7`) with no blanks
- Backward-compatible fields used by legacy Mantis spaces (`title`, `semantic_text`, `x`, `y`, etc.)
- **Hybrid embeddings**:
  - Visual: ResNet50 (torchvision)
  - Semantic: Sentence-Transformers (`all-MiniLM-L6-v2`)
  - Fusion: concatenation + PCA reduction (default 512-d)
- **Semantic metadata** beyond ontology:
  - `metadata` column is JSON with dataset source, license, citation, notes, raw label, etc.

## Datasets included (Unified Atlas)

Minimum required sources:
- CRC_VAL_HE_7K (Zenodo)
- NCT_CRC_HE_100K (Zenodo; large)
- MEDMNIST_PATHMNIST (MedMNIST)
- HF_PCAM (HuggingFace)

Additional open datasets for diversity:
- HF_LC25000 (lung + colon)
- HF_BACH (breast)
- HF_BREAKHIS_RCL_7500 (breast)
- ORCA_ORAL_ANNOTATED_100 (oral cancer)

## How to run (recommended)

1. Open `04_HistoCartography_Mantis_CSV_Export.ipynb` in **Google Colab**.
2. Run cells in order.
3. The output CSV will be written to:
   - `/content/histo_data/exports/mantis_unified_atlas_multimodal.csv`

### Notes

- `NCT_CRC_HE_100K` is ~11.7GB. If Colab disk is limited, disable it in the config cell.
- The ORCA dataset is downloaded via a **Google Drive folder link**. If that link changes, update the `gdrive_url` in the dataset registry cell.

## Upload to Mantis

The final cell provides an **upload template** (endpoint may differ by deployment).
Set your token as an environment variable in Colab:

```bash
%env MANTIS_TOKEN=...your token...
```

Then run the upload cell after updating the `MANTIS_UPLOAD_URL`.
