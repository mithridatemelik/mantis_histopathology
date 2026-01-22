from __future__ import annotations

import hashlib
import math
import os
import random
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from PIL import Image

from .runtime import _log
from .checkpoint import write_manifest


ZENODO_RECORD_1214456 = "https://zenodo.org/records/1214456"

CRC_ZENODO_FILES: Dict[str, Dict[str, str]] = {
    # CRC-VAL-HE-7K.zip: 800.3 MB, md5:2fd165...
    "CRC_VAL_HE_7K": {
        "url": f"{ZENODO_RECORD_1214456}/files/CRC-VAL-HE-7K.zip?download=1",
        "md5": "2fd1651b4f94ebd818ebf90ad2b6ce06",
    },
    # The two 11.7GB zips are intentionally supported but not recommended by default.
    "NCT_CRC_HE_100K": {
        "url": f"{ZENODO_RECORD_1214456}/files/NCT-CRC-HE-100K.zip?download=1",
        # NOTE: Zenodo page lists md5 6fd702... for this file (v0.1).
        "md5": "6fd702d11df6292bc054397ae038a464",
    },
    "NCT_CRC_HE_100K_NONORM": {
        "url": f"{ZENODO_RECORD_1214456}/files/NCT-CRC-HE-100K-NONORM.zip?download=1",
        # NOTE: Zenodo page lists md5 035777... for this file (v0.1).
        "md5": "035777cf327776a71a05c95da6d6325f",
    },
}

# Additional small/public datasets (optional). These are implemented in a
# dependency-light way so the pipeline can run end-to-end on Colab even when
# the large CRC zips are not practical.

MEDMNIST_DATASETS: Dict[str, Dict[str, str]] = {
    # MedMNIST "PathMNIST" - derived from colorectal cancer histopathology.
    # Key uses a prefix to avoid collisions with other registries.
    "MEDMNIST_PATHMNIST": {
        "medmnist_name": "pathmnist",
        "description": "MedMNIST PathMNIST (colorectal histopathology patches)",
    },
}

HF_DATASETS: Dict[str, Dict[str, str]] = {
    # HuggingFace dataset IDs. These require `datasets`.
    "HF_PCAM": {
        "hf_id": "pcam",
        "description": "PatchCamelyon (PCam) patches (binary)",
        "image_col": "image",
        "label_col": "label",
    },
}


def list_available_datasets() -> pd.DataFrame:
    """Return a simple catalog of datasets supported by this project."""
    rows: List[Dict[str, str]] = []
    for k in CRC_ZENODO_FILES:
        rows.append({"dataset_key": k, "provider": "Zenodo", "type": "crc_zip", "description": "Kather CRC patches (zip)"})
    for k, meta in MEDMNIST_DATASETS.items():
        rows.append({"dataset_key": k, "provider": "medmnist", "type": "medmnist", "description": meta.get("description", "")})
    for k, meta in HF_DATASETS.items():
        rows.append({"dataset_key": k, "provider": "huggingface", "type": "hf", "description": meta.get("description", "")})
    return pd.DataFrame(rows).sort_values(["provider", "dataset_key"]).reset_index(drop=True)

CRC_CLASSES_9 = ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def _infer_crc_class_from_member(member: str) -> Optional[str]:
    """Infer CRC class label from a zip member path.

    The Zenodo CRC zips are not fully consistent in their internal folder layout.
    Common patterns include:

      - ADI/xxx.tif
      - CRC-VAL-HE-7K/ADI/xxx.tif
      - NCT-CRC-HE-100K/NCT-CRC-HE-100K/ADI/xxx.tif

    We therefore look for the first path segment that matches a known CRC class.
    """
    parts = [p for p in str(member).split("/") if p]
    for p in parts:
        if p in CRC_CLASSES_9:
            return p
    return None


def _has_any_image_file(root: Path) -> bool:
    """Return True if `root` contains at least one image file (recursive)."""
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    try:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                return True
    except Exception:
        return False
    return False


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def download_file_wget(url: str, dest: Path, *, resume: bool = True) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget"]
    if resume:
        cmd += ["-c"]
    cmd += ["-O", str(dest), url]
    _log().info("Downloading", extra={"extra": {"url": url, "dest": str(dest)}})
    subprocess.run(cmd, check=True)


def download_crc_zip(
    dataset_key: str,
    raw_dir: Path,
    *,
    verify_md5: bool = True,
    allow_large: bool = False,
) -> Path:
    if dataset_key not in CRC_ZENODO_FILES:
        raise ValueError(f"Unknown dataset_key={dataset_key}. Options: {list(CRC_ZENODO_FILES)}")

    if dataset_key != "CRC_VAL_HE_7K" and not allow_large:
        raise RuntimeError(
            f"{dataset_key} is very large (~11.7GB). Set allow_large=True to download intentionally."
        )

    meta = CRC_ZENODO_FILES[dataset_key]
    url, expected_md5 = meta["url"], meta["md5"]
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    dest = raw_dir / f"{dataset_key}.zip"

    if dest.exists():
        _log().info("Zip already present; skipping download", extra={"extra": {"path": str(dest)}})
    else:
        download_file_wget(url, dest)

    if verify_md5:
        got = md5sum(dest)
        if got != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {dest.name}: expected {expected_md5}, got {got}")
        _log().info("MD5 verified", extra={"extra": {"path": str(dest), "md5": got}})

    return dest


def export_medmnist_to_staging(
    dataset_key: str,
    staging_dir: Path,
    *,
    split: str = "train",
    max_items: int = 512,
    seed: int = 1337,
    overwrite: bool = False,
) -> Path:
    """Download (via medmnist) and export a sample split to an images folder.

    Folder layout produced:
      staging_dir/images/<LABEL_NAME>/<file>.png

    This is intentionally a small/simple helper to let the rest of the pipeline
    run without requiring very large zips.
    """
    if dataset_key not in MEDMNIST_DATASETS:
        raise ValueError(f"Unknown MEDMNIST dataset_key={dataset_key}. Options: {list(MEDMNIST_DATASETS)}")

    images_dir = Path(staging_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    marker = images_dir / "_MEDMNIST_DONE.txt"

    if marker.exists() and not overwrite and _has_any_image_file(images_dir):
        _log().info("MedMNIST export already done; skipping", extra={"extra": {"images_dir": str(images_dir)}})
        return images_dir

    if overwrite and images_dir.exists():
        shutil.rmtree(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)

    try:
        import medmnist  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "medmnist is not installed. In Colab, run: !pip -q install medmnist"
        ) from e

    meta = MEDMNIST_DATASETS[dataset_key]
    name = meta["medmnist_name"]
    info = getattr(medmnist, "INFO", {}).get(name)
    if not info:
        raise RuntimeError(f"medmnist.INFO missing dataset '{name}'. Available: {list(getattr(medmnist,'INFO',{}))[:20]}")

    py_class = info.get("python_class") or info.get("python_class_name")
    if not py_class:
        raise RuntimeError(f"Unexpected medmnist INFO format for {name}: missing python_class")
    DataClass = getattr(medmnist, py_class)

    # MedMNIST splits: train/val/test
    split = str(split)
    ds = DataClass(split=split, download=True)

    imgs = getattr(ds, "imgs", None)
    labels = getattr(ds, "labels", None)
    if imgs is None or labels is None:
        raise RuntimeError(f"Unexpected medmnist dataset object for {name}: missing imgs/labels")

    n = int(len(imgs))
    rng = random.Random(seed)
    idxs = list(range(n))
    if max_items is not None and n > max_items:
        rng.shuffle(idxs)
        idxs = idxs[: int(max_items)]

    label_map = info.get("label", {}) or {}
    # label_map keys are strings in medmnist INFO.
    def _label_name(y: int) -> str:
        return str(label_map.get(str(int(y)), int(y)))

    for i in idxs:
        im = imgs[i]
        y = labels[i]
        # y can be shape (1,) or scalar
        try:
            y_int = int(y[0])  # type: ignore[index]
        except Exception:
            y_int = int(y)

        lbl = _label_name(y_int)
        out_dir = images_dir / lbl
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{name}_{split}_{i}.png"

        # Handle grayscale / RGB arrays
        arr = im
        if hasattr(arr, "numpy"):
            arr = arr.numpy()
        if getattr(arr, "ndim", 0) == 2:
            pil = Image.fromarray(arr)
        else:
            pil = Image.fromarray(arr)
        pil.save(out_path)

    marker.write_text(f"done\ndataset={name}\nsplit={split}\nn_items={len(idxs)}\n")
    _log().info(
        "Exported MedMNIST to images_dir",
        extra={"extra": {"dataset_key": dataset_key, "medmnist_name": name, "split": split, "n": len(idxs), "images_dir": str(images_dir)}},
    )
    return images_dir


def export_hf_to_staging(
    dataset_key: str,
    staging_dir: Path,
    *,
    split: str = "train",
    max_items: int = 512,
    seed: int = 1337,
    overwrite: bool = False,
) -> Path:
    """Download (via 🤗 datasets) and export an image classification dataset to images/.

    Folder layout produced:
      staging_dir/images/<LABEL_NAME>/<file>.png

    Requires: pip install datasets
    """
    if dataset_key not in HF_DATASETS:
        raise ValueError(f"Unknown HF dataset_key={dataset_key}. Options: {list(HF_DATASETS)}")

    images_dir = Path(staging_dir) / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    marker = images_dir / "_HF_DONE.txt"

    if marker.exists() and not overwrite and _has_any_image_file(images_dir):
        _log().info("HF export already done; skipping", extra={"extra": {"images_dir": str(images_dir)}})
        return images_dir

    if overwrite and images_dir.exists():
        shutil.rmtree(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "HuggingFace 'datasets' is not installed. In Colab, run: !pip -q install datasets"
        ) from e

    meta = HF_DATASETS[dataset_key]
    hf_id = meta["hf_id"]
    image_col = meta.get("image_col", "image")
    label_col = meta.get("label_col", "label")

    ds = load_dataset(hf_id, split=split)
    n = int(len(ds))
    rng = random.Random(seed)
    idxs = list(range(n))
    if max_items is not None and n > max_items:
        rng.shuffle(idxs)
        idxs = idxs[: int(max_items)]

    # Optional label names from dataset schema
    label_names = None
    try:
        feat = ds.features.get(label_col)
        label_names = getattr(feat, "names", None)
    except Exception:
        label_names = None

    for i in idxs:
        row = ds[int(i)]
        img = row.get(image_col)
        y = row.get(label_col)
        # img is typically a PIL Image. If it's a dict, the datasets library can usually decode it.
        if img is None:
            continue
        if hasattr(img, "convert"):
            pil = img.convert("RGB")
        else:
            # best-effort conversion
            pil = Image.fromarray(img)

        try:
            y_int = int(y)
        except Exception:
            y_int = 0
        lbl = str(label_names[y_int]) if label_names is not None and y_int < len(label_names) else str(y_int)

        out_dir = images_dir / lbl
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{hf_id}_{split}_{i}.png"
        pil.save(out_path)

    marker.write_text(f"done\nhf_id={hf_id}\nsplit={split}\nn_items={len(idxs)}\n")
    _log().info(
        "Exported HF dataset to images_dir",
        extra={"extra": {"dataset_key": dataset_key, "hf_id": hf_id, "split": split, "n": len(idxs), "images_dir": str(images_dir)}},
    )
    return images_dir


def prepare_dataset_to_staging(
    dataset_key: str,
    *,
    raw_dir: Path,
    staging_dir: Path,
    split: str,
    safe_mode: bool,
    max_items: Optional[int],
    seed: int,
    overwrite: bool,
    verify_md5: bool = True,
    allow_large: bool = False,
    mpp: float = 0.5,
    use_text_modality: bool = True,
    text_template_version: str = "v2_no_label",
) -> Tuple[pd.DataFrame, Path]:
    """Prepare a dataset and return (items_df, images_dir).

    This is the *single* entrypoint used by the notebooks.
    """
    dataset_key = str(dataset_key)
    staging_dir = Path(staging_dir)
    raw_dir = Path(raw_dir)

    # 1) CRC (Zenodo zips)
    if dataset_key in CRC_ZENODO_FILES:
        zip_path = download_crc_zip(dataset_key, raw_dir, verify_md5=verify_md5, allow_large=allow_large)
        images_dir = extract_crc_zip_sample(
            zip_path,
            staging_dir,
            safe_mode=safe_mode,
            max_items=int(max_items or 10**9),
            seed=int(seed),
            overwrite=overwrite,
        )
        items_df = build_items_table_from_images_dir(
            images_dir,
            source=dataset_key,
            split=split,
            mpp=mpp,
            use_text_modality=use_text_modality,
            text_template_version=text_template_version,
        )
        return items_df, images_dir

    # 2) MedMNIST
    if dataset_key in MEDMNIST_DATASETS:
        images_dir = export_medmnist_to_staging(
            dataset_key,
            staging_dir,
            split=split,
            max_items=int(max_items or 10**9),
            seed=int(seed),
            overwrite=overwrite,
        )
        items_df = build_items_table_from_images_dir(
            images_dir,
            source=dataset_key,
            split=split,
            mpp=mpp,
            use_text_modality=use_text_modality,
            text_template_version=text_template_version,
        )
        return items_df, images_dir

    # 3) HuggingFace datasets
    if dataset_key in HF_DATASETS:
        images_dir = export_hf_to_staging(
            dataset_key,
            staging_dir,
            split=split,
            max_items=int(max_items or 10**9),
            seed=int(seed),
            overwrite=overwrite,
        )
        items_df = build_items_table_from_images_dir(
            images_dir,
            source=dataset_key,
            split=split,
            mpp=mpp,
            use_text_modality=use_text_modality,
            text_template_version=text_template_version,
        )
        return items_df, images_dir

    raise ValueError(
        "Unknown dataset_key. Supported keys include CRC_ZENODO_FILES, MEDMNIST_DATASETS, HF_DATASETS. "
        f"Got: {dataset_key}. Try: list_available_datasets()"
    )


def extract_crc_zip_sample(
    zip_path: Path,
    staging_dir: Path,
    *,
    safe_mode: bool = True,
    max_items: int = 512,
    seed: int = 1337,
    overwrite: bool = False,
) -> Path:
    """Extracts a stratified sample from a CRC zip to a staging directory.

    - safe_mode=True: extracts at most max_items images (stratified by class folder name)
    - safe_mode=False: extracts all images (can be many / slow)

    Returns path to extracted root: staging_dir / "images"
    """
    zip_path = Path(zip_path)
    staging_dir = Path(staging_dir)
    images_dir = staging_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    marker = images_dir / "_EXTRACTION_DONE.txt"
    if marker.exists() and not overwrite:
        # A previous run may have written the marker but extracted 0 files if the
        # internal zip structure didn't match our assumptions.
        if _has_any_image_file(images_dir):
            _log().info("Extraction already done; skipping", extra={"extra": {"images_dir": str(images_dir)}})
            return images_dir
        _log().warning(
            "Extraction marker present but no images found; re-extracting",
            extra={"extra": {"images_dir": str(images_dir), "marker": str(marker)}},
        )

    if overwrite and images_dir.exists():
        shutil.rmtree(images_dir)
        images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [
            m
            for m in zf.namelist()
            if str(m).lower().endswith((".tif", ".tiff", ".png", ".jpg", ".jpeg"))
        ]

        # Expected structure: <CLASS>/<filename>.<ext>, but some zips wrap this in
        # a top-level folder (e.g., CRC-VAL-HE-7K/CLASS/file.tif). We therefore
        # infer the class label by searching the path segments.
        by_class: Dict[str, List[str]] = {c: [] for c in CRC_CLASSES_9}
        other: List[str] = []
        for m in members:
            cls = _infer_crc_class_from_member(m)
            if cls is not None:
                by_class[cls].append(m)
            else:
                other.append(m)

        class_counts = {c: len(v) for c, v in by_class.items() if len(v) > 0}
        if not class_counts:
            _log().warning(
                "No CRC class folders detected inside zip; will extract under UNK",
                extra={"extra": {"zip": str(zip_path), "n_members": len(members)}},
            )
        else:
            _log().info(
                "CRC zip class histogram",
                extra={"extra": {"zip": str(zip_path), "class_counts": class_counts}},
            )

        selected: List[str] = []
        if safe_mode:
            available_classes = [c for c, v in by_class.items() if len(v) > 0]
            if available_classes:
                # Stratified: roughly equal per class, limited by class size.
                per_class = max(1, int(math.ceil(max_items / len(available_classes))))
                for c in available_classes:
                    lst = by_class[c]
                    rng.shuffle(lst)
                    selected.extend(lst[:per_class])
                rng.shuffle(selected)
                selected = selected[:max_items]
            else:
                # Fallback: sample arbitrary members so the pipeline can proceed,
                # but labels will be UNK.
                rng.shuffle(members)
                selected = members[:max_items]
        else:
            selected = members

        _log().info(
            "Extracting images",
            extra={"extra": {"zip": str(zip_path), "n_selected": len(selected), "safe_mode": safe_mode}},
        )

        for m in selected:
            cls = _infer_crc_class_from_member(m) or "UNK"
            out_dir = images_dir / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / Path(str(m)).name
            with zf.open(m) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

    marker.write_text(
        "\n".join(
            [
                "done",
                f"zip={zip_path.name}",
                f"safe_mode={safe_mode}",
                f"max_items={max_items}",
                f"n_selected={len(selected)}",
            ]
        )
        + "\n"
    )
    return images_dir


def build_items_table_from_images_dir(
    images_dir: Path,
    *,
    source: str,
    split: str,
    mpp: float = 0.5,
    use_text_modality: bool = True,
    text_template_version: str = "v2_no_label",
) -> pd.DataFrame:
    """Create the canonical items table from an extracted images directory."""
    images_dir = Path(images_dir)
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    rows: List[Dict[str, object]] = []
    for cls_dir in sorted([p for p in images_dir.iterdir() if p.is_dir()]):
        label = cls_dir.name
        for img_path in sorted([p for p in cls_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]):
            try:
                with Image.open(img_path) as im:
                    w, h = im.size
            except Exception:
                w, h = None, None
            item_id = f"{source}::{split}::{label}::{img_path.stem}"

            # Keep the `text` field available for multimodal pipelines, but avoid
            # label leakage by default.
            #
            # NOTE: Many public histopathology patch datasets do not provide rich
            # natural-language metadata. If the only thing we can put in `text` is
            # the ground-truth label, that would contaminate any "unsupervised"
            # fusion/clustering/similarity steps.
            if not use_text_modality:
                text = ""
            else:
                v = str(text_template_version or "").strip().lower()
                if v in {"v1", "v1_with_label", "with_label"}:
                    text = f"Histopathology patch. Dataset={source}. Split={split}. Label={label}."
                elif v in {"v2", "v2_no_label", "no_label"}:
                    # Intentionally omit label.
                    # Keep only non-label metadata that is stable across datasets.
                    # (If you have richer metadata—WSI IDs, patient IDs, clinical
                    # context—prefer adding it here rather than using labels.)
                    w_txt = int(w) if w is not None else -1
                    h_txt = int(h) if h is not None else -1
                    text = (
                        f"Histopathology patch. Dataset={source}. Split={split}. "
                        f"TileSize={w_txt}x{h_txt}. MPP={float(mpp):.3f}."
                    )
                else:
                    raise ValueError(
                        "Unknown text_template_version. Supported: v1_with_label, v2_no_label. "
                        f"Got: {text_template_version!r}"
                    )
            rows.append(
                {
                    "item_id": item_id,
                    "source": source,
                    "split": split,
                    "label": label,
                    "text": text,
                    "image_path": str(img_path),
                    "width": int(w) if w is not None else -1,
                    "height": int(h) if h is not None else -1,
                    "mpp": float(mpp),
                }
            )
    return pd.DataFrame(rows)
