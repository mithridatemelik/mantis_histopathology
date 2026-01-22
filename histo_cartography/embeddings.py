from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .runtime import _log

# We intentionally avoid adding heavy dependencies here.
# - Image embeddings: torchvision ResNet-50 (available in Colab)
# - Text embeddings: TF-IDF + TruncatedSVD (scikit-learn)
# - Morphology features: OpenCV (cv2, available in Colab)


def get_device(prefer_gpu: bool = True) -> str:
    try:
        import torch  # type: ignore

        if prefer_gpu and torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def embed_images_resnet50(
    items: pd.DataFrame,
    *,
    image_col: str = "image_path",
    batch_size: int = 64,
    device: Optional[str] = None,
    num_workers: int = 0,
    max_items: Optional[int] = None,
) -> pd.DataFrame:
    """Compute image embeddings using torchvision ResNet-50.

    Output dataframe columns:
      - item_id, modality, model_id, dim, vector (list[float])
    """
    # Torchvision can be broken in some environments due to binary/ABI mismatches.
    # When that happens, we fall back to a lightweight, dependency-minimal image
    # embedding so the rest of the pipeline can still run end-to-end.
    try:
        import torch  # type: ignore
        from torch.utils.data import DataLoader, Dataset  # type: ignore
        from torchvision import models, transforms  # type: ignore
        from PIL import Image  # type: ignore
    except Exception as e:  # pragma: no cover
        _log().warning(
            "torchvision unavailable; falling back to basic image embeddings",
            extra={"extra": {"error": str(e)}},
        )
        return embed_images_basic(items, image_col=image_col, max_items=max_items)

    device = device or get_device()
    model_id = "torchvision/resnet50_imagenet"

    # Respect max_items for SAFE_MODE
    df = items.copy()
    if max_items is not None and len(df) > max_items:
        df = df.sample(n=max_items, random_state=1337).reset_index(drop=True)

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    class ImgDataset(Dataset):
        def __init__(self, frame: pd.DataFrame):
            self.frame = frame

        def __len__(self):
            return len(self.frame)

        def __getitem__(self, idx: int):
            row = self.frame.iloc[idx]
            path = row[image_col]
            with Image.open(path) as im:
                im = im.convert("RGB")
                x = tfm(im)
            return row["item_id"], x

    ds = ImgDataset(df)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load model and remove classification head
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

    item_ids: List[str] = []
    vecs: List[np.ndarray] = []

    with torch.no_grad():
        for batch in dl:
            ids, x = batch
            x = x.to(device)
            y = model(x)
            y = y.detach().cpu().numpy()
            item_ids.extend(list(ids))
            vecs.extend([v.astype(np.float32) for v in y])

    out = pd.DataFrame(
        {
            "item_id": item_ids,
            "modality": "image",
            "model_id": model_id,
            "dim": int(vecs[0].shape[0]) if vecs else 0,
            "vector": [v.tolist() for v in vecs],
        }
    )
    _log().info("Computed image embeddings", extra={"extra": {"n": len(out), "dim": int(out["dim"].iloc[0]) if len(out) else 0}})
    return out


def embed_images_basic(
    items: pd.DataFrame,
    *,
    image_col: str = "image_path",
    bins: int = 16,
    resize: Tuple[int, int] = (128, 128),
    max_items: Optional[int] = None,
) -> pd.DataFrame:
    """Compute simple image embeddings with only PIL + NumPy.

    This is a *fallback* when torchvision/ResNet is not available.

    Features:
      - per-channel color histograms (bins each for R,G,B)
      - global mean/std for each channel

    Output columns follow the same schema as other embedding functions.
    """
    from PIL import Image  # type: ignore

    if "item_id" not in items.columns:
        raise KeyError("items dataframe must contain 'item_id'")
    if image_col not in items.columns:
        raise KeyError(f"items dataframe must contain '{image_col}'")

    df = items.copy()
    if max_items is not None and len(df) > max_items:
        df = df.sample(n=max_items, random_state=1337).reset_index(drop=True)

    exts = (".tif", ".tiff", ".png", ".jpg", ".jpeg")
    item_ids: List[str] = []
    vecs: List[np.ndarray] = []

    # Histogram bin edges in [0, 256)
    edges = np.linspace(0, 256, bins + 1, dtype=np.float32)

    for _, r in df.iterrows():
        p = Path(str(r[image_col]))
        if not p.exists() or (p.suffix.lower() not in exts):
            continue
        try:
            with Image.open(p) as im:
                im = im.convert("RGB")
                if resize:
                    im = im.resize(resize)
                arr = np.asarray(im, dtype=np.float32)
        except Exception:
            continue

        # arr: (H, W, 3)
        flat = arr.reshape(-1, 3)
        means = flat.mean(axis=0)
        stds = flat.std(axis=0)

        hists: List[np.ndarray] = []
        for c in range(3):
            h, _ = np.histogram(flat[:, c], bins=edges)
            hists.append(h.astype(np.float32))
        feat = np.concatenate(hists + [means.astype(np.float32), stds.astype(np.float32)], axis=0)

        # Normalize (avoid divide by zero)
        norm = float(np.linalg.norm(feat))
        if norm > 0:
            feat = feat / norm

        item_ids.append(str(r["item_id"]))
        vecs.append(feat)

    dim = int(vecs[0].shape[0]) if vecs else 0
    out = pd.DataFrame(
        {
            "item_id": item_ids,
            "modality": "image",
            "model_id": f"basic_rgb_hist/b{bins}_r{resize[0]}x{resize[1]}",
            "dim": dim,
            "vector": [v.tolist() for v in vecs],
        }
    )
    _log().info(
        "Computed basic image embeddings",
        extra={"extra": {"n": len(out), "dim": dim, "bins": int(bins), "resize": list(resize)}},
    )
    return out


def embed_text_tfidf_svd(
    items: pd.DataFrame,
    *,
    text_col: str = "text",
    n_components: int = 256,
    max_features: int = 8192,
    max_items: Optional[int] = None,
    allow_label_fallback: bool = False,
) -> pd.DataFrame:
    """TF-IDF -> TruncatedSVD to get dense text embeddings.

    This function is intentionally lightweight (no transformer dependency) and **schema-robust**.

    If `text_col` is missing, we try to synthesize it from common alternates:
      - caption, description, prompt, title, summary, notes
    If `allow_label_fallback=True`, we fall back to `label`, else empty strings.

    Output dataframe columns:
      - item_id, modality, model_id, dim, vector (list[float])
    """
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    from sklearn.decomposition import TruncatedSVD  # type: ignore
    from sklearn.preprocessing import Normalizer  # type: ignore

    if "item_id" not in items.columns:
        raise KeyError("items dataframe must contain 'item_id'")

    df = items.copy()
    if max_items is not None and len(df) > max_items:
        df = df.sample(n=max_items, random_state=1337).reset_index(drop=True)

    # Robustness: older checkpoints / alternate schemas may not have the expected text column.
    if text_col not in df.columns:
        alt_cols = [c for c in ("caption", "description", "prompt", "title", "summary", "notes") if c in df.columns]
        if alt_cols:
            df[text_col] = (
                df[list(alt_cols)]
                .astype(str)
                .agg(" ".join, axis=1)
                .str.replace(r"\s+", " ", regex=True)
                .str.strip()
            )
            _log().warning(
                "Text column missing; synthesized from alternates",
                extra={"extra": {"text_col": text_col, "alt_cols": list(alt_cols)}},
            )
        elif allow_label_fallback and "label" in df.columns:
            df[text_col] = df["label"].fillna("").astype(str)
            _log().warning(
                "Text column missing; synthesized from label (label leakage risk)",
                extra={"extra": {"text_col": text_col, "allow_label_fallback": True}},
            )
        else:
            df[text_col] = ""
            _log().warning("Text column missing; synthesized empty text", extra={"extra": {"text_col": text_col}})

    texts = df[text_col].fillna("").astype(str).tolist()

    vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2), min_df=1)
    X = vec.fit_transform(texts)  # sparse

    # Handle extremely small vocabularies (e.g., empty/constant text)
    if X.shape[1] <= 1:
        Z = np.zeros((len(df), 1), dtype=np.float32)
        n_eff = 1
    else:
        # TruncatedSVD requires n_components < n_features
        n_eff = int(min(n_components, max(2, X.shape[1] - 1)))
        svd = TruncatedSVD(n_components=n_eff, random_state=1337)
        Z = svd.fit_transform(X)  # dense
        Z = Normalizer(copy=False).fit_transform(Z).astype(np.float32)

    out = pd.DataFrame(
        {
            "item_id": df["item_id"].tolist(),
            "modality": "text",
            "model_id": f"tfidf_svd/{max_features}_svd{n_eff}",
            "dim": int(n_eff),
            "vector": [z.tolist() for z in Z],
        }
    )
    _log().info("Computed text embeddings", extra={"extra": {"n": len(out), "dim": int(n_eff)}})
    return out


def compute_morphology_features(
    items: pd.DataFrame,
    *,
    image_col: str = "image_path",
    max_items: Optional[int] = None,
) -> pd.DataFrame:
    """Compute lightweight morphology-ish features from RGB patches.

    These are NOT clinical-grade features. They are quick proxies useful for:
      - data QC (blur, brightness, background)
      - simple morphology signals (tissue coverage proxy)
    """
    import cv2  # type: ignore

    df = items.copy()
    if max_items is not None and len(df) > max_items:
        df = df.sample(n=max_items, random_state=1337).reset_index(drop=True)

    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        p = str(r[image_col])
        img = cv2.imread(p)
        if img is None:
            rows.append({"item_id": r["item_id"], "morph_error": 1})
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape

        # Basic intensity stats
        mean_rgb = img.reshape(-1, 3).mean(axis=0)
        std_rgb = img.reshape(-1, 3).std(axis=0)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mean_gray = float(gray.mean())
        std_gray = float(gray.std())

        # Tissue coverage proxy via Otsu (background tends to be bright)
        _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        tissue_mask = (gray < thr).astype(np.uint8)
        tissue_coverage = float(tissue_mask.mean())

        # Blur proxy: variance of Laplacian
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        blur_var_lap = float(lap.var())

        # Saturation proxy
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat_mean = float(hsv[..., 1].mean())
        sat_std = float(hsv[..., 1].std())

        rows.append(
            {
                "item_id": r["item_id"],
                "morph_error": 0,
                "mean_r": float(mean_rgb[0]),
                "mean_g": float(mean_rgb[1]),
                "mean_b": float(mean_rgb[2]),
                "std_r": float(std_rgb[0]),
                "std_g": float(std_rgb[1]),
                "std_b": float(std_rgb[2]),
                "mean_gray": mean_gray,
                "std_gray": std_gray,
                "tissue_coverage": tissue_coverage,
                "blur_var_lap": blur_var_lap,
                "sat_mean": sat_mean,
                "sat_std": sat_std,
                "width": int(w),
                "height": int(h),
            }
        )

    feats = pd.DataFrame(rows)
    _log().info("Computed morphology features", extra={"extra": {"n": len(feats), "cols": len(feats.columns)}})
    return feats


def embed_morphology_features(
    morph_df: pd.DataFrame,
    *,
    feature_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Turn morphology feature table into an embedding vector per item."""
    from sklearn.preprocessing import StandardScaler  # type: ignore

    df = morph_df.copy()
    if feature_cols is None:
        # Default: all numeric columns except item_id
        feature_cols = [c for c in df.columns if c not in ("item_id",) and df[c].dtype != "object"]

    X = df[feature_cols].fillna(0.0).astype(float).to_numpy()
    X = StandardScaler().fit_transform(X).astype(np.float32)

    out = pd.DataFrame(
        {
            "item_id": df["item_id"].tolist(),
            "modality": "morphology",
            "model_id": f"morph_standardized/{len(feature_cols)}",
            "dim": int(X.shape[1]),
            "vector": [x.tolist() for x in X],
        }
    )
    _log().info("Embedded morphology features", extra={"extra": {"n": len(out), "dim": int(X.shape[1])}})
    return out


def fuse_embeddings_concat_pca(
    emb_dfs: Sequence[pd.DataFrame],
    *,
    target_dim: int = 256,
    model_id: str = "fuse/concat_pca",
) -> pd.DataFrame:
    """Concat embeddings for each item_id, then PCA to target_dim."""
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore

    # Build dense matrix per modality
    # Assumes each emb_df has unique item_id rows.
    mats: List[np.ndarray] = []
    item_ids = emb_dfs[0]["item_id"].tolist() if emb_dfs else []
    for edf in emb_dfs:
        # Align order
        edf = edf.set_index("item_id").loc[item_ids].reset_index()
        X = np.array(edf["vector"].tolist(), dtype=np.float32)
        mats.append(X)

    Xcat = np.concatenate(mats, axis=1)
    Xcat = StandardScaler(with_mean=True, with_std=True).fit_transform(Xcat).astype(np.float32)

    d = int(min(target_dim, Xcat.shape[1]))
    pca = PCA(n_components=d, random_state=1337)
    Z = pca.fit_transform(Xcat).astype(np.float32)

    out = pd.DataFrame(
        {
            "item_id": item_ids,
            "model_id": model_id,
            "dim": d,
            "vector": [z.tolist() for z in Z],
        }
    )
    _log().info("Fused embeddings (concat+pca)", extra={"extra": {"n": len(out), "dim": d}})
    return out
