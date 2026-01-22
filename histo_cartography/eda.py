from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .runtime import _log


def _safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


# -------------------------
# Core EDA functions
# -------------------------

def eda_label_distribution(items: pd.DataFrame) -> Dict[str, Any]:
    vc = items["label"].value_counts(dropna=False).to_dict()
    return {"label_counts": vc, "n": int(len(items))}


def eda_image_size_summary(items: pd.DataFrame) -> Dict[str, Any]:
    return {
        "n": int(len(items)),
        "width_min": int(items["width"].min()) if len(items) else None,
        "width_max": int(items["width"].max()) if len(items) else None,
        "height_min": int(items["height"].min()) if len(items) else None,
        "height_max": int(items["height"].max()) if len(items) else None,
    }


def eda_missingness(items: pd.DataFrame) -> Dict[str, Any]:
    miss = items.isna().mean().sort_values(ascending=False)
    top = miss.head(50).to_dict()
    return {"missing_rate": {k: _safe_float(v) for k, v in top.items()}}


def eda_morph_summary(morph: pd.DataFrame) -> Dict[str, Any]:
    cols = [c for c in morph.columns if c not in ("item_id",) and morph[c].dtype != "object"]
    desc = morph[cols].describe(include="all").to_dict()
    return {"describe": desc, "n": int(len(morph)), "cols": cols}


def eda_numeric_hist(morph: pd.DataFrame, col: str, bins: int = 30) -> Dict[str, Any]:
    x = morph[col].dropna().astype(float).to_numpy()
    if x.size == 0:
        return {"col": col, "bins": bins, "hist": [], "edges": []}
    hist, edges = np.histogram(x, bins=bins)
    return {"col": col, "bins": bins, "hist": hist.tolist(), "edges": edges.tolist()}


def eda_outlier_iqr(morph: pd.DataFrame, col: str, k: float = 1.5) -> Dict[str, Any]:
    x = morph[col].dropna().astype(float)
    if len(x) < 4:
        return {"col": col, "n": int(len(x)), "outlier_frac": None}
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    out = ((x < lo) | (x > hi)).mean()
    return {"col": col, "n": int(len(x)), "outlier_frac": float(out), "lo": float(lo), "hi": float(hi)}


def eda_morph_corr(morph: pd.DataFrame, cols: Optional[Sequence[str]] = None, method: str = "pearson") -> Dict[str, Any]:
    if cols is None:
        cols = [c for c in morph.columns if c not in ("item_id",) and morph[c].dtype != "object"]
    df = morph[list(cols)].astype(float)
    corr = df.corr(method=method)
    return {"cols": list(cols), "method": method, "corr": corr.round(4).to_dict()}


def eda_morph_group_stats(
    morph: pd.DataFrame,
    items: pd.DataFrame,
    group_col: str = "label",
    feature_cols: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    if feature_cols is None:
        feature_cols = [c for c in morph.columns if c not in ("item_id", "morph_error") and morph[c].dtype != "object"]
    df = morph.merge(items[["item_id", group_col]], on="item_id", how="left")
    agg = df.groupby(group_col)[list(feature_cols)].agg(["count", "mean", "std"]).round(4)
    agg.columns = [f"{a}__{b}" for a, b in agg.columns.to_list()]
    return {"group_col": group_col, "feature_cols": list(feature_cols), "table": agg.reset_index().to_dict(orient="records")}


def eda_embedding_norms(emb: pd.DataFrame) -> Dict[str, Any]:
    vecs = np.array(emb["vector"].tolist(), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1)
    return {
        "n": int(len(emb)),
        "dim": int(emb["dim"].iloc[0]) if len(emb) else 0,
        "norm_min": float(norms.min()) if norms.size else None,
        "norm_max": float(norms.max()) if norms.size else None,
        "norm_mean": float(norms.mean()) if norms.size else None,
        "norm_std": float(norms.std()) if norms.size else None,
    }


def eda_embedding_pca_variance(emb: pd.DataFrame, n_components: int = 50) -> Dict[str, Any]:
    from sklearn.decomposition import PCA  # type: ignore

    X = np.array(emb["vector"].tolist(), dtype=np.float32)
    n = int(min(n_components, X.shape[1], max(2, X.shape[0] - 1)))
    pca = PCA(n_components=n, random_state=1337)
    pca.fit(X)
    return {
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "n_components": n,
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_cumsum": np.cumsum(pca.explained_variance_ratio_).tolist(),
    }


def eda_embedding_pairwise_cosine_stats(emb: pd.DataFrame, sample_pairs: int = 2000) -> Dict[str, Any]:
    rng = np.random.default_rng(1337)
    X = np.array(emb["vector"].tolist(), dtype=np.float32)
    if X.shape[0] < 3:
        return {"note": "not enough rows"}
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n = Xn.shape[0]
    pairs = int(min(sample_pairs, n * (n - 1) // 2))
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    sims = (Xn[i] * Xn[j]).sum(axis=1)
    return {
        "pairs": int(sims.shape[0]),
        "cos_min": float(sims.min()),
        "cos_max": float(sims.max()),
        "cos_mean": float(sims.mean()),
        "cos_std": float(sims.std()),
        "cos_p01": float(np.quantile(sims, 0.01)),
        "cos_p50": float(np.quantile(sims, 0.50)),
        "cos_p99": float(np.quantile(sims, 0.99)),
    }


def eda_embedding_nn_distance_stats(emb: pd.DataFrame, k: int = 5) -> Dict[str, Any]:
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
    except Exception as e:
        return {"error": f"sklearn missing? {e}"}

    X = np.array(emb["vector"].tolist(), dtype=np.float32)
    if X.shape[0] < (k + 1):
        return {"note": "not enough rows"}
    nn = NearestNeighbors(n_neighbors=int(k) + 1, metric="cosine")
    nn.fit(X)
    dists, _ = nn.kneighbors(X, return_distance=True)
    d = dists[:, 1:].reshape(-1)
    return {
        "k": int(k),
        "n": int(X.shape[0]),
        "dist_min": float(d.min()),
        "dist_max": float(d.max()),
        "dist_mean": float(d.mean()),
        "dist_std": float(d.std()),
        "dist_p01": float(np.quantile(d, 0.01)),
        "dist_p50": float(np.quantile(d, 0.50)),
        "dist_p99": float(np.quantile(d, 0.99)),
    }


def eda_cluster_size_distribution(clusters: pd.DataFrame) -> Dict[str, Any]:
    vc = clusters["cluster_id"].value_counts().to_dict()
    sizes = list(vc.values())
    return {
        "n": int(len(clusters)),
        "n_clusters": int(len(vc)),
        "cluster_sizes": vc,
        "size_min": int(min(sizes)) if sizes else None,
        "size_max": int(max(sizes)) if sizes else None,
    }


def eda_cluster_noise_ratio(clusters: pd.DataFrame) -> Dict[str, Any]:
    if "cluster_id" not in clusters.columns:
        return {"error": "missing cluster_id"}
    noise = float((clusters["cluster_id"] == -1).mean())
    return {"noise_ratio": noise}


def eda_cluster_label_purity(items: pd.DataFrame, clusters: pd.DataFrame, label_col: str = "label") -> Dict[str, Any]:
    df = clusters.merge(items[["item_id", label_col]], on="item_id", how="left")
    # per cluster: majority label fraction
    purity_rows = []
    for cid, g in df.groupby("cluster_id"):
        vc = g[label_col].value_counts(dropna=False)
        top = float(vc.iloc[0]) if len(vc) else 0.0
        purity = top / max(1.0, float(len(g)))
        purity_rows.append({"cluster_id": int(cid), "n": int(len(g)), "purity": float(purity), "top_label": str(vc.index[0]) if len(vc) else None})
    pur = pd.DataFrame(purity_rows).sort_values("n", ascending=False)
    weighted = float((pur["purity"] * pur["n"]).sum() / max(1.0, pur["n"].sum())) if len(pur) else None
    return {"weighted_purity": weighted, "table": pur.head(50).to_dict(orient="records")}


def eda_clustering_metrics(fused: pd.DataFrame, clusters: pd.DataFrame) -> Dict[str, Any]:
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # type: ignore

    X = np.array(fused["vector"].tolist(), dtype=np.float32)
    y = clusters.set_index("item_id").loc[fused["item_id"].tolist()]["cluster_id"].to_numpy()

    mask = y != -1
    if mask.sum() < 3 or len(set(y[mask])) < 2:
        return {"note": "Not enough clustered points for metrics"}

    Xm = X[mask]
    ym = y[mask]
    return {
        "silhouette": float(silhouette_score(Xm, ym)),
        "calinski_harabasz": float(calinski_harabasz_score(Xm, ym)),
        "davies_bouldin": float(davies_bouldin_score(Xm, ym)),
        "n_points": int(mask.sum()),
        "n_clusters": int(len(set(ym))),
    }


EDA_FUNCTIONS: Dict[str, Callable[..., Dict[str, Any]]] = {
    "label_distribution": eda_label_distribution,
    "image_size_summary": eda_image_size_summary,
    "missingness": eda_missingness,
    "morph_summary": eda_morph_summary,
    "numeric_hist": eda_numeric_hist,
    "outlier_iqr": eda_outlier_iqr,
    "morph_corr": eda_morph_corr,
    "morph_group_stats": eda_morph_group_stats,
    "embedding_norms": eda_embedding_norms,
    "embedding_pca_variance": eda_embedding_pca_variance,
    "embedding_pairwise_cosine_stats": eda_embedding_pairwise_cosine_stats,
    "embedding_nn_distance_stats": eda_embedding_nn_distance_stats,
    "cluster_size_distribution": eda_cluster_size_distribution,
    "cluster_noise_ratio": eda_cluster_noise_ratio,
    "cluster_label_purity": eda_cluster_label_purity,
    "clustering_metrics": eda_clustering_metrics,
}


# -------------------------
# Catalog runner
# -------------------------

def run_eda_item(name: str, spec: Dict[str, Any], ctx: Dict[str, Any], out_dir: Path) -> Dict[str, Any]:
    fn_name = spec.get("fn")
    if fn_name not in EDA_FUNCTIONS:
        _log().warning("EDA fn not implemented; skipping", extra={"extra": {"name": name, "fn": fn_name}})
        return {"name": name, "status": "skipped", "reason": f"fn_not_implemented:{fn_name}"}

    fn = EDA_FUNCTIONS[fn_name]
    inputs = spec.get("inputs", {})
    kwargs = spec.get("kwargs", {})

    args = {}
    for arg_name, ctx_key in inputs.items():
        if ctx_key not in ctx:
            raise KeyError(f"EDA {name} missing ctx key: {ctx_key}")
        args[arg_name] = ctx[ctx_key]

    res = fn(**args, **kwargs)  # type: ignore[misc]
    payload = {"name": name, "spec": spec, "result": res}

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    _log().info("EDA written", extra={"extra": {"name": name, "out": str(out_path)}})
    return {"name": name, "status": "ok", "out": str(out_path)}


def run_eda_catalog(catalog: Dict[str, Any], ctx: Dict[str, Any], out_dir: Path, *, include: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    items = catalog.get("items", [])
    results: List[Dict[str, Any]] = []
    for spec in items:
        name = spec.get("name")
        if not name:
            continue
        if include is not None and name not in include:
            continue
        try:
            results.append(run_eda_item(name, spec, ctx, out_dir))
        except Exception as e:
            _log().error("EDA failed", extra={"extra": {"name": name, "error": str(e)}})
            results.append({"name": name, "status": "error", "error": str(e)})
    return results
