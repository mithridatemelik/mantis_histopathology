from __future__ import annotations

"""EDA report helpers (visual-first, notebook-friendly).

These helpers generate:
- compact overview tables
- common diagnostic plots
- optional heavy HTML profiling (deep mode)

All plotting uses matplotlib (no seaborn) to keep dependencies stable in Colab.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import json
import numpy as np
import pandas as pd


def df_overview_table(df: pd.DataFrame, *, max_cols: int = 40) -> pd.DataFrame:
    """Compact per-column overview: dtype, missing%, nunique, example."""
    rows: List[Dict[str, Any]] = []
    cols = list(df.columns)[: int(max_cols)]
    for c in cols:
        s = df[c]
        ex = None
        try:
            ex = s.dropna().iloc[0] if s.dropna().size else None
        except Exception:
            ex = None
        rows.append(
            {
                "column": str(c),
                "dtype": str(s.dtype),
                "missing_frac": float(s.isna().mean()),
                "nunique": int(s.nunique(dropna=False)),
                "example": str(ex)[:80] if ex is not None else "",
            }
        )
    out = pd.DataFrame(rows).sort_values("missing_frac", ascending=False).reset_index(drop=True)
    return out


def missingness_table(df: pd.DataFrame, *, top_k: int = 30) -> pd.DataFrame:
    miss = df.isna().mean().sort_values(ascending=False).head(int(top_k))
    return miss.reset_index().rename(columns={"index": "column", 0: "missing_frac"})


def plot_missingness(df: pd.DataFrame, *, top_k: int = 30, title: str = "Missingness (top columns)"):
    """Return a matplotlib Figure (bar plot)."""
    import matplotlib.pyplot as plt  # type: ignore

    tab = missingness_table(df, top_k=int(top_k))
    fig = plt.figure(figsize=(8, max(3, 0.25 * len(tab))))
    plt.barh(tab["column"].astype(str).iloc[::-1], tab["missing_frac"].iloc[::-1])
    plt.xlabel("missing fraction")
    plt.title(title)
    plt.tight_layout()
    return fig


def _to_matrix(df: pd.DataFrame, *, vector_col: str = "vector") -> np.ndarray:
    return np.asarray(df[vector_col].tolist(), dtype=np.float32)


def plot_embedding_norms(emb: pd.DataFrame, *, vector_col: str = "vector", title: str = "Embedding norm histogram"):
    import matplotlib.pyplot as plt  # type: ignore

    X = _to_matrix(emb, vector_col=vector_col)
    norms = np.linalg.norm(X, axis=1)
    fig = plt.figure(figsize=(7, 4))
    plt.hist(norms, bins=40)
    plt.xlabel("L2 norm")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_pca_explained_variance(
    emb: pd.DataFrame,
    *,
    vector_col: str = "vector",
    n_components: int = 50,
    title: str = "PCA explained variance (cumulative)",
):
    import matplotlib.pyplot as plt  # type: ignore
    from sklearn.decomposition import PCA  # type: ignore

    X = _to_matrix(emb, vector_col=vector_col)
    n = int(min(n_components, X.shape[1], max(2, X.shape[0] - 1)))
    pca = PCA(n_components=n, random_state=1337)
    pca.fit(X)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    fig = plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(cumsum) + 1), cumsum)
    plt.xlabel("n components")
    plt.ylabel("cumulative explained variance")
    plt.title(title)
    plt.tight_layout()
    meta = {
        "n": int(X.shape[0]),
        "dim": int(X.shape[1]),
        "n_components": int(n),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "explained_variance_ratio_cumsum": cumsum.tolist(),
    }
    return fig, meta


def plot_cosine_similarity_distribution(
    emb: pd.DataFrame,
    *,
    vector_col: str = "vector",
    sample_pairs: int = 2000,
    title: str = "Pairwise cosine similarity (sampled)",
):
    import matplotlib.pyplot as plt  # type: ignore

    rng = np.random.default_rng(1337)
    X = _to_matrix(emb, vector_col=vector_col)
    if X.shape[0] < 3:
        fig = plt.figure(figsize=(7, 4))
        plt.title(title + " (not enough rows)")
        plt.tight_layout()
        return fig, {"pairs": 0}

    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    n = Xn.shape[0]
    pairs = int(min(sample_pairs, n * (n - 1) // 2))
    i = rng.integers(0, n, size=pairs)
    j = rng.integers(0, n, size=pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    sims = (Xn[i] * Xn[j]).sum(axis=1)

    fig = plt.figure(figsize=(7, 4))
    plt.hist(sims, bins=50)
    plt.xlabel("cosine similarity")
    plt.ylabel("count")
    plt.title(title)
    plt.tight_layout()

    meta = {
        "pairs": int(sims.shape[0]),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "p01": float(np.quantile(sims, 0.01)),
        "p50": float(np.quantile(sims, 0.50)),
        "p99": float(np.quantile(sims, 0.99)),
    }
    return fig, meta


def compute_umap_2d(
    emb: pd.DataFrame,
    *,
    vector_col: str = "vector",
    sample_n: int = 5000,
    random_state: int = 1337,
):
    """Return a 2D embedding (DataFrame with x,y plus sampled row indices)."""
    try:
        import umap  # type: ignore
    except Exception:
        # fallback to PCA
        from sklearn.decomposition import PCA  # type: ignore

        X = _to_matrix(emb, vector_col=vector_col)
        n = min(int(sample_n), X.shape[0])
        idx = np.random.default_rng(random_state).choice(X.shape[0], size=n, replace=False)
        Xs = X[idx]
        pca = PCA(n_components=2, random_state=random_state)
        xy = pca.fit_transform(Xs)
        out = pd.DataFrame({"sample_idx": idx.tolist(), "x": xy[:, 0].tolist(), "y": xy[:, 1].tolist()})
        meta = {"method": "pca_fallback", "n": int(n)}
        return out, meta

    X = _to_matrix(emb, vector_col=vector_col)
    n = min(int(sample_n), X.shape[0])
    idx = np.random.default_rng(random_state).choice(X.shape[0], size=n, replace=False)
    Xs = X[idx]
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=random_state)
    xy = reducer.fit_transform(Xs)
    out = pd.DataFrame({"sample_idx": idx.tolist(), "x": xy[:, 0].tolist(), "y": xy[:, 1].tolist()})
    meta = {"method": "umap", "n": int(n), "n_neighbors": 15, "min_dist": 0.1, "metric": "cosine"}
    return out, meta


def plot_scatter_2d(
    xy: pd.DataFrame,
    color: Sequence[Any],
    *,
    title: str = "2D embedding",
    s: int = 8,
    alpha: float = 0.6,
):
    """Scatter plot for a 2D embedding; `color` is aligned with xy rows."""
    import matplotlib.pyplot as plt  # type: ignore

    fig = plt.figure(figsize=(7, 6))
    plt.scatter(xy["x"], xy["y"], s=int(s), alpha=float(alpha), c=pd.Series(color).astype("category").cat.codes)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    return fig


def plot_category_scatter_2d(
    xy: pd.DataFrame,
    categories: Sequence[Any],
    *,
    title: str,
    max_legend: int = 25,
    s: int = 8,
    alpha: float = 0.6,
):
    """Scatter plot with legend groups (limited to max_legend)."""
    import matplotlib.pyplot as plt  # type: ignore

    cats = pd.Series(categories).fillna("").astype(str)
    df = xy.copy()
    df["_cat"] = cats.values
    # Limit legend categories
    top = df["_cat"].value_counts().head(int(max_legend)).index.tolist()
    df["_cat2"] = df["_cat"].where(df["_cat"].isin(top), other="(other)")

    fig = plt.figure(figsize=(7, 6))
    for cat, g in df.groupby("_cat2"):
        plt.scatter(g["x"], g["y"], s=int(s), alpha=float(alpha), label=str(cat))
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=7)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    return fig


def ydata_profiling_report(
    df: pd.DataFrame,
    *,
    out_html: Union[str, Path],
    sample_rows: int = 5000,
    minimal: bool = True,
) -> Optional[str]:
    """Generate an HTML profiling report (optional heavy dependency).

    Returns the path as string if generated, else None.
    """
    try:
        from ydata_profiling import ProfileReport  # type: ignore
    except Exception:
        return None

    out_html = Path(out_html)
    out_html.parent.mkdir(parents=True, exist_ok=True)
    d = df.copy()
    if len(d) > int(sample_rows):
        d = d.sample(n=int(sample_rows), random_state=1337)

    prof = ProfileReport(d, minimal=bool(minimal), explorative=not bool(minimal))
    prof.to_file(str(out_html))
    return str(out_html)
