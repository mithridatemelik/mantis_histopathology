from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .runtime import _log


def _to_matrix(df: pd.DataFrame) -> np.ndarray:
    return np.array(df["vector"].tolist(), dtype=np.float32)


def reduce_to_2d(
    fused: pd.DataFrame,
    *,
    method: str = "umap",
    random_state: int = 1337,
    umap_n_neighbors: int = 15,
    umap_min_dist: float = 0.1,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Dimensionality reduction to 2D with safe fallbacks."""
    X = _to_matrix(fused)
    used = method
    meta: Dict[str, Any] = {"method_requested": method}

    if method.lower() == "umap":
        try:
            import umap  # type: ignore

            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=int(umap_n_neighbors),
                min_dist=float(umap_min_dist),
                random_state=int(random_state),
                metric="cosine",
            )
            Y = reducer.fit_transform(X)
            used = "umap"
            meta.update({"umap_n_neighbors": umap_n_neighbors, "umap_min_dist": umap_min_dist})
        except Exception as e:
            _log().warning("UMAP unavailable; falling back to PCA", extra={"extra": {"error": str(e)}})
            method = "pca"

    if method.lower() == "tsne":
        try:
            from sklearn.manifold import TSNE  # type: ignore

            Y = TSNE(n_components=2, random_state=int(random_state), init="pca", learning_rate="auto").fit_transform(X)
            used = "tsne"
        except Exception as e:
            _log().warning("t-SNE failed; falling back to PCA", extra={"extra": {"error": str(e)}})
            method = "pca"

    if method.lower() == "pca":
        from sklearn.decomposition import PCA  # type: ignore

        Y = PCA(n_components=2, random_state=int(random_state)).fit_transform(X)
        used = "pca"

    out = pd.DataFrame({"item_id": fused["item_id"].tolist(), "x": Y[:, 0], "y": Y[:, 1]})
    meta["method_used"] = used
    _log().info("Reduced to 2D", extra={"extra": {"method": used, "n": len(out)}})
    return out, meta


def cluster_embeddings(
    fused: pd.DataFrame,
    *,
    method: str = "hdbscan",
    random_state: int = 1337,
    kmeans_k: int = 20,
    hdbscan_min_cluster_size: int = 15,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Cluster fused embeddings with safe fallbacks."""
    X = _to_matrix(fused)
    used = method
    meta: Dict[str, Any] = {"method_requested": method}

    labels = None
    if method.lower() == "hdbscan":
        try:
            import hdbscan  # type: ignore

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=int(hdbscan_min_cluster_size),
                metric="euclidean",
                prediction_data=False,
            )
            labels = clusterer.fit_predict(X)
            used = "hdbscan"
            meta.update({"hdbscan_min_cluster_size": hdbscan_min_cluster_size})
        except Exception as e:
            _log().warning("HDBSCAN unavailable; falling back to KMeans", extra={"extra": {"error": str(e)}})
            method = "kmeans"

    if method.lower() == "kmeans":
        from sklearn.cluster import KMeans  # type: ignore

        km = KMeans(n_clusters=int(kmeans_k), random_state=int(random_state), n_init="auto")
        labels = km.fit_predict(X)
        used = "kmeans"
        meta.update({"kmeans_k": kmeans_k})

    if labels is None:
        raise RuntimeError("Clustering produced no labels")

    out = pd.DataFrame({"item_id": fused["item_id"].tolist(), "cluster_id": labels.astype(int).tolist()})
    meta["method_used"] = used

    # Basic diagnostics
    try:
        noise_ratio = float((out["cluster_id"] == -1).mean())
        meta["noise_ratio"] = noise_ratio
    except Exception:
        pass

    _log().info("Clustering done", extra={"extra": {"method": used, "n": len(out), "clusters": int(out["cluster_id"].nunique())}})
    return out, meta


def clustering_metrics(
    fused: pd.DataFrame,
    clusters: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute a few clustering metrics (best-effort)."""
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score  # type: ignore

    X = _to_matrix(fused)
    y = clusters.set_index("item_id").loc[fused["item_id"].tolist()]["cluster_id"].to_numpy()

    # Remove noise label -1 for metrics where needed
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
