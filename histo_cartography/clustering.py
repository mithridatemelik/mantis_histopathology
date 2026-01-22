from __future__ import annotations

"""Clustering utilities (UMAP → HDBSCAN sweep + KMeans baseline).

This module exists to make clustering results:
  - less fragile (hyperparameter sweep + explicit selection)
  - easier to audit (CSV/MD outputs + per-run plots)
  - more defensible in a demo (purity + noise + internal metrics)

Design goals:
  - Keep the pipeline runnable even when optional deps are missing.
  - Avoid hard-coding label information into embeddings (see datasets.py).

Optional dependencies:
  - umap-learn (import name: `umap`)
  - hdbscan
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import json

import numpy as np
import pandas as pd

from .runtime import _log


def _to_matrix(fused: pd.DataFrame) -> np.ndarray:
    return np.array(fused["vector"].tolist(), dtype=np.float32)


def l2_normalize(X: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def _try_import_umap() -> Optional[Any]:
    try:
        import umap  # type: ignore

        return umap
    except Exception:
        return None


def _try_import_hdbscan() -> Optional[Any]:
    try:
        import hdbscan  # type: ignore

        return hdbscan
    except Exception:
        return None


def compute_umap(
    X: np.ndarray,
    *,
    n_neighbors: int,
    min_dist: float,
    n_components: int,
    metric: str = "cosine",
    random_state: int = 1337,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute a UMAP embedding, falling back to PCA when UMAP isn't available."""
    meta: Dict[str, Any] = {
        "method_requested": "umap",
        "metric": metric,
        "n_neighbors": int(n_neighbors),
        "min_dist": float(min_dist),
        "n_components": int(n_components),
    }

    umap_mod = _try_import_umap()
    if umap_mod is None:
        # Safe fallback so notebooks remain runnable.
        from sklearn.decomposition import PCA  # type: ignore

        Y = PCA(n_components=int(n_components), random_state=int(random_state)).fit_transform(X)
        meta.update({"method_used": "pca_fallback", "note": "umap-learn not installed"})
        _log().warning("umap-learn not installed; using PCA fallback", extra={"extra": meta})
        return Y.astype(np.float32), meta

    reducer = umap_mod.UMAP(
        n_components=int(n_components),
        n_neighbors=int(n_neighbors),
        min_dist=float(min_dist),
        random_state=int(random_state),
        metric=str(metric),
    )
    Y = reducer.fit_transform(X)
    meta.update({"method_used": "umap"})
    return np.asarray(Y, dtype=np.float32), meta


def run_hdbscan(
    X: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Run HDBSCAN (if available)."""
    meta = {
        "method_requested": "hdbscan",
        "min_cluster_size": int(min_cluster_size),
        "min_samples": None if min_samples is None else int(min_samples),
        "metric": str(metric),
    }

    hdbscan_mod = _try_import_hdbscan()
    if hdbscan_mod is None:
        raise RuntimeError(
            "hdbscan is not installed. Install with: pip install hdbscan\n"
            "(In Colab: !pip -q install hdbscan)"
        )

    clusterer = hdbscan_mod.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=None if min_samples is None else int(min_samples),
        metric=str(metric),
        prediction_data=False,
    )
    labels = clusterer.fit_predict(X)
    meta.update({"method_used": "hdbscan"})
    return labels.astype(int), meta


def run_kmeans(
    X: np.ndarray,
    *,
    k: int,
    random_state: int = 1337,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.cluster import KMeans  # type: ignore

    km = KMeans(n_clusters=int(k), random_state=int(random_state), n_init="auto")
    labels = km.fit_predict(X)
    return labels.astype(int), {"method_used": "kmeans", "k": int(k)}


def dominant_label_fraction_per_cluster(
    clusters: pd.Series,
    labels: pd.Series,
    *,
    noise_label: int = -1,
) -> pd.DataFrame:
    """Return per-cluster dominant label fraction and cluster size."""
    df = pd.DataFrame({"cluster_id": clusters.astype(int), "label": labels.astype(str)})
    df = df[df["cluster_id"].notna()].copy()
    out_rows: List[Dict[str, Any]] = []
    for cid, g in df.groupby("cluster_id"):
        cid_int = int(cid)
        if cid_int == int(noise_label):
            continue
        vc = g["label"].value_counts(dropna=False)
        n = int(vc.sum())
        dom = str(vc.index[0]) if len(vc) else ""
        dom_frac = float(vc.iloc[0] / max(1, n)) if len(vc) else 0.0
        out_rows.append({"cluster_id": cid_int, "n_items": n, "dominant_label": dom, "dominant_frac": dom_frac})
    return pd.DataFrame(out_rows).sort_values(["n_items", "dominant_frac"], ascending=[False, False]).reset_index(drop=True)


def weighted_purity(clusters: pd.Series, labels: pd.Series, *, noise_label: int = -1) -> float:
    """Weighted purity (dominant label fraction), excluding noise."""
    per = dominant_label_fraction_per_cluster(clusters, labels, noise_label=noise_label)
    if per.empty:
        return float("nan")
    return float((per["dominant_frac"] * per["n_items"]).sum() / max(1, per["n_items"].sum()))


def evaluate_clustering(
    *,
    X: np.ndarray,
    clusters: pd.Series,
    labels: Optional[pd.Series] = None,
    noise_label: int = -1,
) -> Dict[str, Any]:
    """Compute clustering diagnostics (best-effort).

    - Noise ratio
    - # clusters (excluding noise)
    - cluster size list
    - weighted purity (if labels provided)
    - silhouette + Davies-Bouldin (if applicable)
    """
    from sklearn.metrics import davies_bouldin_score, silhouette_score  # type: ignore

    y = clusters.astype(int).to_numpy()
    noise_ratio = float((y == int(noise_label)).mean()) if len(y) else float("nan")

    # sizes (including noise for visibility)
    sizes = pd.Series(y).value_counts().sort_index()
    sizes_dict = {int(k): int(v) for k, v in sizes.items()}

    # internal metrics exclude noise
    mask = y != int(noise_label)
    y_nn = y[mask]
    X_nn = X[mask]
    n_points = int(mask.sum())
    n_clusters = int(len(set(y_nn))) if n_points else 0

    sil = float("nan")
    db = float("nan")
    if n_points >= 3 and n_clusters >= 2:
        try:
            sil = float(silhouette_score(X_nn, y_nn))
        except Exception:
            sil = float("nan")
        try:
            db = float(davies_bouldin_score(X_nn, y_nn))
        except Exception:
            db = float("nan")

    out: Dict[str, Any] = {
        "noise_ratio": noise_ratio,
        "n_clusters_ex_noise": n_clusters,
        "cluster_sizes": sizes_dict,
        "silhouette": sil,
        "davies_bouldin": db,
        "n_points_ex_noise": n_points,
    }

    if labels is not None:
        per = dominant_label_fraction_per_cluster(clusters, labels, noise_label=noise_label)
        out["weighted_purity"] = float((per["dominant_frac"] * per["n_items"]).sum() / max(1, per["n_items"].sum())) if not per.empty else float("nan")
        # For auditability: keep per-cluster dominant label fractions too.
        out["cluster_dominant_label_fractions"] = (
            per[["cluster_id", "n_items", "dominant_label", "dominant_frac"]].to_dict(orient="records")
            if not per.empty
            else []
        )
    return out


def _score_run(metrics: Dict[str, Any]) -> float:
    """Single scalar used for 'best run' selection.

    This is a heuristic, not a scientific objective:
      - prefer lower noise
      - prefer higher purity
      - prefer higher silhouette / lower DB where available
    """
    noise = float(metrics.get("noise_ratio") or 0.0)
    purity = float(metrics.get("weighted_purity") or 0.0)
    sil = metrics.get("silhouette")
    db = metrics.get("davies_bouldin")

    score = (1.0 - noise) * purity
    if sil is not None and np.isfinite(sil):
        score *= max(0.0, float(sil))
    if db is not None and np.isfinite(db):
        score *= 1.0 / (1.0 + max(0.0, float(db)))
    return float(score)


def _ensure_dir(p: Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, sort_keys=True)
    except Exception:
        return json.dumps(str(obj))


def plot_run_artifacts(
    *,
    umap2d: pd.DataFrame,
    clusters: pd.DataFrame,
    labels: pd.Series,
    out_dir: Path,
    run_name: str,
    noise_label: int = -1,
) -> None:
    """Save required audit plots for one run."""
    import matplotlib.pyplot as plt

    out_dir = _ensure_dir(Path(out_dir))
    merged = umap2d.merge(clusters, on="item_id", how="left")
    merged["label"] = labels.values

    def _savefig(name: str) -> None:
        p = out_dir / name
        plt.savefig(p, dpi=160, bbox_inches="tight")

    # 1) scatter by predicted cluster
    plt.figure(figsize=(7, 6))
    cid = merged["cluster_id"].astype(int)
    is_noise = cid == int(noise_label)
    plt.scatter(merged.loc[~is_noise, "x"], merged.loc[~is_noise, "y"], s=8, c=cid.loc[~is_noise])
    if is_noise.any():
        plt.scatter(merged.loc[is_noise, "x"], merged.loc[is_noise, "y"], s=12, c="k", alpha=0.35, label="noise")
        plt.legend(loc="best")
    plt.title(f"UMAP2D — predicted clusters ({run_name})")
    plt.xlabel("x")
    plt.ylabel("y")
    _savefig(f"{run_name}__umap2d_by_cluster.png")
    plt.close()

    # 2) scatter by ground-truth label
    plt.figure(figsize=(7, 6))
    # map labels to ints for consistent coloring without specifying a palette
    lab_codes = pd.Categorical(merged["label"].astype(str)).codes
    plt.scatter(merged["x"], merged["y"], s=8, c=lab_codes)
    plt.title(f"UMAP2D — ground-truth labels ({run_name})")
    plt.xlabel("x")
    plt.ylabel("y")
    _savefig(f"{run_name}__umap2d_by_label.png")
    plt.close()

    # 3) cluster size histogram
    vc = merged["cluster_id"].astype(int).value_counts().sort_values(ascending=False)
    plt.figure(figsize=(8, 4))
    plt.bar([str(i) for i in vc.index.tolist()], vc.values.tolist())
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Cluster sizes ({run_name})")
    plt.ylabel("n items")
    _savefig(f"{run_name}__cluster_size_hist.png")
    plt.close()


def hdbscan_parameter_sweep(
    *,
    fused: pd.DataFrame,
    labels: pd.Series,
    out_dir: Path,
    umap_n_neighbors: Sequence[int] = (10, 20, 50),
    umap_min_dist: Sequence[float] = (0.0, 0.1, 0.3),
    umap_n_components_cluster: int = 10,
    hdbscan_min_cluster_size: Sequence[int] = (5, 8, 10, 15),
    hdbscan_min_samples: Sequence[Optional[int]] = (None, 5, 10),
    random_state: int = 1337,
    max_noise_ratio: float = 0.40,
    make_plots: bool = True,
    noise_label: int = -1,
) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]], Optional[pd.DataFrame]]:
    """Run a UMAP→HDBSCAN sweep and write audit artifacts.

    Returns:
      - results_df (one row per run)
      - best_run dict (config+metrics) or None
      - best_clusters_df (item_id, cluster_id) or None
    """
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    plots_dir = _ensure_dir(out_dir / "plots")

    X0 = _to_matrix(fused)
    X0 = l2_normalize(X0)

    # If optional deps aren't installed, write a stub summary and return.
    if _try_import_hdbscan() is None:
        stub_csv = out_dir / "hdbscan_sweep.csv"
        stub_md = out_dir / "hdbscan_sweep_summary.md"
        pd.DataFrame(
            [
                {
                    "status": "skipped",
                    "reason": "hdbscan_not_installed",
                    "install": "pip install hdbscan umap-learn",
                }
            ]
        ).to_csv(stub_csv, index=False)
        stub_md.write_text(
            "# HDBSCAN sweep (skipped)\n\n"
            "This environment does not have **hdbscan** installed, so the UMAP→HDBSCAN sweep was skipped.\n\n"
            "Install dependencies and re-run notebook 02:\n\n"
            "```bash\n"
            "pip install umap-learn hdbscan\n"
            "```\n"
        )
        _log().warning("HDBSCAN sweep skipped (hdbscan not installed)", extra={"extra": {"out_dir": str(out_dir)}})
        return pd.read_csv(stub_csv), None, None

    rows: List[Dict[str, Any]] = []
    best: Optional[Dict[str, Any]] = None
    best_clusters: Optional[pd.DataFrame] = None

    run_idx = 0
    for nn in umap_n_neighbors:
        for md in umap_min_dist:
            # UMAP embedding for clustering (n_components >= 10 helps density)
            emb_cluster, umap_meta = compute_umap(
                X0,
                n_neighbors=int(nn),
                min_dist=float(md),
                n_components=int(umap_n_components_cluster),
                metric="cosine",
                random_state=int(random_state),
            )

            # UMAP2D for plots
            emb2d, _ = compute_umap(
                X0,
                n_neighbors=int(nn),
                min_dist=float(md),
                n_components=2,
                metric="cosine",
                random_state=int(random_state),
            )
            umap2d_df = pd.DataFrame({"item_id": fused["item_id"].tolist(), "x": emb2d[:, 0], "y": emb2d[:, 1]})

            for mcs in hdbscan_min_cluster_size:
                for ms in hdbscan_min_samples:
                    run_idx += 1
                    run_name = f"run{run_idx:03d}__nn{nn}__md{md}__mcs{mcs}__ms{ms if ms is not None else 'None'}"
                    try:
                        y_pred, hdb_meta = run_hdbscan(
                            emb_cluster,
                            min_cluster_size=int(mcs),
                            min_samples=None if ms is None else int(ms),
                            metric="euclidean",
                        )
                    except Exception as e:
                        # Failed run: record and continue.
                        rows.append(
                            {
                                "run_name": run_name,
                                "status": "error",
                                "error": str(e),
                                "umap_n_neighbors": int(nn),
                                "umap_min_dist": float(md),
                                "umap_n_components_cluster": int(umap_n_components_cluster),
                                "hdbscan_min_cluster_size": int(mcs),
                                "hdbscan_min_samples": None if ms is None else int(ms),
                            }
                        )
                        continue

                    clus_df = pd.DataFrame({"item_id": fused["item_id"].tolist(), "cluster_id": y_pred.tolist()})
                    metrics = evaluate_clustering(X=emb_cluster, clusters=clus_df["cluster_id"], labels=labels, noise_label=noise_label)
                    score = _score_run(metrics)

                    row = {
                        "run_name": run_name,
                        "status": "ok",
                        "score": score,
                        "umap_n_neighbors": int(nn),
                        "umap_min_dist": float(md),
                        "umap_n_components_cluster": int(umap_n_components_cluster),
                        "hdbscan_min_cluster_size": int(mcs),
                        "hdbscan_min_samples": None if ms is None else int(ms),
                        **{k: metrics.get(k) for k in ["noise_ratio", "n_clusters_ex_noise", "silhouette", "davies_bouldin", "weighted_purity", "n_points_ex_noise"]},
                        "cluster_sizes_json": _safe_json(metrics.get("cluster_sizes", {})),
                        "cluster_dominant_label_fractions_json": _safe_json(metrics.get("cluster_dominant_label_fractions", [])),
                    }
                    rows.append(row)

                    if make_plots:
                        try:
                            plot_run_artifacts(
                                umap2d=umap2d_df,
                                clusters=clus_df,
                                labels=labels,
                                out_dir=plots_dir,
                                run_name=run_name,
                                noise_label=noise_label,
                            )
                        except Exception as e:
                            _log().warning("Plotting failed for run", extra={"extra": {"run_name": run_name, "error": str(e)}})

                    # best-run selection (prefer runs under noise threshold; else best score overall)
                    is_eligible = np.isfinite(metrics.get("noise_ratio", np.nan)) and float(metrics["noise_ratio"]) <= float(max_noise_ratio)
                    if best is None:
                        best = {"run_name": run_name, "eligible": bool(is_eligible), "score": score, **row}
                        best_clusters = clus_df
                    else:
                        # If current best is ineligible but this is eligible, promote.
                        if (not bool(best.get("eligible"))) and is_eligible:
                            best = {"run_name": run_name, "eligible": True, "score": score, **row}
                            best_clusters = clus_df
                        # Else compare among same eligibility class.
                        elif bool(best.get("eligible")) == bool(is_eligible) and score > float(best.get("score", -1)):
                            best = {"run_name": run_name, "eligible": bool(is_eligible), "score": score, **row}
                            best_clusters = clus_df

    results_df = pd.DataFrame(rows)
    results_path = out_dir / "hdbscan_sweep.csv"
    results_df.to_csv(results_path, index=False)

    # Summary markdown (top 10 runs)
    summary_path = out_dir / "hdbscan_sweep_summary.md"
    if results_df.empty:
        summary_path.write_text("# HDBSCAN sweep\n\nNo runs were executed.\n")
    else:
        ok_df = results_df[results_df["status"] == "ok"].copy()
        ok_df = ok_df.sort_values("score", ascending=False)
        top = ok_df.head(10)

        lines: List[str] = []
        lines.append("# HDBSCAN sweep summary")
        lines.append("")
        lines.append(f"Total runs attempted: {len(results_df)}")
        lines.append(f"Successful runs: {int((results_df['status']=='ok').sum())}")
        lines.append("")
        if best is not None:
            lines.append("## Best run")
            lines.append("")
            lines.append(f"- run_name: `{best['run_name']}`")
            lines.append(f"- eligible (noise <= {max_noise_ratio:.2f}): `{bool(best.get('eligible'))}`")
            lines.append(f"- noise_ratio: `{best.get('noise_ratio')}`")
            lines.append(f"- n_clusters_ex_noise: `{best.get('n_clusters_ex_noise')}`")
            lines.append(f"- weighted_purity: `{best.get('weighted_purity')}`")
            lines.append(f"- silhouette: `{best.get('silhouette')}`")
            lines.append(f"- davies_bouldin: `{best.get('davies_bouldin')}`")
            lines.append("")
            lines.append("### Best config")
            lines.append("")
            lines.append("```json")
            cfg_obj = {
                "umap": {
                    "n_neighbors": int(best.get("umap_n_neighbors")),
                    "min_dist": float(best.get("umap_min_dist")),
                    "n_components_cluster": int(best.get("umap_n_components_cluster")),
                    "metric": "cosine",
                },
                "hdbscan": {
                    "min_cluster_size": int(best.get("hdbscan_min_cluster_size")),
                    "min_samples": best.get("hdbscan_min_samples"),
                    "metric": "euclidean",
                },
            }
            lines.append(json.dumps(cfg_obj, indent=2))
            lines.append("```")
            lines.append("")

        if not top.empty:
            lines.append("## Top runs (by score)")
            lines.append("")
            # keep it readable
            show_cols = [
                "run_name",
                "score",
                "noise_ratio",
                "n_clusters_ex_noise",
                "weighted_purity",
                "silhouette",
                "davies_bouldin",
                "umap_n_neighbors",
                "umap_min_dist",
                "hdbscan_min_cluster_size",
                "hdbscan_min_samples",
            ]
            lines.append(top[show_cols].to_markdown(index=False))
            lines.append("")

        summary_path.write_text("\n".join(lines) + "\n")

    # Persist best-run config/metrics as JSON too.
    if best is not None:
        (out_dir / "best_run.json").write_text(json.dumps(best, indent=2, sort_keys=True))

    return results_df, best, best_clusters


def kmeans_baseline_exports(
    *,
    fused: pd.DataFrame,
    labels: pd.Series,
    out_dir: Path,
    k: int = 9,
    random_state: int = 1337,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run KMeans baseline (no noise label) + write requested exports."""
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    X = l2_normalize(_to_matrix(fused))
    y_pred, meta = run_kmeans(X, k=int(k), random_state=int(random_state))
    assign = pd.DataFrame({"item_id": fused["item_id"].tolist(), "cluster_id": y_pred.tolist()})
    assign_path = out_dir / "kmeans_assignments.parquet"
    assign.to_parquet(assign_path, index=False)

    # Eval: purity + confusion
    per = dominant_label_fraction_per_cluster(assign["cluster_id"], labels)
    pur = float((per["dominant_frac"] * per["n_items"]).sum() / max(1, per["n_items"].sum())) if not per.empty else float("nan")

    conf = pd.crosstab(labels.astype(str), assign["cluster_id"].astype(int))

    md_lines: List[str] = []
    md_lines.append("# KMeans baseline evaluation")
    md_lines.append("")
    md_lines.append(f"- k: **{int(k)}**")
    md_lines.append("- embedding: **L2-normalized fused embeddings**")
    md_lines.append("")
    md_lines.append(f"## Weighted purity: `{pur:.4f}`")
    md_lines.append("")
    md_lines.append("## Cluster label composition (dominant label)")
    md_lines.append("")
    if per.empty:
        md_lines.append("(no clusters)\n")
    else:
        md_lines.append(per.to_markdown(index=False))
        md_lines.append("")
    md_lines.append("## Confusion (label × cluster counts)")
    md_lines.append("")
    md_lines.append(conf.to_markdown())
    md_lines.append("")
    (out_dir / "kmeans_eval.md").write_text("\n".join(md_lines) + "\n")

    meta_out = {**meta, "weighted_purity": pur, "n_clusters": int(assign["cluster_id"].nunique())}
    (out_dir / "kmeans_meta.json").write_text(json.dumps(meta_out, indent=2, sort_keys=True))
    return assign, meta_out


def label_composition_table(
    *,
    clusters: pd.DataFrame,
    items: pd.DataFrame,
    label_col: str = "label",
    noise_label: int = -1,
) -> pd.DataFrame:
    """Return per-cluster per-label counts + fractions.

    Output columns:
      - cluster_id, label, n_items, frac
    """
    lab = items[["item_id", label_col]].rename(columns={label_col: "label"}).copy()
    lab["label"] = lab["label"].fillna("").astype(str)
    merged = clusters.merge(lab, on="item_id", how="left")
    merged["cluster_id"] = merged["cluster_id"].astype(int)
    tab = (
        merged.groupby(["cluster_id", "label"]).size().reset_index(name="n_items")
    )
    # fractions per cluster
    totals = tab.groupby("cluster_id")["n_items"].transform("sum").replace(0, np.nan)
    tab["frac"] = (tab["n_items"] / totals).fillna(0.0)
    return tab.sort_values(["cluster_id", "n_items"], ascending=[True, False]).reset_index(drop=True)


def name_clusters_by_label_composition(
    *,
    clusters: pd.DataFrame,
    items: pd.DataFrame,
    label_col: str = "label",
    label_name_map: Optional[Dict[str, str]] = None,
    min_label_fraction_for_name: float = 0.10,
    noise_label: int = -1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assign defensible cluster names based on label composition.

    Rules:
      - Never name noise (-1) as a tissue label.
      - Include a label in the cluster name only if its fraction >= threshold.
      - Always include the top label (even if below threshold) *only as* `C{id}`
        when composition is weak/ambiguous.

    Returns:
      - clusters_named (clusters dataframe with cluster_name, purity, entropy)
      - cluster_summary_df (one row per cluster)
      - composition_df (cluster × label counts + fractions)
    """
    label_name_map = label_name_map or {}
    composition = label_composition_table(clusters=clusters, items=items, label_col=label_col, noise_label=noise_label)

    # Compute purity/entropy per cluster (excluding noise label id, not label strings)
    pivot = composition.pivot_table(index="cluster_id", columns="label", values="n_items", fill_value=0, aggfunc="sum")
    totals = pivot.sum(axis=1).replace(0, np.nan)
    p = pivot.div(totals, axis=0).fillna(0.0)
    entropy = (-(p * np.log(np.maximum(p, 1e-12)))).sum(axis=1).fillna(0.0)
    purity = (pivot.max(axis=1) / totals).fillna(0.0)

    names: Dict[int, str] = {}
    top_labels: Dict[int, List[str]] = {}

    for cid in pivot.index.tolist():
        cid_int = int(cid)
        if cid_int == int(noise_label):
            names[cid_int] = "Noise / outliers"
            top_labels[cid_int] = ["-"]
            continue
        row = p.loc[cid]
        # Sort labels by fraction
        sorted_fracs = row.sort_values(ascending=False)
        # Top label always exists (could be "" if unlabeled)
        lbl1 = str(sorted_fracs.index[0]) if len(sorted_fracs) else ""
        frac1 = float(sorted_fracs.iloc[0]) if len(sorted_fracs) else 0.0
        lbl2 = str(sorted_fracs.index[1]) if len(sorted_fracs) > 1 else ""
        frac2 = float(sorted_fracs.iloc[1]) if len(sorted_fracs) > 1 else 0.0

        # Decide which labels are allowed into the name
        parts: List[str] = []
        if frac1 >= float(min_label_fraction_for_name) and lbl1:
            pretty1 = label_name_map.get(lbl1, lbl1)
            parts.append(f"{pretty1} ({lbl1})" if pretty1 != lbl1 else lbl1)
        if frac2 >= float(min_label_fraction_for_name) and lbl2:
            pretty2 = label_name_map.get(lbl2, lbl2)
            parts.append(f"{pretty2} ({lbl2})" if pretty2 != lbl2 else lbl2)

        if parts:
            names[cid_int] = f"C{cid_int}: " + " + ".join(parts)
        else:
            # Avoid misleading names when composition is weak/ambiguous.
            names[cid_int] = f"C{cid_int}: (mixed)"

        top_labels[cid_int] = [lbl1] + ([lbl2] if lbl2 else [])

    # Build summary table
    summary = (
        pd.DataFrame(
            {
                "cluster_id": [int(c) for c in pivot.index.tolist()],
                "n_items": [int(totals.loc[c]) if np.isfinite(totals.loc[c]) else 0 for c in pivot.index.tolist()],
                "purity": [float(purity.loc[c]) if np.isfinite(purity.loc[c]) else 0.0 for c in pivot.index.tolist()],
                "entropy": [float(entropy.loc[c]) if np.isfinite(entropy.loc[c]) else 0.0 for c in pivot.index.tolist()],
                "cluster_name": [names[int(c)] for c in pivot.index.tolist()],
            }
        )
        .sort_values("n_items", ascending=False)
        .reset_index(drop=True)
    )

    clusters_named = clusters.merge(summary[["cluster_id", "cluster_name", "purity", "entropy"]], on="cluster_id", how="left")
    return clusters_named, summary, composition
