from __future__ import annotations

"""KG QA utilities.

These functions produce defensible, demo-friendly validation outputs for a
similarity-based Knowledge Graph.

Requested artifacts (written by the notebook, using helpers here):
  - within-label vs cross-label similarity rates
  - similarity weight distribution checks
  - node connectivity diagnostics + outlier candidates
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import json

import numpy as np
import pandas as pd

from .runtime import _log


def _item_eid_to_item_id(entities: pd.DataFrame) -> Dict[str, str]:
    """Map entity_id -> item_id (for ITEM/PATCH nodes)."""
    ent = entities.copy()
    # Backwards compatible: older exports used ITEM; semantic layer might use PATCH.
    mask = ent["entity_type"].astype(str).isin(["ITEM", "PATCH"])
    ent = ent[mask]
    # We store item_id in `name`.
    return {str(eid): str(name) for eid, name in zip(ent["entity_id"].tolist(), ent["name"].tolist())}


def attach_labels_to_similarity_edges(
    *,
    entities: pd.DataFrame,
    edges: pd.DataFrame,
    items: pd.DataFrame,
    label_col: str = "label",
) -> pd.DataFrame:
    """Return SIMILAR_TO edges with src/dst item_id + labels."""
    eid2item = _item_eid_to_item_id(entities)
    sim = edges[edges["rel"].astype(str) == "SIMILAR_TO"].copy()
    if sim.empty:
        return sim

    sim["src_item_id"] = sim["src"].map(eid2item)
    sim["dst_item_id"] = sim["dst"].map(eid2item)
    sim = sim.dropna(subset=["src_item_id", "dst_item_id"]).copy()

    lab = items[["item_id", label_col]].rename(columns={label_col: "label"}).copy()
    lab["label"] = lab["label"].fillna("").astype(str)
    sim = sim.merge(lab.add_prefix("src_"), left_on="src_item_id", right_on="src_item_id", how="left")
    sim = sim.merge(lab.add_prefix("dst_"), left_on="dst_item_id", right_on="dst_item_id", how="left")
    return sim


def within_label_similarity_by_label(sim_edges: pd.DataFrame) -> pd.DataFrame:
    """Compute within-label SIMILAR_TO rates per source label."""
    if sim_edges.empty or "src_label" not in sim_edges.columns or "dst_label" not in sim_edges.columns:
        return pd.DataFrame(columns=["label", "n_edges", "within_label_frac"])

    df = sim_edges.copy()
    df["src_label"] = df["src_label"].fillna("").astype(str)
    df["dst_label"] = df["dst_label"].fillna("").astype(str)
    df["same_label"] = df["src_label"] == df["dst_label"]

    out = (
        df.groupby("src_label")
        .agg(n_edges=("same_label", "size"), within_label_frac=("same_label", "mean"))
        .reset_index()
        .rename(columns={"src_label": "label"})
        .sort_values("n_edges", ascending=False)
        .reset_index(drop=True)
    )
    out["within_label_frac"] = out["within_label_frac"].astype(float)
    return out


def cross_label_top_pairs(sim_edges: pd.DataFrame, *, top_n: int = 25) -> pd.DataFrame:
    """Top cross-label pairs among SIMILAR_TO edges."""
    if sim_edges.empty or "src_label" not in sim_edges.columns or "dst_label" not in sim_edges.columns:
        return pd.DataFrame(columns=["src_label", "dst_label", "n_edges", "rate"])

    df = sim_edges.copy()
    df["src_label"] = df["src_label"].fillna("").astype(str)
    df["dst_label"] = df["dst_label"].fillna("").astype(str)
    df = df[df["src_label"] != df["dst_label"]].copy()
    if df.empty:
        return pd.DataFrame(columns=["src_label", "dst_label", "n_edges", "rate"])

    tab = df.groupby(["src_label", "dst_label"]).size().reset_index(name="n_edges")
    total = float(tab["n_edges"].sum()) if len(tab) else 0.0
    tab["rate"] = tab["n_edges"] / total if total > 0 else 0.0
    tab = tab.sort_values("n_edges", ascending=False).head(int(top_n)).reset_index(drop=True)
    return tab


def similarity_weight_stats(sim_edges: pd.DataFrame) -> Dict[str, Any]:
    """Basic statistics + simple spike detection for similarity weights."""
    if sim_edges.empty:
        return {"n_edges": 0}

    w = pd.to_numeric(sim_edges.get("weight"), errors="coerce").dropna()
    if w.empty:
        return {"n_edges": int(len(sim_edges)), "note": "no_numeric_weights"}

    q = w.quantile([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]).to_dict()
    # spike check: top repeated rounded weights
    rounded = w.round(3)
    vc = rounded.value_counts().head(10)
    spikes = [{"weight_rounded": float(idx), "count": int(cnt)} for idx, cnt in vc.items()]
    return {
        "n_edges": int(len(w)),
        "min": float(w.min()),
        "max": float(w.max()),
        "mean": float(w.mean()),
        "std": float(w.std(ddof=0)),
        "quantiles": {str(k): float(v) for k, v in q.items()},
        "top_weight_spikes_rounded_0p001": spikes,
        "n_unique_weights_rounded_0p001": int(rounded.nunique()),
    }


def node_connectivity_diagnostics(
    *,
    sim_edges: pd.DataFrame,
    items: pd.DataFrame,
    label_col: str = "label",
    low_degree_threshold: int = 3,
    outlier_cross_label_threshold: float = 0.8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (low_degree_nodes, outlier_candidates) tables."""
    # Degrees
    if sim_edges.empty:
        low = items[["item_id", label_col]].rename(columns={label_col: "label"}).copy()
        low["degree"] = 0
        low["in_degree"] = 0
        low["out_degree"] = 0
        return low[low["degree"] <= int(low_degree_threshold)], low.iloc[0:0].copy()

    df = sim_edges.copy()
    # directed degrees
    out_deg = df.groupby("src_item_id").size().rename("out_degree")
    in_deg = df.groupby("dst_item_id").size().rename("in_degree")
    deg = pd.concat([out_deg, in_deg], axis=1).fillna(0).astype(int)
    deg["degree"] = deg["out_degree"] + deg["in_degree"]
    deg = deg.reset_index().rename(columns={"index": "item_id"})
    if "src_item_id" in deg.columns and "item_id" not in deg.columns:
        deg = deg.rename(columns={"src_item_id": "item_id"})

    lab = items[["item_id", label_col]].rename(columns={label_col: "label"}).copy()
    lab["label"] = lab["label"].fillna("").astype(str)
    deg = lab.merge(deg, on="item_id", how="left").fillna({"degree": 0, "in_degree": 0, "out_degree": 0})
    deg[["degree", "in_degree", "out_degree"]] = deg[["degree", "in_degree", "out_degree"]].astype(int)

    low_degree = deg[deg["degree"] <= int(low_degree_threshold)].copy().sort_values(["degree", "item_id"])

    # Outlier candidates: for each source, cross-label fraction of outgoing neighbors
    if "src_label" not in df.columns or "dst_label" not in df.columns:
        return low_degree.reset_index(drop=True), pd.DataFrame(columns=["item_id", "label", "out_degree", "cross_label_frac"])

    tmp = df[["src_item_id", "src_label", "dst_label", "weight"]].copy()
    tmp["src_label"] = tmp["src_label"].fillna("").astype(str)
    tmp["dst_label"] = tmp["dst_label"].fillna("").astype(str)
    tmp["is_cross"] = tmp["src_label"] != tmp["dst_label"]
    per_src = (
        tmp.groupby(["src_item_id", "src_label"])
        .agg(
            out_degree=("is_cross", "size"),
            cross_label_frac=("is_cross", "mean"),
            avg_similarity=("weight", "mean"),
        )
        .reset_index()
        .rename(columns={"src_item_id": "item_id", "src_label": "label"})
    )
    outliers = per_src[
        (per_src["out_degree"] >= 3)
        & (per_src["cross_label_frac"].astype(float) >= float(outlier_cross_label_threshold))
    ].copy().sort_values(["cross_label_frac", "out_degree"], ascending=[False, False])

    return low_degree.reset_index(drop=True), outliers.reset_index(drop=True)


def write_kg_qa_exports(
    *,
    entities: pd.DataFrame,
    edges: pd.DataFrame,
    items: pd.DataFrame,
    out_dir: Path,
    label_col: str = "label",
) -> Dict[str, str]:
    """Compute and write all requested KG QA artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sim = attach_labels_to_similarity_edges(entities=entities, edges=edges, items=items, label_col=label_col)

    # 1) within-label vs cross-label
    within = within_label_similarity_by_label(sim)
    within_path = out_dir / "within_label_similarity_by_label.csv"
    within.to_csv(within_path, index=False)

    cross = cross_label_top_pairs(sim, top_n=25)
    cross_path = out_dir / "cross_label_top_pairs.csv"
    cross.to_csv(cross_path, index=False)

    # 2) edge weight distribution
    stats = similarity_weight_stats(sim)
    stats_path = out_dir / "similarity_weight_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))

    # histogram png
    hist_path = out_dir / "similarity_weight_hist.png"
    try:
        import matplotlib.pyplot as plt

        w = pd.to_numeric(sim.get("weight"), errors="coerce").dropna()
        if len(w):
            plt.figure(figsize=(7, 4))
            plt.hist(w, bins=50)
            plt.title("SIMILAR_TO weight distribution")
            plt.xlabel("similarity")
            plt.ylabel("count")
            plt.tight_layout()
            plt.savefig(hist_path, dpi=160, bbox_inches="tight")
            plt.close()
    except Exception as e:
        _log().warning("Failed to plot similarity histogram", extra={"extra": {"error": str(e)}})

    # 3) node connectivity
    low_deg, outliers = node_connectivity_diagnostics(sim_edges=sim, items=items, label_col=label_col)
    low_path = out_dir / "low_degree_nodes.csv"
    low_deg.to_csv(low_path, index=False)
    out_path = out_dir / "outlier_candidates.csv"
    outliers.to_csv(out_path, index=False)

    return {
        "within_label_similarity_by_label.csv": str(within_path),
        "cross_label_top_pairs.csv": str(cross_path),
        "similarity_weight_stats.json": str(stats_path),
        "similarity_weight_hist.png": str(hist_path),
        "low_degree_nodes.csv": str(low_path),
        "outlier_candidates.csv": str(out_path),
    }
