from __future__ import annotations

"""Relationship / xref verification for Agent 2 outputs.

Goal:
- Provide "glass box" evidence for each proposed cluster link.
- Flag contradictions and low-evidence edges before KG export.

Artifacts written by notebook (using these helpers):
- link_evidence.parquet
- link_flags.parquet
- cluster_links_verified.parquet

This module is deliberately conservative:
- it does not overrule human-in-the-loop decisions
- it marks edges as "needs_more_evidence" rather than forcing a label
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _to_matrix(vectors: Sequence[Any]) -> np.ndarray:
    return np.asarray(list(vectors), dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = float(np.linalg.norm(a)) + 1e-12
    nb = float(np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / (na * nb))


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa = set([str(x) for x in a if str(x)])
    sb = set([str(x) for x in b if str(x)])
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return float(len(sa & sb) / max(1, len(sa | sb)))


def sample_cross_similarity(
    items: pd.DataFrame,
    *,
    cluster_a: int,
    cluster_b: int,
    vector_col: str = "vector",
    cluster_col: str = "cluster_id",
    sample_n: int = 50,
    seed: int = 1337,
) -> Dict[str, Any]:
    """Sample cross-cluster item cosine similarities (diagnostic)."""
    rng = np.random.default_rng(int(seed))
    a = items[items[cluster_col].astype(int) == int(cluster_a)]
    b = items[items[cluster_col].astype(int) == int(cluster_b)]
    if a.empty or b.empty or vector_col not in items.columns:
        return {"pairs": 0, "note": "missing_items_or_vectors"}

    a_idx = rng.choice(len(a), size=min(int(sample_n), len(a)), replace=False)
    b_idx = rng.choice(len(b), size=min(int(sample_n), len(b)), replace=False)
    Xa = _to_matrix(a.iloc[a_idx][vector_col].tolist())
    Xb = _to_matrix(b.iloc[b_idx][vector_col].tolist())

    # Normalize
    Xa = Xa / (np.linalg.norm(Xa, axis=1, keepdims=True) + 1e-12)
    Xb = Xb / (np.linalg.norm(Xb, axis=1, keepdims=True) + 1e-12)

    sims = (Xa @ Xb.T).ravel()
    if sims.size == 0:
        return {"pairs": 0}
    return {
        "pairs": int(sims.size),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "p05": float(np.quantile(sims, 0.05)),
        "p50": float(np.quantile(sims, 0.50)),
        "p95": float(np.quantile(sims, 0.95)),
    }


def build_link_evidence(
    *,
    clusters: pd.DataFrame,
    links: pd.DataFrame,
    centroids: Optional[pd.DataFrame] = None,
    items: Optional[pd.DataFrame] = None,
    cluster_col: str = "cluster_id",
    dataset_col: str = "source",
    label_col: str = "label",
    vector_col_items: str = "vector",
    vector_col_centroids: str = "vector",
    cross_sim_sample_n: int = 50,
    seed: int = 1337,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (evidence_df, flags_df, verified_links_df)."""
    clusters = clusters.copy()
    links = links.copy()

    # centroid lookup
    centroid_vec: Dict[int, np.ndarray] = {}
    if centroids is not None and len(centroids):
        for _, r in centroids.iterrows():
            try:
                cid = int(r[cluster_col])
                centroid_vec[cid] = np.asarray(r[vector_col_centroids], dtype=np.float32)
            except Exception:
                continue

    # cluster membership aggregates from items
    cluster2datasets: Dict[int, List[str]] = {}
    cluster2labels: Dict[int, List[str]] = {}
    if items is not None and len(items):
        for cid, g in items.groupby(cluster_col):
            cid = int(cid)
            if dataset_col in g.columns:
                cluster2datasets[cid] = g[dataset_col].dropna().astype(str).unique().tolist()
            if label_col in g.columns:
                cluster2labels[cid] = g[label_col].dropna().astype(str).unique().tolist()

    evidence_rows: List[Dict[str, Any]] = []
    flag_rows: List[Dict[str, Any]] = []

    # Relationship contradiction thresholds (conservative defaults)
    thr_same_as_min_sim = 0.75
    thr_subtype_min_sim = 0.55
    thr_overlap_min_sim = 0.45

    for _, r in links.iterrows():
        src = int(r["src_cluster_id"])
        dst = int(r["dst_cluster_id"])
        rel = str(r.get("relationship", "related_to")).strip().lower()
        conf = float(r.get("confidence", 0.0) or 0.0)

        # centroid similarity
        sim = r.get("similarity")
        sim_val: Optional[float] = None
        if sim is not None:
            try:
                sim_val = float(sim)
            except Exception:
                sim_val = None
        if sim_val is None and (src in centroid_vec) and (dst in centroid_vec):
            sim_val = _cosine(centroid_vec[src], centroid_vec[dst])

        # overlap evidence
        ds_overlap = _jaccard(cluster2datasets.get(src, []), cluster2datasets.get(dst, []))
        lab_overlap = _jaccard(cluster2labels.get(src, []), cluster2labels.get(dst, []))

        cross_sim = None
        if items is not None and vector_col_items in items.columns:
            cross_sim = sample_cross_similarity(
                items,
                cluster_a=src,
                cluster_b=dst,
                vector_col=vector_col_items,
                cluster_col=cluster_col,
                sample_n=int(cross_sim_sample_n),
                seed=int(seed),
            )

        evidence = {
            "src_cluster_id": src,
            "dst_cluster_id": dst,
            "relationship": rel,
            "confidence": conf,
            "centroid_similarity": sim_val,
            "dataset_overlap_jaccard": ds_overlap,
            "label_overlap_jaccard": lab_overlap,
            "cross_similarity_sample": cross_sim or {"pairs": 0},
        }
        evidence_rows.append(evidence)

        # Flags
        flags: List[str] = []
        needs_more = False

        if sim_val is None:
            flags.append("missing_centroid_similarity")
            needs_more = True
        else:
            if rel == "same_as" and sim_val < thr_same_as_min_sim:
                flags.append(f"contradiction_same_as_low_sim<{thr_same_as_min_sim}")
            if rel == "subtype_of" and sim_val < thr_subtype_min_sim:
                flags.append(f"contradiction_subtype_low_sim<{thr_subtype_min_sim}")
            if rel == "overlaps_with" and sim_val < thr_overlap_min_sim:
                flags.append(f"contradiction_overlap_low_sim<{thr_overlap_min_sim}")
            if rel == "unrelated" and sim_val > 0.75:
                flags.append("contradiction_unrelated_high_sim>0.75")

        # Evidence sufficiency (very conservative)
        if (sim_val is not None and sim_val < 0.35) and ds_overlap < 0.05 and lab_overlap < 0.05:
            needs_more = True
            flags.append("evidence_weak_low_sim_low_overlap")

        flag_rows.append(
            {
                "src_cluster_id": src,
                "dst_cluster_id": dst,
                "relationship": rel,
                "flags": flags,
                "needs_more_evidence": bool(needs_more),
            }
        )

    evidence_df = pd.DataFrame(evidence_rows)
    flags_df = pd.DataFrame(flag_rows)

    verified = links.merge(
        evidence_df[["src_cluster_id", "dst_cluster_id", "centroid_similarity", "dataset_overlap_jaccard", "label_overlap_jaccard"]],
        on=["src_cluster_id", "dst_cluster_id"],
        how="left",
    ).merge(
        flags_df[["src_cluster_id", "dst_cluster_id", "needs_more_evidence", "flags"]],
        on=["src_cluster_id", "dst_cluster_id"],
        how="left",
    )

    return evidence_df, flags_df, verified


def write_link_verification_artifacts(
    *,
    evidence_df: pd.DataFrame,
    flags_df: pd.DataFrame,
    verified_links_df: pd.DataFrame,
    out_dir: Path,
) -> Dict[str, str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    p1 = out_dir / "link_evidence.parquet"
    p2 = out_dir / "link_flags.parquet"
    p3 = out_dir / "cluster_links_verified.parquet"

    evidence_df.to_parquet(p1, index=False)
    flags_df.to_parquet(p2, index=False)
    verified_links_df.to_parquet(p3, index=False)

    return {"link_evidence": str(p1), "link_flags": str(p2), "cluster_links_verified": str(p3)}
