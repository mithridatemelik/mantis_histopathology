from __future__ import annotations

"""Critic / health gates.

This module implements "hard validation gates" and "soft warning signals"
used throughout the notebook DAG.

Philosophy ("glass box"):
- Prefer explicit, inspectable checks.
- If evidence is insufficient, do not force conclusions.
- Emit actionable warnings and point to the relevant artifacts/plots.

The critic outputs are structured so notebooks can:
- render a compact table inline (PEEP/POST/CHECKPOINT)
- write full JSON to exports/<stage>/qa/
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import json
import numpy as np
import pandas as pd


@dataclass
class CriticResult:
    stage: str
    gate: str
    passed: bool
    fails: List[str]
    warnings: List[str]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def warnings_count(self) -> int:
        return int(len(self.warnings))

    @property
    def fails_count(self) -> int:
        return int(len(self.fails))


def _finite_ratio(x: np.ndarray) -> float:
    if x.size == 0:
        return 1.0
    return float(np.isfinite(x).mean())


def _vector_lengths(v: pd.Series, *, sample_n: int = 5000) -> Tuple[int, int]:
    """Return (min_len, max_len) over a sample to keep it cheap."""
    if v.empty:
        return 0, 0
    if len(v) > sample_n:
        v = v.sample(n=int(sample_n), random_state=1337)
    lens = v.map(lambda z: len(z) if isinstance(z, (list, tuple, np.ndarray)) else -1)
    return int(lens.min()), int(lens.max())


def _embedding_numeric_health(v: pd.Series, *, sample_n: int = 2000) -> Dict[str, Any]:
    """Return a few numeric diagnostics for embedding vectors."""
    if v.empty:
        return {"n_checked": 0}
    if len(v) > sample_n:
        v = v.sample(n=int(sample_n), random_state=1337)
    try:
        X = np.asarray(v.tolist(), dtype=np.float32)
    except Exception:
        return {"n_checked": int(len(v)), "note": "vector_to_matrix_failed"}

    norms = np.linalg.norm(X, axis=1)
    return {
        "n_checked": int(X.shape[0]),
        "dim": int(X.shape[1]) if X.ndim == 2 else None,
        "finite_ratio": _finite_ratio(X),
        "norm_min": float(np.nanmin(norms)) if norms.size else None,
        "norm_max": float(np.nanmax(norms)) if norms.size else None,
        "norm_mean": float(np.nanmean(norms)) if norms.size else None,
        "norm_std": float(np.nanstd(norms)) if norms.size else None,
        "duplicate_vector_frac_sample": float(pd.Series([tuple(np.round(row, 4)) for row in X]).duplicated().mean()) if X.size else None,
    }


def run_critic(
    *,
    df: pd.DataFrame,
    stage: str,
    gate: str,
    required_cols: Sequence[str] = (),
    id_col: Optional[str] = None,
    min_rows: int = 10,
    key_nonnull_cols: Sequence[str] = (),
    vector_col: Optional[str] = None,
    expected_vector_dim: Optional[int] = None,
    finite_cols: Sequence[str] = (),
    max_missing_frac_soft: float = 0.25,
) -> CriticResult:
    """Run a set of health checks and return a structured result."""
    fails: List[str] = []
    warns: List[str] = []
    metrics: Dict[str, Any] = {}

    n = int(len(df))
    metrics["rows"] = n
    metrics["cols"] = int(len(df.columns))

    # Required columns (hard)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        fails.append(f"missing_required_columns: {missing}")

    # Min rows (hard)
    if n < int(min_rows):
        fails.append(f"too_few_rows: {n} < {int(min_rows)}")

    # ID uniqueness (hard-ish)
    if id_col and id_col in df.columns:
        nunique = int(df[id_col].nunique(dropna=False))
        metrics[f"{id_col}_nunique"] = nunique
        if nunique != n:
            fails.append(f"id_not_unique: {id_col} unique={nunique} rows={n}")

    # Key non-null (hard)
    for c in key_nonnull_cols:
        if c in df.columns:
            miss = float(df[c].isna().mean())
            metrics[f"missing_frac__{c}"] = miss
            if miss > 0:
                fails.append(f"key_column_has_nulls: {c} missing_frac={miss:.3f}")

    # Generic missingness (soft)
    miss_rates = df.isna().mean().sort_values(ascending=False)
    top_miss = miss_rates.head(15).to_dict()
    metrics["top_missing_frac"] = {str(k): float(v) for k, v in top_miss.items()}
    for c, frac in top_miss.items():
        if float(frac) >= float(max_missing_frac_soft):
            warns.append(f"high_missingness: {c} missing_frac={float(frac):.2f}")

    # Finite numeric columns (hard-ish)
    for c in finite_cols:
        if c in df.columns:
            arr = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
            finite = _finite_ratio(arr)
            metrics[f"finite_ratio__{c}"] = finite
            if finite < 0.999:
                fails.append(f"non_finite_values: {c} finite_ratio={finite:.3f}")

    # Embedding vector checks
    if vector_col and vector_col in df.columns:
        mn, mx = _vector_lengths(df[vector_col])
        metrics["vector_len_min_sample"] = mn
        metrics["vector_len_max_sample"] = mx
        if mn <= 0 or mx <= 0:
            fails.append(f"vector_length_invalid: min={mn} max={mx}")
        if mn != mx:
            fails.append(f"vector_length_inconsistent: min={mn} max={mx}")
        if expected_vector_dim is not None and mn != int(expected_vector_dim):
            fails.append(f"vector_dim_mismatch: expected={int(expected_vector_dim)} got={mn}")
        metrics["embedding_health_sample"] = _embedding_numeric_health(df[vector_col])

        # Soft: collapsed embeddings (very low norm variance)
        h = metrics["embedding_health_sample"]
        if isinstance(h, dict) and h.get("norm_std") is not None and float(h["norm_std"]) < 1e-3:
            warns.append("collapsed_embeddings: norm_std_sample < 1e-3")

    passed = len(fails) == 0
    return CriticResult(stage=str(stage), gate=str(gate), passed=bool(passed), fails=fails, warnings=warns, metrics=metrics)


def write_critic_report(result: CriticResult, out_json: Union[str, Path]) -> Path:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return out_json


def critic_result_table(result: CriticResult) -> pd.DataFrame:
    """Compact 1-row table for inline display."""
    return pd.DataFrame(
        [
            {
                "stage": result.stage,
                "gate": result.gate,
                "passed": bool(result.passed),
                "fails_count": int(result.fails_count),
                "warnings_count": int(result.warnings_count),
            }
        ]
    )


def critic_issues_table(result: CriticResult) -> pd.DataFrame:
    """Exploded list of fails/warnings for inline display."""
    rows: List[Dict[str, Any]] = []
    for f in result.fails:
        rows.append({"level": "FAIL", "issue": str(f)})
    for w in result.warnings:
        rows.append({"level": "WARN", "issue": str(w)})
    return pd.DataFrame(rows)
