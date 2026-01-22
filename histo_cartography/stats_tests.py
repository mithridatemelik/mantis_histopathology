from __future__ import annotations

"""Lightweight statistical tests used by health/QA gates.

We prefer SciPy when available, but fall back gracefully when it's not.
All functions are designed for *diagnostics*, not for publishing claims.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np


def _try_import_scipy_stats():
    try:
        from scipy import stats  # type: ignore
        return stats
    except Exception:
        return None


@dataclass
class TestResult:
    name: str
    statistic: Optional[float]
    p_value: Optional[float]
    n_a: int
    n_b: int
    notes: str = ""


def ks_2samp(a: Sequence[float], b: Sequence[float], *, name: str = "ks_2samp") -> TestResult:
    """Kolmogorov–Smirnov test (two-sample) with SciPy fallback."""
    aa = np.asarray(list(a), dtype=float)
    bb = np.asarray(list(b), dtype=float)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    stats = _try_import_scipy_stats()
    if stats is None or len(aa) < 10 or len(bb) < 10:
        return TestResult(name=name, statistic=None, p_value=None, n_a=int(len(aa)), n_b=int(len(bb)), notes="scipy_missing_or_small_n")
    r = stats.ks_2samp(aa, bb)
    return TestResult(name=name, statistic=float(r.statistic), p_value=float(r.pvalue), n_a=int(len(aa)), n_b=int(len(bb)))


def chi2_contingency(table: np.ndarray, *, name: str = "chi2_contingency") -> TestResult:
    """Chi-square test of independence for a contingency table."""
    stats = _try_import_scipy_stats()
    tab = np.asarray(table, dtype=float)
    if stats is None or tab.size == 0 or tab.shape[0] < 2 or tab.shape[1] < 2:
        return TestResult(name=name, statistic=None, p_value=None, n_a=int(tab.shape[0]), n_b=int(tab.shape[1]), notes="scipy_missing_or_bad_shape")
    chi2, p, _, _ = stats.chi2_contingency(tab)
    return TestResult(name=name, statistic=float(chi2), p_value=float(p), n_a=int(tab.shape[0]), n_b=int(tab.shape[1]))


def permutation_test_mean_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_perm: int = 500,
    seed: int = 1337,
    name: str = "perm_mean_diff",
) -> TestResult:
    """Permutation test on difference in means (diagnostic)."""
    rng = np.random.default_rng(int(seed))
    aa = np.asarray(list(a), dtype=float)
    bb = np.asarray(list(b), dtype=float)
    aa = aa[np.isfinite(aa)]
    bb = bb[np.isfinite(bb)]
    if len(aa) < 10 or len(bb) < 10:
        return TestResult(name=name, statistic=None, p_value=None, n_a=int(len(aa)), n_b=int(len(bb)), notes="small_n")

    obs = float(np.mean(aa) - np.mean(bb))
    pooled = np.concatenate([aa, bb], axis=0)
    n_a = len(aa)

    more_extreme = 0
    for _ in range(int(n_perm)):
        rng.shuffle(pooled)
        aa2 = pooled[:n_a]
        bb2 = pooled[n_a:]
        d = float(np.mean(aa2) - np.mean(bb2))
        if abs(d) >= abs(obs):
            more_extreme += 1
    p = (more_extreme + 1) / (int(n_perm) + 1)
    return TestResult(name=name, statistic=obs, p_value=float(p), n_a=int(len(aa)), n_b=int(len(bb)), notes=f"n_perm={int(n_perm)}")


def effective_rank(singular_values: Sequence[float]) -> float:
    """Effective rank from singular values (Roy & Vetterli).

    Returns exp(entropy(p)) where p are normalized singular values.
    """
    s = np.asarray(list(singular_values), dtype=float)
    s = s[s > 0]
    if s.size == 0:
        return 0.0
    p = s / s.sum()
    ent = -np.sum(p * np.log(p + 1e-12))
    return float(np.exp(ent))
