from __future__ import annotations

"""Visualization helpers (Colab-first, parquet-first).

Design goals:
- Every plot should be:
  1) saved to exports/<stage>/plots/
  2) displayed inline in the notebook immediately
- Keep notebook cells clean:
  - one visualization per cell
  - short interpretation in a following markdown cell (done in notebooks)

This module is intentionally lightweight and does not depend on seaborn.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import pandas as pd


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_dir(p: Union[str, Path]) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_and_display(fig, out_path: Union[str, Path], *, dpi: int = 160, close: bool = True) -> Path:
    """Save a matplotlib Figure and display it inline.

    Important: this function displays exactly ONE output (the figure).
    Do NOT print a bunch of extra text in the same cell if you want "one output per cell".
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save first (so the saved artifact matches the displayed figure).
    fig.savefig(out_path, dpi=int(dpi), bbox_inches="tight")

    # Display inline (best-effort; safe outside notebooks).
    try:
        from IPython.display import display  # type: ignore

        display(fig)
    except Exception:
        pass

    if close:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.close(fig)
        except Exception:
            pass

    return out_path


def display_image(path: Union[str, Path]):
    """Display a saved image inline (PNG/JPG/etc.)."""
    path = Path(path)
    try:
        from IPython.display import Image, display  # type: ignore

        display(Image(filename=str(path)))
    except Exception:
        # Non-notebook environment; do nothing.
        return


@dataclass
class VizRecord:
    stage: str
    plot_id: str
    title: str
    path: str
    created_at_utc: str
    tags_json: str = "[]"
    is_core: bool = True
    notes: str = ""


def register_plot(
    viz_records: List[Dict[str, Any]],
    *,
    stage: str,
    plot_id: str,
    title: str,
    path: Union[str, Path],
    tags: Optional[Sequence[str]] = None,
    is_core: bool = True,
    notes: str = "",
) -> None:
    tags = list(tags or [])
    viz_records.append(
        {
            "stage": str(stage),
            "plot_id": str(plot_id),
            "title": str(title),
            "path": str(Path(path)),
            "created_at_utc": _utc_now_iso(),
            "tags_json": json.dumps(tags, ensure_ascii=False),
            "is_core": bool(is_core),
            "notes": str(notes or ""),
        }
    )


def viz_records_to_df(viz_records: List[Dict[str, Any]]) -> pd.DataFrame:
    cols = ["stage", "plot_id", "title", "path", "created_at_utc", "tags_json", "is_core", "notes"]
    if not viz_records:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(viz_records)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].copy()


def write_viz_index(
    viz_records: List[Dict[str, Any]],
    *,
    out_parquet: Union[str, Path],
    out_csv: Optional[Union[str, Path]] = None,
) -> Path:
    """Write viz index to parquet (+ optional csv)."""
    out_parquet = Path(out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df = viz_records_to_df(viz_records)
    df.to_parquet(out_parquet, index=False)
    if out_csv is not None:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
    return out_parquet
