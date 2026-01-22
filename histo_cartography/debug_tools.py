from __future__ import annotations

import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from .runtime import _log, env_fingerprint


def assert_has_columns(df: pd.DataFrame, required: Sequence[str], *, df_name: str = "df") -> None:
    """Raise a helpful KeyError if required columns are missing."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"{df_name} is missing columns: {missing}. Present columns (first 50): {list(df.columns)[:50]}"
        )


def ensure_text_column(
    items: pd.DataFrame,
    *,
    text_col: str = "text",
    alt_cols: Sequence[str] = ("caption", "description", "prompt", "title", "summary", "notes"),
    fallback_to_label: bool = True,
) -> pd.DataFrame:
    """Ensure a usable text column exists (schema-robust).

    Returns a *copy* of the dataframe if any modification is needed.
    """
    if text_col in items.columns:
        out = items.copy()
        out[text_col] = out[text_col].fillna("").astype(str)
        return out

    out = items.copy()
    present_alts = [c for c in alt_cols if c in out.columns]

    if present_alts:
        out[text_col] = (
            out[present_alts]
            .astype(str)
            .agg(" ".join, axis=1)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )
        _log().warning(
            "Text column missing; synthesized from alternates",
            extra={"extra": {"text_col": text_col, "alt_cols": present_alts}},
        )
        return out

    if fallback_to_label and "label" in out.columns:
        out[text_col] = out["label"].fillna("").astype(str)
        _log().warning(
            "Text column missing; synthesized from label",
            extra={"extra": {"text_col": text_col}},
        )
        return out

    # Last resort: minimal metadata template
    for col in ("source", "split", "label"):
        if col not in out.columns:
            out[col] = ""
    out[text_col] = out.apply(
        lambda r: f"Histopathology patch. Source={r.get('source','')}. Split={r.get('split','')}. Label={r.get('label','')}.",
        axis=1,
    )
    _log().warning(
        "Text column missing; synthesized minimal metadata text",
        extra={"extra": {"text_col": text_col}},
    )
    return out


def _ipython_display_available() -> bool:
    try:
        import IPython  # noqa: F401
        from IPython.display import display  # noqa: F401

        return True
    except Exception:
        return False


def display_df(df: pd.DataFrame, *, title: Optional[str] = None, n: int = 10) -> None:
    """Display a dataframe head as a *table* (works in notebooks; falls back to print)."""
    if title:
        print(title)
    print(f"shape={df.shape}")
    if _ipython_display_available():
        from IPython.display import display

        display(df.head(n))
    else:
        print(df.head(n).to_string(index=False))


def show_parquet(path: Path | str, *, title: Optional[str] = None, n: int = 10) -> pd.DataFrame:
    p = Path(path)
    if title is None:
        title = str(p)
    if not p.exists():
        raise FileNotFoundError(f"Parquet not found: {p}")
    df = pd.read_parquet(p)
    display_df(df, title=f"📄 {title}", n=n)
    return df


def list_artifacts(root: Path | str, *, patterns: Sequence[str] = ("*.parquet", "*.png", "*.json", "*.ttl"), max_items: int = 200) -> List[Path]:
    root = Path(root)
    out: List[Path] = []
    for pat in patterns:
        out.extend(sorted(root.rglob(pat)))
    out = sorted(set(out))
    return out[:max_items]


def show_images(img_paths: Sequence[Path | str], *, max_images: int = 24, width: int = 420) -> None:
    """Display saved images (PNG/JPG) inside a notebook."""
    if not _ipython_display_available():
        print("(IPython display not available; skipping image rendering)")
        for p in img_paths[:max_images]:
            print(p)
        return

    from IPython.display import Image, display

    shown = 0
    for p in img_paths:
        if shown >= max_images:
            break
        p = Path(p)
        if not p.exists():
            continue
        try:
            display(Image(filename=str(p), width=width))
            print("Shown:", p)
            shown += 1
        except Exception as e:
            print("Failed to show", p, ":", e)


def make_llm_debug_prompt(
    *,
    cell_id: str,
    error: BaseException,
    code: Optional[str] = None,
    notes: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a copy/paste-ready prompt for any LLM (Gemini / GPT / etc.)."""
    payload: Dict[str, Any] = {
        "cell_id": cell_id,
        "error_type": type(error).__name__,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "env": env_fingerprint(),
        "notes": notes or "",
        "extra": extra or {},
    }
    prompt = [
        "You are a debugging assistant. Fix the bug and propose a minimal patch.",
        "\n---\n",
        "## Context\n",
        json.dumps(payload, indent=2, ensure_ascii=False),
    ]
    if code:
        prompt.extend(["\n---\n", "## Code\n", code])
    return "\n".join(prompt)


@dataclass
class CellRunnerResult:
    ok: bool
    error: Optional[str] = None
    traceback: Optional[str] = None
    prompt: Optional[str] = None


def run_with_debug_prompt(cell_id: str, fn, *, code: Optional[str] = None) -> CellRunnerResult:
    """Run a callable; on exception, return a ready-to-share debug prompt."""
    try:
        fn()
        return CellRunnerResult(ok=True)
    except Exception as e:
        tb = traceback.format_exc()
        prompt = make_llm_debug_prompt(cell_id=cell_id, error=e, code=code)
        _log().error("CellRunner caught exception", extra={"extra": {"cell_id": cell_id, "error": str(e)}})
        return CellRunnerResult(ok=False, error=str(e), traceback=tb, prompt=prompt)
