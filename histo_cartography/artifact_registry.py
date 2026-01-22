from __future__ import annotations

"""Parquet-first artifact registry / manifest.

This module keeps a lightweight run log in:

  exports/artifact_manifest.parquet

Design goals:
- Colab-friendly (no external DB required)
- cheap fingerprints for large artifacts (sample-hash)
- supports idempotent DAG execution (stages can decide to skip if inputs unchanged)

The manifest serves two purposes:
1) artifact-level logging (one row per output parquet/csv/html/plot index)
2) stage-run logging (one row per stage run, artifact="__stage_run__")

The *stage-run* rows contain the critical DAG wiring metadata required by the prompt:
- inputs + input hashes
- outputs
- schema_version
- timestamp
- warnings_count / fails_count
- runtime_sec
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import pandas as pd


DEFAULT_MANIFEST_REL_PATH = Path("exports") / "artifact_manifest.parquet"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fingerprint_file(path: Union[str, Path], *, sample_bytes: int = 65536) -> Dict[str, Any]:
    """Cheap file fingerprint: size + mtime + sha256(sample head+tail).

    We avoid hashing entire multi-GB artifacts. For parquet artifacts this is
    typically enough to detect changes in practice.
    """
    p = Path(path)
    if not p.exists() or not p.is_file():
        return {"path": str(p), "exists": False}

    st = p.stat()
    size = int(st.st_size)
    mtime = float(st.st_mtime)

    with p.open("rb") as f:
        head = f.read(sample_bytes)
        if size > sample_bytes:
            try:
                f.seek(max(0, size - sample_bytes))
                tail = f.read(sample_bytes)
            except Exception:
                tail = b""
        else:
            tail = b""

    digest = _sha256(head + b"||" + tail)
    return {
        "path": str(p),
        "exists": True,
        "bytes": size,
        "mtime": mtime,
        "sha256_sample": digest,
    }


def fingerprint_inputs(paths: Iterable[Union[str, Path]]) -> List[Dict[str, Any]]:
    return [fingerprint_file(p) for p in paths]


# ---------------------------------------------------------------------
# Manifest schema
# ---------------------------------------------------------------------

def _manifest_columns() -> List[str]:
    # NOTE: keep stable column names; add new ones only with default None.
    return [
        "stage",
        "artifact",
        "path",
        "schema_version",
        "created_at_utc",
        "inputs_json",
        "outputs_json",
        "rows",
        "cols",
        "warnings_count",
        "fails_count",
        "runtime_sec",
        "notes",
        "extra_json",
    ]


def manifest_path(project_root: Union[str, Path], *, rel_path: Union[str, Path] = DEFAULT_MANIFEST_REL_PATH) -> Path:
    return Path(project_root) / Path(rel_path)


def load_manifest(project_root: Union[str, Path]) -> pd.DataFrame:
    mp = manifest_path(project_root)
    cols = _manifest_columns()
    if not mp.exists():
        return pd.DataFrame(columns=cols)
    df = pd.read_parquet(mp)
    # Upgrade schema if needed
    for c in cols:
        if c not in df.columns:
            df[c] = None
    return df[cols].copy()


def append_manifest(project_root: Union[str, Path], row: Dict[str, Any]) -> Path:
    mp = manifest_path(project_root)
    mp.parent.mkdir(parents=True, exist_ok=True)

    df = load_manifest(project_root)
    # Align row keys
    aligned: Dict[str, Any] = {}
    for c in df.columns:
        aligned[c] = row.get(c, None)

    df = pd.concat([df, pd.DataFrame([aligned])], ignore_index=True)
    df.to_parquet(mp, index=False)
    return mp


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def register_artifact(
    *,
    project_root: Union[str, Path],
    stage: str,
    artifact: str,
    path: Union[str, Path],
    schema_version: str,
    inputs: Optional[Sequence[Union[str, Path]]] = None,
    outputs: Optional[Sequence[Union[str, Path]]] = None,
    df: Optional[pd.DataFrame] = None,
    warnings_count: int = 0,
    fails_count: int = 0,
    runtime_sec: Optional[float] = None,
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Register an artifact in exports/artifact_manifest.parquet.

    This is typically called after writing a parquet/csv/html artifact.
    """
    inputs = inputs or []
    outputs = outputs or [path]

    inputs_fp = fingerprint_inputs(inputs)
    outputs_fp = [fingerprint_file(p) for p in outputs]

    row = {
        "stage": str(stage),
        "artifact": str(artifact),
        "path": str(Path(path)),
        "schema_version": str(schema_version),
        "created_at_utc": _utc_now_iso(),
        "inputs_json": json.dumps(inputs_fp, ensure_ascii=False),
        "outputs_json": json.dumps(outputs_fp, ensure_ascii=False),
        "rows": int(len(df)) if df is not None else None,
        "cols": int(len(df.columns)) if df is not None else None,
        "warnings_count": int(warnings_count),
        "fails_count": int(fails_count),
        "runtime_sec": float(runtime_sec) if runtime_sec is not None else None,
        "notes": str(notes or ""),
        "extra_json": json.dumps(extra or {}, ensure_ascii=False),
    }
    return append_manifest(project_root, row)


def append_stage_manifest(
    *,
    project_root: Union[str, Path],
    stage: str,
    inputs: Sequence[Union[str, Path]],
    outputs: Sequence[Union[str, Path]],
    schema_version: str,
    warnings_count: int = 0,
    fails_count: int = 0,
    runtime_sec: Optional[float] = None,
    notes: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Append a single stage-run row to the manifest.

    This is useful when you want one row that summarizes the whole stage run.
    """
    return register_artifact(
        project_root=project_root,
        stage=stage,
        artifact="__stage_run__",
        path=str(Path(outputs[0])) if outputs else "",
        schema_version=schema_version,
        inputs=list(inputs),
        outputs=list(outputs),
        df=None,
        warnings_count=int(warnings_count),
        fails_count=int(fails_count),
        runtime_sec=runtime_sec,
        notes=notes,
        extra=extra,
    )


def read_latest_stage_outputs(project_root: Union[str, Path], *, stage: str) -> Dict[str, str]:
    """Return a dict {artifact: path} for the latest record per artifact in a stage."""
    df = load_manifest(project_root)
    if df.empty:
        return {}
    sdf = df[df["stage"].astype(str) == str(stage)].copy()
    if sdf.empty:
        return {}
    # Sort by created_at_utc (ISO format sorts lexicographically) then take last per artifact
    sdf = sdf.sort_values("created_at_utc")
    latest = sdf.groupby("artifact", as_index=False).tail(1)
    out = {str(r["artifact"]): str(r["path"]) for _, r in latest.iterrows() if str(r.get("path", ""))}
    return out
