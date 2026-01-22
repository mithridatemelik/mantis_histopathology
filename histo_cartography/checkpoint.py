from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

from .runtime import env_fingerprint, _utc_now, _log


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def dataframe_fingerprint(df, key_cols: Sequence[str] = (), sample_n: int = 1000) -> Dict[str, Any]:
    """Cheap-ish fingerprint for a table: row count + column names + sample hash."""
    info: Dict[str, Any] = {
        "rows": int(getattr(df, "shape", (0, 0))[0]),
        "cols": list(getattr(df, "columns", [])),
    }
    try:
        import pandas as pd  # type: ignore

        if isinstance(df, pd.DataFrame) and key_cols:
            sub = df[list(key_cols)].head(sample_n).astype(str)
            info["key_cols"] = list(key_cols)
            info["key_cols_sha256"] = _sha256_bytes("\n".join(["|".join(r) for r in sub.values.tolist()]).encode("utf-8"))
    except Exception:
        pass
    return info


def write_manifest(
    artifact_path: Path,
    *,
    schema_version: str,
    df=None,
    key_cols: Sequence[str] = (),
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    artifact_path = Path(artifact_path)
    manifest_path = artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")
    payload: Dict[str, Any] = {
        "schema_version": schema_version,
        "artifact": str(artifact_path),
        "created_at_utc": _utc_now(),
        "env": env_fingerprint(),
    }
    if df is not None:
        payload["table"] = dataframe_fingerprint(df, key_cols=key_cols)
    if extra:
        payload.update(extra)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    _log().info("Wrote manifest", extra={"extra": {"manifest": str(manifest_path)}})
    return manifest_path


def has_valid_manifest(artifact_path: Path) -> bool:
    mp = Path(str(artifact_path) + ".manifest.json") if str(artifact_path).endswith(".manifest.json") else Path(artifact_path.with_suffix(artifact_path.suffix + ".manifest.json"))
    return mp.exists()


def load_manifest(artifact_path: Path) -> Dict[str, Any]:
    mp = artifact_path.with_suffix(artifact_path.suffix + ".manifest.json")
    return json.loads(mp.read_text())
