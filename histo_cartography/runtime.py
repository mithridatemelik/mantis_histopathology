from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import os
import platform
import random
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


STATE: Dict[str, Any] = {
    "schema_version": "0.1.0",
    "run_id": None,
    "started_at": None,
    "project_root": None,
    "safe_mode": True,
    "debug_level": 1,
    "cells": {},
    "artifacts": {},
    "last_success": {
        "cell_id": None,
        "checkpoint": None,
        "time": None,
    },
    "errors": [],
}

RUNTIME: Dict[str, Any] = {
    "PROJECT_ROOT": None,
    "LOG_DIR": None,
    "CHECKPOINT_DIR": None,
    "LOGGER": None,
    "STATE_PATH": None,
}


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def env_fingerprint() -> Dict[str, Any]:
    fp: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "executable": sys.executable,
        "time_utc": _utc_now(),
    }
    # Torch / CUDA
    if torch is not None:
        fp.update(
            {
                "torch": getattr(torch, "__version__", None),
                "cuda_available": bool(torch.cuda.is_available()),
                "cuda_version": getattr(torch.version, "cuda", None),
                "cudnn": getattr(torch.backends.cudnn, "version", lambda: None)() if hasattr(torch, "backends") else None,
            }
        )
        if torch.cuda.is_available():
            try:
                fp["gpu_name"] = torch.cuda.get_device_name(0)
                fp["gpu_capability"] = torch.cuda.get_device_capability(0)
            except Exception:
                pass
    # NumPy / Pandas
    if np is not None:
        fp["numpy"] = getattr(np, "__version__", None)
    try:
        import pandas as pd  # type: ignore

        fp["pandas"] = getattr(pd, "__version__", None)
    except Exception:
        fp["pandas"] = None

    try:
        import sklearn  # type: ignore

        fp["sklearn"] = getattr(sklearn, "__version__", None)
    except Exception:
        fp["sklearn"] = None

    return fp


def _get_memory_stats() -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    # psutil is typically available on Colab; but keep optional.
    try:
        import psutil  # type: ignore

        vm = psutil.virtual_memory()
        stats.update(
            {
                "ram_total_gb": round(vm.total / (1024**3), 3),
                "ram_available_gb": round(vm.available / (1024**3), 3),
                "ram_used_gb": round(vm.used / (1024**3), 3),
                "ram_percent": vm.percent,
            }
        )
    except Exception:
        pass
    return stats


def _get_gpu_stats() -> Dict[str, Any]:
    stats: Dict[str, Any] = {}
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], stderr=subprocess.STDOUT)
        line = out.decode("utf-8").strip().splitlines()[0]
        name, mem_total, mem_used, util = [x.strip() for x in line.split(",")]
        stats.update(
            {
                "gpu_name": name,
                "gpu_mem_total_mb": int(mem_total),
                "gpu_mem_used_mb": int(mem_used),
                "gpu_util_percent": int(util),
            }
        )
    except Exception:
        # If no GPU / no nvidia-smi, ignore.
        pass
    return stats


class JsonlFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        payload = {
            "time_utc": _utc_now(),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # Merge structured extras if present
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def _make_logger(log_dir: Path, debug_level: int) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("histo_cartography")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers in re-runs
    if getattr(logger, "_configured", False):
        return logger

    # Console handler (human readable)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel([logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG][min(max(debug_level, 0), 3)])
    ch.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    logger.addHandler(ch)

    # File handler (JSONL)
    fh_path = log_dir / "run.jsonl"
    fh = logging.FileHandler(fh_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(JsonlFormatter())
    logger.addHandler(fh)

    logger._configured = True  # type: ignore[attr-defined]
    logger.info(f"Logging to: {fh_path}")
    return logger


def init_runtime(
    project_root: Path,
    *,
    safe_mode: bool = True,
    debug_level: int = 1,
    log_dir_rel: str = "logs",
    checkpoint_dir_rel: str = "checkpoints",
) -> None:
    project_root = Path(project_root)
    RUNTIME["PROJECT_ROOT"] = project_root
    RUNTIME["LOG_DIR"] = project_root / log_dir_rel
    RUNTIME["CHECKPOINT_DIR"] = project_root / checkpoint_dir_rel
    RUNTIME["CHECKPOINT_DIR"].mkdir(parents=True, exist_ok=True)

    RUNTIME["LOGGER"] = _make_logger(RUNTIME["LOG_DIR"], debug_level=debug_level)

    RUNTIME["STATE_PATH"] = RUNTIME["CHECKPOINT_DIR"] / "_STATE.json"

    # Initialize state
    if STATE.get("run_id") is None:
        STATE["run_id"] = f"run_{datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')}_{random.randint(1000,9999)}"
        STATE["started_at"] = _utc_now()
    STATE["project_root"] = str(project_root)
    STATE["safe_mode"] = bool(safe_mode)
    STATE["debug_level"] = int(debug_level)

    # Load existing state if present (resume mode)
    if RUNTIME["STATE_PATH"].exists():
        try:
            prev = json.loads(RUNTIME["STATE_PATH"].read_text())
            # Merge carefully: keep current run_id but import previous cell statuses & artifacts
            for k in ("cells", "artifacts", "last_success", "errors"):
                if k in prev:
                    STATE[k] = prev[k]
            _log().info("Loaded existing STATE for resume mode", extra={"extra": {"state_path": str(RUNTIME["STATE_PATH"])}})
        except Exception:
            _log().warning("Failed to load existing STATE; starting fresh", extra={"extra": {"state_path": str(RUNTIME["STATE_PATH"])}})

    save_state()


def _log() -> logging.Logger:
    lg = RUNTIME.get("LOGGER")
    if lg is None:
        # Fallback basic logger
        logging.basicConfig(level=logging.INFO)
        lg = logging.getLogger("histo_cartography")
        RUNTIME["LOGGER"] = lg
    return lg


def save_state() -> None:
    sp: Path = RUNTIME.get("STATE_PATH")  # type: ignore[assignment]
    if sp is None:
        return
    try:
        sp.write_text(json.dumps(STATE, indent=2, ensure_ascii=False))
    except Exception:
        # Don't crash on state save
        pass


def set_seed(seed: int) -> None:
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    _log().info(f"Seeds set to {seed}")


def disk_free_gb(path: Path) -> float:
    try:
        st = os.statvfs(str(path))
        return (st.f_bavail * st.f_frsize) / (1024**3)
    except Exception:
        return float("nan")


def health_check(
    cell_id: str,
    *,
    namespace: Optional[Dict[str, Any]] = None,
    require_vars: Sequence[str] = (),
    require_files: Sequence[Path] = (),
    require_dirs: Sequence[Path] = (),
    min_free_gb: float = 1.0,
) -> None:
    """Fast pre-cell checks: disk, required variables and files."""
    namespace = namespace or {}
    proj = RUNTIME.get("PROJECT_ROOT")
    if proj is None:
        raise RuntimeError("Runtime not initialized. Run init_runtime(project_root=...) first.")

    free = disk_free_gb(Path(proj))
    if not (free >= min_free_gb):
        raise RuntimeError(f"Low disk space under {proj}. free_gb={free:.3f} < {min_free_gb:.3f}")

    missing_vars = [v for v in require_vars if v not in namespace]
    if missing_vars:
        raise RuntimeError(f"Missing required variables for {cell_id}: {missing_vars}")

    missing_files = [str(p) for p in require_files if not Path(p).exists()]
    if missing_files:
        raise RuntimeError(f"Missing required files for {cell_id}: {missing_files}")

    missing_dirs = [str(p) for p in require_dirs if not Path(p).exists()]
    if missing_dirs:
        raise RuntimeError(f"Missing required dirs for {cell_id}: {missing_dirs}")

    # Lightweight memory stats (optional)
    mem = _get_memory_stats()
    extra = {"cell_id": cell_id, **mem}
    _log().debug("health_check ok", extra={"extra": extra})


@contextlib.contextmanager
def cell_context(
    cell_id: str,
    *,
    purpose: str,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    artifacts: Optional[Dict[str, Any]] = None,
    depends_on: Sequence[str] = (),
    stage: str = "",
    checkpoint_paths: Sequence[str] = (),
) -> Any:
    """Context manager to instrument a cell with structured logging + crash capture."""
    t0 = time.time()
    inputs = inputs or {}
    outputs = outputs or {}
    artifacts = artifacts or {}

    STATE.setdefault("cells", {})
    STATE["cells"].setdefault(cell_id, {})
    STATE["cells"][cell_id].update(
        {
            "cell_id": cell_id,
            "purpose": purpose,
            "stage": stage,
            "depends_on": list(depends_on),
            "checkpoint_paths": list(checkpoint_paths),
            "status": "running",
            "started_at": _utc_now(),
        }
    )
    save_state()

    _log().info(f"▶️  {cell_id}: {purpose}", extra={"extra": {"cell_id": cell_id, "stage": stage}})
    try:
        yield
        dt = time.time() - t0
        STATE["cells"][cell_id].update(
            {
                "status": "ok",
                "finished_at": _utc_now(),
                "duration_s": round(dt, 3),
                "inputs": list(inputs.keys()),
                "outputs": list(outputs.keys()),
                "artifacts": artifacts,
            }
        )
        # Update last success
        STATE["last_success"] = {
            "cell_id": cell_id,
            "checkpoint": checkpoint_paths[-1] if checkpoint_paths else None,
            "time": _utc_now(),
        }
        save_state()
        _log().info(f"✅ {cell_id} finished in {dt:.2f}s", extra={"extra": {"cell_id": cell_id, "duration_s": round(dt, 3)}})

    except Exception as e:
        dt = time.time() - t0
        err = {
            "time_utc": _utc_now(),
            "cell_id": cell_id,
            "stage": stage,
            "duration_s": round(dt, 3),
            "error_type": type(e).__name__,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "env": env_fingerprint(),
            "memory": _get_memory_stats(),
            "gpu": _get_gpu_stats(),
            "last_success": STATE.get("last_success"),
        }
        STATE.setdefault("errors", []).append(err)
        STATE["cells"][cell_id].update(
            {
                "status": "error",
                "finished_at": _utc_now(),
                "duration_s": round(dt, 3),
                "error_type": type(e).__name__,
                "error": str(e),
            }
        )
        save_state()
        _log().error(f"❌ {cell_id} failed: {e}", extra={"extra": err})
        raise


def binary_compatibility_guard() -> Dict[str, Any]:
    """Heuristic checks for common Colab binary-compat issues.

    This cannot predict every crash, but it can warn when risky combos are detected.
    """
    fp = env_fingerprint()
    warnings: List[str] = []
    # Example heuristic: pandas + numpy ABI mismatch often appears after upgrades.
    # If numpy is very new compared to pandas (or vice versa), warn.
    try:
        import numpy as _np  # type: ignore
        import pandas as _pd  # type: ignore

        n, p = _np.__version__, _pd.__version__
        # crude: if major versions diverge by >= 1, warn
        if n.split(".")[0] != p.split(".")[0]:
            warnings.append(f"Potential ABI risk: numpy={n} vs pandas={p}. Avoid upgrading either in Colab.")
    except Exception:
        pass

    # GPU / torch sanity
    if torch is not None and torch.cuda.is_available():
        try:
            _ = torch.tensor([1.0]).cuda()
        except Exception as e:
            warnings.append(f"Torch CUDA smoke test failed: {e}")

    report = {"fingerprint": fp, "warnings": warnings}
    _log().info("Binary compatibility guard report", extra={"extra": report})
    return report
