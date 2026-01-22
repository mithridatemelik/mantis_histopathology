from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, List


def is_colab() -> bool:
    return "google.colab" in sys.modules


def mount_drive(mountpoint: str = "/content/drive", *, force_remount: bool = False, timeout_ms: int = 300000, max_tries: int = 3) -> bool:
    """Mount Google Drive if running in Colab, with retries.

    Returns True if mounted (or already mounted), else False.

    Notes:
      - Colab Drive mount can fail transiently (DriveFS timeout). We retry.
      - If it fails consistently, it is often a browser privacy/cookie issue.
    """
    if not is_colab():
        return False
    try:
        from google.colab import drive  # type: ignore
    except Exception:
        return False

    mp = Path(mountpoint)
    if (mp / "MyDrive").exists() and not force_remount:
        return True

    last: Optional[BaseException] = None
    for t in range(max_tries):
        try:
            kwargs = {"force_remount": force_remount or (t > 0)}
            # Some Colab versions accept timeout_ms; ignore if not.
            kwargs["timeout_ms"] = timeout_ms
            try:
                drive.mount(mountpoint, **kwargs)
            except TypeError:
                kwargs.pop("timeout_ms", None)
                drive.mount(mountpoint, **kwargs)
            if (mp / "MyDrive").exists():
                return True
        except Exception as e:
            last = e

    # Failure: keep it quiet (callers decide how to message), but return False.
    return False

def _find_candidate_roots(
    search_base: Path,
    required_files: Sequence[str],
    max_candidates: int = 50,
) -> List[Path]:
    # We search for the *required file* and then take its parent directory as a candidate.
    candidates = []
    for rf in required_files:
        for p in search_base.glob(f"**/{rf}"):
            candidates.append(p.parent)
            if len(candidates) >= max_candidates:
                return candidates
    # Deduplicate while preserving order
    seen = set()
    deduped: List[Path] = []
    for c in candidates:
        s = str(c.resolve())
        if s not in seen:
            seen.add(s)
            deduped.append(c)
    return deduped


def resolve_project_root(
    *,
    env_var: str = "HISTO_PROJECT_ROOT",
    search_base: str = "/content/drive/MyDrive/mit",
    required_files: Sequence[str] = ("pipeline_config.yaml", "label_taxonomy.yaml"),
    prefer_newest: bool = True,
) -> Path:
    """Resolve the project root reliably in Colab + Drive.

    Strategy:
      1) If env var HISTO_PROJECT_ROOT is set and exists -> use it.
      2) If /content/drive/MyDrive is mounted and search_base exists -> find a folder containing required_files.
         If multiple candidates -> pick the newest (mtime) unless prefer_newest=False.
      3) Fallback to CWD.

    NOTE: This avoids reliance on fragile relative paths in Colab.
    """
    ev = os.environ.get(env_var)
    if ev:
        p = Path(ev).expanduser()
        if p.exists():
            return p

    sb = Path(search_base)
    if sb.exists():
        candidates = _find_candidate_roots(sb, required_files=required_files)
        if candidates:
            if len(candidates) == 1:
                return candidates[0]
            if prefer_newest:
                candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
                return candidates[0]
            return candidates[0]

    # Some users place the project directly under MyDrive (not under /mit)
    alt = Path("/content/drive/MyDrive")
    if alt.exists():
        candidates = _find_candidate_roots(alt, required_files=required_files)
        if candidates:
            candidates = sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)
            return candidates[0]

    return Path.cwd()


def ensure_dirs(project_root: Path, rel_dirs: Iterable[str]) -> None:
    for d in rel_dirs:
        (project_root / d).mkdir(parents=True, exist_ok=True)


def as_posix(p: Path) -> str:
    return str(p.as_posix())
