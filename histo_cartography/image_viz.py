from __future__ import annotations

"""Image visualization helpers for "glass box" cluster/relationship verification.

These utilities generate montages that can be:
- displayed inline in Colab notebooks
- saved to exports/<stage>/plots/

We keep the implementation dependency-minimal: PIL + numpy + matplotlib.
"""

from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


def _open_image(path: Union[str, Path], *, size: Tuple[int, int]) -> np.ndarray:
    from PIL import Image  # type: ignore

    p = Path(path)
    with Image.open(p) as im:
        im = im.convert("RGB").resize(size)
        return np.asarray(im, dtype=np.uint8)


def tile_montage(
    image_paths: Sequence[Union[str, Path]],
    *,
    out_path: Union[str, Path],
    title: str = "",
    max_tiles: int = 36,
    tile_size: Tuple[int, int] = (128, 128),
    n_cols: int = 6,
    random_state: int = 1337,
) -> Path:
    """Create and save a montage grid as a PNG.

    Returns the saved path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    paths = [Path(p) for p in image_paths if p is not None]
    paths = [p for p in paths if p.exists()]
    if not paths:
        # Write an empty placeholder image to avoid crashing notebooks.
        from PIL import Image, ImageDraw  # type: ignore

        im = Image.new("RGB", (tile_size[0] * n_cols, tile_size[1]), color=(255, 255, 255))
        draw = ImageDraw.Draw(im)
        draw.text((10, 10), "no images found", fill=(0, 0, 0))
        im.save(out_path)
        return out_path

    rng = np.random.default_rng(int(random_state))
    if len(paths) > int(max_tiles):
        idx = rng.choice(len(paths), size=int(max_tiles), replace=False)
        paths = [paths[int(i)] for i in idx]

    n = len(paths)
    cols = int(n_cols)
    rows = int(np.ceil(n / cols))

    W = tile_size[0] * cols
    H = tile_size[1] * rows

    montage = np.zeros((H, W, 3), dtype=np.uint8) + 255

    for k, p in enumerate(paths):
        r = k // cols
        c = k % cols
        try:
            arr = _open_image(p, size=tile_size)
        except Exception:
            continue
        y0 = r * tile_size[1]
        x0 = c * tile_size[0]
        montage[y0 : y0 + tile_size[1], x0 : x0 + tile_size[0], :] = arr

    # Save via PIL for crisp output.
    from PIL import Image  # type: ignore

    im = Image.fromarray(montage)
    im.save(out_path)
    return out_path


def montage_by_cluster(
    items: pd.DataFrame,
    *,
    cluster_id: int,
    out_path: Union[str, Path],
    image_col: str = "image_path",
    cluster_col: str = "cluster_id",
    n: int = 36,
    random_state: int = 1337,
    title: Optional[str] = None,
) -> Path:
    df = items[items[cluster_col].astype(int) == int(cluster_id)].copy()
    paths = df[image_col].dropna().astype(str).tolist()
    return tile_montage(
        paths,
        out_path=out_path,
        title=title or f"cluster {cluster_id} montage",
        max_tiles=int(n),
        random_state=int(random_state),
    )


def montage_pair_for_link(
    items: pd.DataFrame,
    *,
    cluster_a: int,
    cluster_b: int,
    out_path: Union[str, Path],
    image_col: str = "image_path",
    cluster_col: str = "cluster_id",
    n_each: int = 18,
    random_state: int = 1337,
) -> Path:
    """Side-by-side montage for link verification.

    Creates a 6x6 montage where first half tiles are from cluster_a and second half from cluster_b.
    """
    df_a = items[items[cluster_col].astype(int) == int(cluster_a)].copy()
    df_b = items[items[cluster_col].astype(int) == int(cluster_b)].copy()

    paths_a = df_a[image_col].dropna().astype(str).tolist()
    paths_b = df_b[image_col].dropna().astype(str).tolist()

    rng = np.random.default_rng(int(random_state))
    if len(paths_a) > int(n_each):
        paths_a = [paths_a[i] for i in rng.choice(len(paths_a), size=int(n_each), replace=False)]
    if len(paths_b) > int(n_each):
        paths_b = [paths_b[i] for i in rng.choice(len(paths_b), size=int(n_each), replace=False)]

    paths = list(paths_a) + list(paths_b)
    return tile_montage(paths, out_path=out_path, max_tiles=int(n_each) * 2, n_cols=6, random_state=int(random_state))


def montage_sample(
    items: pd.DataFrame,
    *,
    out_path: Union[str, Path],
    image_col: str = "image_path",
    n: int = 49,
    random_state: int = 1337,
    title: Optional[str] = None,
) -> Path:
    """Random sample montage across the whole dataset (quick sanity check)."""
    paths = items[image_col].dropna().astype(str).tolist()
    return tile_montage(
        paths,
        out_path=out_path,
        title=title or "sample montage",
        max_tiles=int(n),
        n_cols=int(np.ceil(np.sqrt(int(n)))),
        random_state=int(random_state),
    )
