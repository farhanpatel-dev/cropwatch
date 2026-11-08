"""Cut an image into overlapping square tiles and paste per-tile results back.

The model only accepts tile x tile inputs (256 x 256), so a large image is cut into tiles
that overlap by `overlap` pixels (32). Predictions near a tile's edge see less context and
are less reliable, so where two tiles overlap each one keeps only the half of the overlap
nearest its own centre. Every image pixel is therefore owned by exactly one tile: stitching
is a plain copy (no averaging), and only the final H x W maps are kept in memory.

Images smaller than one tile are padded (edge pixels repeated) up to the tile size; the
padding is never pasted back because no tile owns it.

Pure numpy, no model: tested on its own in tests/test_tiling.py.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

TILE = 256
OVERLAP = 32


@dataclass(frozen=True)
class Tile:
    """One tile: where it is cut from (padded image) and which pixels it owns (image).

    y, x:   top-left corner of the tile in the padded image.
    own:    (y0, y1, x0, x1) half-open box of image pixels this tile writes.
    """

    y: int
    x: int
    own: tuple[int, int, int, int]


def tile_starts(length: int, tile: int = TILE, overlap: int = OVERLAP) -> list[int]:
    """Start positions along one axis so tiles cover [0, length) with at least `overlap`.

    Steps by tile - overlap; the last tile is moved back to end exactly at `length`, so it
    may overlap its neighbour more than `overlap`. A length up to one tile gives [0].
    """
    if length < 1:
        raise ValueError("length must be at least 1")
    if not 0 <= overlap < tile:
        raise ValueError("overlap must be in [0, tile)")
    if length <= tile:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def _owned_ranges(starts: list[int], length: int, tile: int) -> list[tuple[int, int]]:
    """For each tile on one axis, the [lo, hi) image range it owns.

    The border between neighbours is the middle of their overlap, so each tile keeps the
    part nearest its own centre. The first tile starts at 0, the last ends at `length`.
    """
    borders = [(a + tile + b) // 2 for a, b in zip(starts, starts[1:], strict=False)]
    los = [0, *borders]
    his = [*borders, length]
    return list(zip(los, his, strict=True))


def plan_tiles(height: int, width: int, tile: int = TILE, overlap: int = OVERLAP) -> list[Tile]:
    """All tiles for a height x width image, row by row."""
    ys = tile_starts(height, tile, overlap)
    xs = tile_starts(width, tile, overlap)
    y_own = _owned_ranges(ys, height, tile)
    x_own = _owned_ranges(xs, width, tile)
    return [
        Tile(y=y, x=x, own=(y0, y1, x0, x1))
        for y, (y0, y1) in zip(ys, y_own, strict=True)
        for x, (x0, x1) in zip(xs, x_own, strict=True)
    ]


def pad_to_tile(image: np.ndarray, tile: int = TILE) -> np.ndarray:
    """Pad an H x W (x C) image at the bottom/right to at least tile x tile.

    Edge pixels are repeated (not zeros), so padding looks like the nearby field rather
    than a black border. Images already at least tile x tile are returned unchanged.
    """
    h, w = image.shape[:2]
    pad_h, pad_w = max(0, tile - h), max(0, tile - w)
    if pad_h == 0 and pad_w == 0:
        return image
    pad = [(0, pad_h), (0, pad_w)] + [(0, 0)] * (image.ndim - 2)
    return np.pad(image, pad, mode="edge")


def iter_tiles(image: np.ndarray, plan: list[Tile], tile: int = TILE) -> Iterator[np.ndarray]:
    """Yield the tile x tile (x C) crop for each tile in `plan`, in order."""
    padded = pad_to_tile(image, tile)
    for t in plan:
        yield padded[t.y : t.y + tile, t.x : t.x + tile]


def paste(t: Tile, tile_result: np.ndarray, out: np.ndarray) -> None:
    """Copy the part of one tile's result that the tile owns into the full-size `out`.

    tile_result: tile x tile (x ...) result for tile `t` (for example class ids).
    out:         H x W (x ...) array for the whole image, written in place.
    """
    y0, y1, x0, x1 = t.own
    out[y0:y1, x0:x1] = tile_result[y0 - t.y : y1 - t.y, x0 - t.x : x1 - t.x]


def batched(items: list[Tile], size: int) -> Iterator[list[Tile]]:
    """Split a tile plan into lists of at most `size` tiles (one model call each)."""
    if size < 1:
        raise ValueError("size must be at least 1")
    for i in range(0, len(items), size):
        yield items[i : i + size]
