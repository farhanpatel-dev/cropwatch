import numpy as np
import pytest

from app.tiling import (
    OVERLAP,
    TILE,
    batched,
    iter_tiles,
    pad_to_tile,
    paste,
    plan_tiles,
    tile_starts,
)

# Sizes that hit the edge cases: smaller than a tile, exactly one tile, one pixel more,
# a size where the last tile has to move back, a non-square image and a 512 image
# (the Agriculture-Vision size).
SIZES = [(1, 1), (100, 300), (256, 256), (257, 256), (500, 700), (512, 512), (1000, 333)]


def test_starts_basic() -> None:
    assert tile_starts(256) == [0]
    assert tile_starts(100) == [0]
    # stride 224; last tile moved back to end exactly at 512
    assert tile_starts(512) == [0, 224, 256]
    assert tile_starts(480) == [0, 224]


@pytest.mark.parametrize("length", [257, 300, 480, 481, 512, 1000, 8000])
def test_starts_cover_with_overlap(length: int) -> None:
    starts = tile_starts(length)
    assert starts[0] == 0
    assert starts[-1] + TILE == length
    for a, b in zip(starts, starts[1:], strict=False):
        assert b > a
        assert a + TILE - b >= OVERLAP  # neighbours overlap by at least 32 px


def test_starts_reject_bad_input() -> None:
    with pytest.raises(ValueError):
        tile_starts(0)
    with pytest.raises(ValueError):
        tile_starts(500, tile=256, overlap=256)


@pytest.mark.parametrize(("h", "w"), SIZES)
def test_every_pixel_owned_exactly_once(h: int, w: int) -> None:
    count = np.zeros((h, w), dtype=np.int32)
    for t in plan_tiles(h, w):
        y0, y1, x0, x1 = t.own
        assert t.y <= y0 < y1 <= t.y + TILE  # owned box lies inside the tile
        assert t.x <= x0 < x1 <= t.x + TILE
        count[y0:y1, x0:x1] += 1
    assert (count == 1).all()


@pytest.mark.parametrize(("h", "w"), SIZES)
def test_cut_and_stitch_gives_back_the_image(h: int, w: int) -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    plan = plan_tiles(h, w)
    out = np.zeros_like(image)
    for t, crop in zip(plan, iter_tiles(image, plan), strict=True):
        assert crop.shape == (TILE, TILE, 3)
        paste(t, crop, out)
    assert np.array_equal(out, image)


def test_owned_part_stays_away_from_inner_edges() -> None:
    # Where tiles meet, each keeps at least half the overlap (16 px) away from its own edge.
    for t in plan_tiles(1000, 1000):
        y0, y1, x0, x1 = t.own
        if y0 > 0:
            assert y0 - t.y >= OVERLAP // 2
        if y1 < 1000:
            assert t.y + TILE - y1 >= OVERLAP // 2
        if x0 > 0:
            assert x0 - t.x >= OVERLAP // 2
        if x1 < 1000:
            assert t.x + TILE - x1 >= OVERLAP // 2


def test_pad_repeats_edge_pixels() -> None:
    image = np.arange(6, dtype=np.uint8).reshape(2, 3)
    padded = pad_to_tile(image)
    assert padded.shape == (TILE, TILE)
    assert np.array_equal(padded[:2, :3], image)
    assert (padded[100, :3] == image[-1]).all()
    assert pad_to_tile(np.zeros((300, 300), np.uint8)).shape == (300, 300)


def test_tile_count_for_512() -> None:
    assert len(plan_tiles(512, 512)) == 9  # 3 x 3


def test_batched() -> None:
    plan = plan_tiles(512, 512)
    sizes = [len(b) for b in batched(plan, 4)]
    assert sizes == [4, 4, 1]
    with pytest.raises(ValueError):
        next(batched(plan, 0))
