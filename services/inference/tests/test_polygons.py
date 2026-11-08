import numpy as np
import pytest
from affine import Affine
from shapely.geometry import shape

from app.polygons import image_bounds, polygonize, transform_for_bounds

NAMES = {0: "background", 1: "nutrient_deficiency", 2: "drydown", 3: "weed_cluster"}
UTM10 = "EPSG:32610"  # UTM zone 10N (metres), covers -126 to -120 longitude
# 1 m pixels, top-left corner at easting 500000 (central meridian), northing 5440000 (~49.1 N)
UTM_TRANSFORM = Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 5440000.0)
BOUNDS = [-123.2, 49.1, -123.1, 49.2]


def maps(h: int = 100, w: int = 100) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((h, w), np.uint8), np.full((h, w), 0.9, np.float32)


def run(classes: np.ndarray, conf: np.ndarray, transform=UTM_TRANSFORM, crs=UTM10) -> list:
    return polygonize(classes, conf, transform, crs, NAMES)["features"]


def test_square_area_in_square_metres() -> None:
    classes, conf = maps()
    classes[10:40, 20:60] = 1  # 30 x 40 pixels of 1 m = 1200 m2
    (f,) = run(classes, conf)
    assert f["properties"]["class_id"] == 1
    assert f["properties"]["class_name"] == "nutrient_deficiency"
    # geodesic area vs UTM plane: UTM scale factor 0.9996 at the central meridian
    assert f["properties"]["area_m2"] == pytest.approx(1200, rel=0.002)


def test_geometry_is_lon_lat_multipolygon_with_closed_rings() -> None:
    classes, conf = maps()
    classes[10:40, 20:60] = 2
    (f,) = run(classes, conf)
    geom = f["geometry"]
    assert geom["type"] == "MultiPolygon"
    for polygon in geom["coordinates"]:
        for ring in polygon:
            assert ring[0] == ring[-1]
            for lon, lat in ring:
                assert -123.1 < lon < -122.9  # 500000 E in zone 10 is -123 longitude
                assert 49.0 < lat < 49.2


def test_background_is_never_returned() -> None:
    classes, conf = maps()
    assert run(classes, conf) == []


def test_small_patches_dropped() -> None:
    classes, conf = maps()
    classes[0:4, 0:4] = 3  # 16 px: dropped
    classes[50:55, 50:55] = 3  # 25 px: kept
    feats = run(classes, conf)
    assert len(feats) == 1
    assert feats[0]["properties"]["area_m2"] == pytest.approx(25, rel=0.01)


def test_one_feature_per_patch_and_per_class() -> None:
    classes, conf = maps()
    classes[0:10, 0:10] = 1
    classes[50:60, 50:60] = 1  # second patch, same class
    classes[0:10, 80:100] = 3
    feats = run(classes, conf)
    assert [f["properties"]["class_id"] for f in feats] == [1, 1, 3]


def test_hole_is_kept_and_not_counted_in_area() -> None:
    classes, conf = maps()
    classes[10:50, 10:50] = 2  # 1600 px
    classes[20:30, 20:30] = 0  # minus a 100 px hole
    (f,) = run(classes, conf)
    assert len(f["geometry"]["coordinates"][0]) == 2  # outer ring + hole
    assert f["properties"]["area_m2"] == pytest.approx(1500, rel=0.002)


def test_confidence_is_mean_over_patch_pixels() -> None:
    classes, conf = maps()
    classes[0:10, 0:10] = 1
    conf[0:10, 0:5] = 0.5  # half the patch at 0.5, half at 0.9
    conf[50:60, 50:60] = 0.0  # outside the patch: must not count
    (f,) = run(classes, conf)
    assert f["properties"]["confidence"] == pytest.approx(0.7, abs=1e-3)


def test_simplify_reduces_staircase_vertices() -> None:
    classes, conf = maps(200, 200)
    yy, xx = np.mgrid[:200, :200]
    classes[(yy - 100) ** 2 + (xx - 100) ** 2 < 80**2] = 1  # pixel disc: staircase outline
    (f,) = run(classes, conf)
    ring = f["geometry"]["coordinates"][0][0]
    assert len(ring) < 150  # a raw pixel outline of this disc has ~640 vertices
    assert f["properties"]["area_m2"] == pytest.approx(np.pi * 80**2, rel=0.02)


def test_png_bounds_in_wgs84() -> None:
    classes, conf = maps(100, 200)
    classes[:, :] = 1  # whole image
    t = transform_for_bounds(BOUNDS, width=200, height=100)
    (f,) = run(classes, conf, transform=t, crs="EPSG:4326")
    assert shape(f["geometry"]).bounds == pytest.approx((-123.2, 49.1, -123.1, 49.2))
    # 0.1 x 0.1 degrees at 49.15 N: about 7.29 km x 11.12 km
    assert f["properties"]["area_m2"] == pytest.approx(8.1e7, rel=0.01)
    assert image_bounds(t, 200, 100, "EPSG:4326") == pytest.approx(BOUNDS)


def test_image_bounds_reprojects_utm() -> None:
    lon0, lat0, lon1, lat1 = image_bounds(UTM_TRANSFORM, 100, 100, UTM10)
    assert lon0 == pytest.approx(-123.0, abs=1e-6)  # left edge on the central meridian
    assert 0 < lon1 - lon0 < 0.002 and 0 < lat1 - lat0 < 0.001  # 100 m in degrees


@pytest.mark.parametrize("bad", [[1, 0, 0, 1], [-123, 49, -123, 49.1], [-200, 0, 0, 1]])
def test_bad_bounds_rejected(bad: list[float]) -> None:
    with pytest.raises(ValueError):
        transform_for_bounds(bad, 10, 10)
