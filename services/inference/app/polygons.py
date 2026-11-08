"""Turn the stitched class map into GeoJSON polygons in EPSG:4326 with area in m2.

Steps (docs/architecture.md, inference step 4; rules from docs/api.md, Inference):
1. Polygonize each stress class (background 0 is skipped) in pixel units: one polygon per
   connected patch of pixels, holes kept.
2. Drop patches smaller than 20 pixels; simplify outlines with a tolerance of about 1 pixel.
3. Confidence = mean softmax probability of the class over the patch's pixels.
4. Move to map coordinates with the image's affine transform (pixel -> source CRS), then
   reproject to EPSG:4326 with [longitude, latitude] order.
5. Area in m2 on the WGS84 ellipsoid (geodesic), never from degrees.

Steps 1-3 work in pixels so the size and simplify rules mean the same at every resolution.
"""

from collections.abc import Mapping, Sequence

import numpy as np
import shapely
from affine import Affine
from pyproj import CRS, Geod, Transformer
from rasterio.features import rasterize, shapes
from rasterio.transform import from_bounds
from rasterio.warp import transform_bounds
from shapely.geometry import MultiPolygon, Polygon, mapping, shape

WGS84 = CRS.from_epsg(4326)
GEOD = Geod(ellps="WGS84")
MIN_PIXELS = 20
SIMPLIFY_PX = 1.0


def transform_for_bounds(bounds: Sequence[float], width: int, height: int) -> Affine:
    """Affine transform for a plain image (png/jpg) placed at minLon,minLat,maxLon,maxLat."""
    min_lon, min_lat, max_lon, max_lat = bounds
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("bounds must be minLon,minLat,maxLon,maxLat with min < max")
    if not (-180 <= min_lon and max_lon <= 180 and -90 <= min_lat and max_lat <= 90):
        raise ValueError("bounds must be longitude -180..180 and latitude -90..90")
    return from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)


def image_bounds(transform: Affine, width: int, height: int, crs: CRS) -> list[float]:
    """[minLon, minLat, maxLon, maxLat] of the whole image in EPSG:4326."""
    left, top = transform @ (0, 0)
    right, bottom = transform @ (width, height)
    box = (min(left, right), min(top, bottom), max(left, right), max(top, bottom))
    if CRS.from_user_input(crs) == WGS84:
        return [float(v) for v in box]
    # densify: edges of a projected image are curved in lon/lat, so sample along them
    return [float(v) for v in transform_bounds(crs, WGS84, *box, densify_pts=21)]


def _to_multipolygon(geom: shapely.Geometry) -> MultiPolygon | None:
    """Keep only the polygon parts (make_valid can return lines); None if nothing is left."""
    if geom.is_empty:
        return None
    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    parts = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon) and not g.is_empty]
    return MultiPolygon(parts) if parts else None


def _mean_confidence(pixel_poly: Polygon, confidence: np.ndarray) -> float:
    """Mean of `confidence` over the pixels inside an (unsimplified) pixel-unit polygon."""
    x0, y0, x1, y1 = (int(round(v)) for v in pixel_poly.bounds)
    window = confidence[y0:y1, x0:x1]
    inside = rasterize(
        [pixel_poly], out_shape=window.shape, transform=Affine.translation(x0, y0), fill=0
    ).astype(bool)
    return float(window[inside].astype(np.float32).mean())


def polygonize(
    classes: np.ndarray,
    confidence: np.ndarray,
    transform: Affine,
    crs: CRS | str,
    class_names: Mapping[int, str],
    min_pixels: int = MIN_PIXELS,
    simplify_px: float = SIMPLIFY_PX,
) -> dict:
    """GeoJSON FeatureCollection of stress patches, EPSG:4326, one feature per patch.

    classes:     H x W uint8 class ids (0 = background, never returned).
    confidence:  H x W probability of the predicted class.
    transform:   affine pixel -> source CRS (from the GeoTIFF, or transform_for_bounds).
    crs:         source CRS of `transform`.
    Feature properties: class_id, class_name, confidence (0-1), area_m2. The geo service
    adds id and analysis_id when it stores them.
    """
    if classes.shape != confidence.shape or classes.ndim != 2:
        raise ValueError("classes and confidence must be H x W arrays of the same shape")
    to_wgs84 = Transformer.from_crs(CRS.from_user_input(crs), WGS84, always_xy=True)
    t = transform
    # shapely affine order: [a, b, d, e, xoff, yoff]
    pixel_to_map = [t.a, t.b, t.d, t.e, t.c, t.f]

    features = []
    for class_id in sorted(int(c) for c in np.unique(classes) if c != 0):
        mask = classes == class_id
        for geojson, _ in shapes(classes, mask=mask, connectivity=4):
            pixel_poly = shape(geojson)
            if pixel_poly.area < min_pixels:  # pixel-boundary polygon: area = pixel count
                continue
            conf = _mean_confidence(pixel_poly, confidence)
            simple = pixel_poly.simplify(simplify_px, preserve_topology=True)
            if not simple.is_valid:
                simple = shapely.make_valid(simple)
            geom = shapely.affinity.affine_transform(simple, pixel_to_map)
            geom = shapely.transform(geom, lambda xy: np.column_stack(to_wgs84.transform(*xy.T)))
            multi = _to_multipolygon(geom)
            if multi is None:
                continue
            area_m2 = abs(GEOD.geometry_area_perimeter(multi)[0])
            features.append(
                {
                    "type": "Feature",
                    "geometry": mapping(multi),
                    "properties": {
                        "class_id": class_id,
                        "class_name": class_names.get(class_id, str(class_id)),
                        "confidence": round(conf, 4),
                        "area_m2": round(area_m2, 1),
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}
