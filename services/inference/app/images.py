"""Read an uploaded image into RGB pixels plus where it sits on the map.

Rules (docs/api.md, Inference):
- GeoTIFF with a CRS: placed by its own transform and CRS; `bounds` is ignored.
- PNG / JPEG (or a GeoTIFF without CRS): `bounds` "minLon,minLat,maxLon,maxLat" is required.
- More than 3 bands: bands 1-3 are used as RGB (for example PNG alpha, or NIR in a GeoTIFF).
- Fewer than 3 bands: rejected (400). Larger than 6000 x 6000: rejected (413).
- Proposal (not yet in docs/api.md): only 8-bit images (0-255), as in training; others 400.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS
from rasterio.errors import NotGeoreferencedWarning, RasterioIOError
from rasterio.io import MemoryFile

from app.config import MAX_PIXELS_SIDE
from app.polygons import WGS84, transform_for_bounds

DRIVERS = {"GTiff", "PNG", "JPEG"}


class ImageError(ValueError):
    """Upload that cannot be used. status_code is the HTTP status to return."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class PlacedImage:
    rgb: np.ndarray  # H x W x 3 uint8
    transform: Affine  # pixel -> crs
    crs: CRS


def parse_bounds(text: str) -> list[float]:
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError as e:
        raise ImageError("bounds must be four numbers: minLon,minLat,maxLon,maxLat") from e
    if len(values) != 4:
        raise ImageError("bounds must be four numbers: minLon,minLat,maxLon,maxLat")
    return values


def read_image(data: bytes, bounds: str | None, max_side: int = MAX_PIXELS_SIDE) -> PlacedImage:
    if not data:
        raise ImageError("empty file")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)  # png/jpg have no CRS
        try:
            with MemoryFile(data) as mem, mem.open() as src:
                return _read(src, bounds, max_side)
        except RasterioIOError as e:
            raise ImageError("not a readable image: use GeoTIFF, PNG or JPEG") from e


def _read(src: rasterio.DatasetReader, bounds: str | None, max_side: int) -> PlacedImage:
    if src.driver not in DRIVERS:
        raise ImageError(f"unsupported image type {src.driver}: use GeoTIFF, PNG or JPEG")
    if src.width > max_side or src.height > max_side:
        raise ImageError(
            f"image is {src.width} x {src.height} pixels; the maximum is {max_side} x {max_side}",
            status_code=413,
        )
    if src.count < 3:
        raise ImageError(f"image has {src.count} band(s); 3 (RGB) are needed")
    if src.dtypes[0] != "uint8":
        raise ImageError(f"image is {src.dtypes[0]}; only 8-bit (uint8) images are supported")

    if src.driver == "GTiff" and src.crs is not None:
        transform, crs = src.transform, CRS.from_user_input(src.crs.to_wkt())
    elif bounds is None or not bounds.strip():
        raise ImageError("bounds are required for images without a CRS (PNG, JPEG)")
    else:
        try:
            transform = transform_for_bounds(parse_bounds(bounds), src.width, src.height)
        except ValueError as e:
            raise ImageError(str(e)) from e
        crs = WGS84

    rgb = src.read([1, 2, 3]).transpose(1, 2, 0)
    return PlacedImage(rgb=np.ascontiguousarray(rgb), transform=transform, crs=crs)
