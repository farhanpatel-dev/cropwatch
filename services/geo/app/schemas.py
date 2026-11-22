"""Request bodies (docs/api.md, Geo). GeoJSON is EPSG:4326 with [longitude, latitude]."""

from typing import Literal

from pydantic import BaseModel, Field


class PolygonGeometry(BaseModel):
    type: Literal["Polygon"]
    coordinates: list[list[list[float]]] = Field(min_length=1)


class AnalysisCreate(BaseModel):
    name: str | None = None
    source_filename: str | None = None
    model_name: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    image_bounds: PolygonGeometry | None = None


class FeatureGeometry(BaseModel):
    type: Literal["Polygon", "MultiPolygon"]
    coordinates: list


class FeatureProperties(BaseModel):
    class_id: int = Field(ge=0, le=32767)
    class_name: str = Field(min_length=1)
    confidence: float | None = Field(default=None, ge=0, le=1)
    area_m2: float | None = Field(default=None, ge=0)


class Feature(BaseModel):
    type: Literal["Feature"]
    geometry: FeatureGeometry
    properties: FeatureProperties


class FeatureCollection(BaseModel):
    type: Literal["FeatureCollection"]
    features: list[Feature]


def parse_bbox(text: str) -> tuple[float, float, float, float]:
    """'minLon,minLat,maxLon,maxLat' -> four floats; ValueError with a clear message."""
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError as e:
        raise ValueError("bbox must be four numbers: minLon,minLat,maxLon,maxLat") from e
    if len(values) != 4:
        raise ValueError("bbox must be four numbers: minLon,minLat,maxLon,maxLat")
    min_lon, min_lat, max_lon, max_lat = values
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("bbox must have min < max")
    if not (-180 <= min_lon and max_lon <= 180 and -90 <= min_lat and max_lat <= 90):
        raise ValueError("bbox must be longitude -180..180 and latitude -90..90")
    return min_lon, min_lat, max_lon, max_lat
