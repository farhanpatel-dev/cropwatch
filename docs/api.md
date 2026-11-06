# API contract and database schema (single source of truth)

Copy of project file 03. Changes agreed on 2026-11-04: model input is RGB (3 channels); bbox query also filters by `analysis_id`; stats rule for bbox defined.

Conventions: JSON over HTTP; geometry is GeoJSON in EPSG:4326 with [longitude, latitude]; ids are UUID strings for analyses and integers for polygons; areas in square metres (`area_m2`) in features and hectares (`area_ha`) in stats; errors use FastAPI's default body `{"detail": "message"}`. Changing this file requires updating the services and tests in the same pull request.

## Classes (frozen; names match the Agriculture-Vision 2021 label folders)
| id | name | colour |
| --- | --- | --- |
| 0 | background | transparent |
| 1 | nutrient_deficiency | #e6194b |
| 2 | drydown | #f58231 |
| 3 | weed_cluster | #911eb4 |
Defined once in `ml/configs/classes.yaml` and copied to `services/inference/models/classes.yaml`.

## Gateway (public) : base http://localhost:8000
| Method and path | Purpose |
| --- | --- |
| GET /health | `{"status":"ok"}` |
| GET /api/v1/classes | List of classes above |
| POST /api/v1/analyses | Upload and analyse an image |
| GET /api/v1/analyses | List analyses, newest first |
| GET /api/v1/analyses/{id} | One analysis with stats |
| GET /api/v1/analyses/{id}/polygons | FeatureCollection for that analysis |
| GET /api/v1/polygons?bbox=minLon,minLat,maxLon,maxLat&class_id=&analysis_id=&limit=1000 | FeatureCollection in a map view |
| GET /api/v1/stats?analysis_id=&bbox= | Area per class |

### POST /api/v1/analyses
Request: multipart/form-data
- `file` (required): .tif/.tiff (GeoTIFF with CRS), .png, .jpg/.jpeg; max 50 MB
- `bounds` (required for png/jpg, ignored for GeoTIFF): "minLon,minLat,maxLon,maxLat"
- `name` (optional)

Response 201:
```json
{
  "id": "0b7c1d1e-6e1f-4c58-9a3e-2f3f0a6a9b11",
  "name": "north-field",
  "status": "completed",
  "created_at": "2026-11-04T17:20:00Z",
  "model": {"name": "unet", "version": "3"},
  "image_bounds": {"type": "Polygon", "coordinates": [[[-123.2, 49.1], [-123.1, 49.1], [-123.1, 49.2], [-123.2, 49.2], [-123.2, 49.1]]]},
  "stats": [
    {"class_id": 1, "class_name": "nutrient_deficiency", "polygon_count": 12, "area_ha": 3.42}
  ]
}
```
Errors: 400 bad file type or bounds; 413 file too large; 422 validation; 502 upstream service error; 504 inference timeout.

### Polygon feature (used everywhere)
```json
{
  "type": "Feature",
  "geometry": {"type": "MultiPolygon", "coordinates": [[[[-123.15, 49.15], [-123.14, 49.15], [-123.14, 49.16], [-123.15, 49.15]]]]},
  "properties": {"id": 101, "analysis_id": "0b7c...", "class_id": 1, "class_name": "nutrient_deficiency", "confidence": 0.83, "area_m2": 5120.4}
}
```
Responses that return polygons use `{"type": "FeatureCollection", "features": [ ... ]}`.

## Inference : http://inference:8001
| Method and path | Purpose |
| --- | --- |
| GET /health | liveness |
| GET /model-info | `{"name":"unet","version":"3","mlflow_run_id":"...","input_channels":3,"tile_size":256,"preprocessing":{"input_scale":"divide by 255","mean":[r,g,b],"std":[r,g,b]},"classes":[{"id":0,"name":"background"}, ...]}` |
| POST /predict | multipart `file`, optional `bounds`. Response 200: `{"model":{"name","version"},"image":{"width","height","crs":"EPSG:4326","bounds":[minLon,minLat,maxLon,maxLat]},"polygons":FeatureCollection}` |

Rules: the model takes 3 channels (RGB). For GeoTIFFs with more bands, bands 1 to 3 are used as RGB and the rest are ignored; single-band images are rejected with 400. Before the model, each RGB pixel is divided by 255, then normalised per channel as (value - mean) / std using `preprocessing` from `model_info.json` (the values measured on the training data); never hard-code them. Pixel size must not exceed 8000x8000 in the MVP (reject larger with 413); drop polygons smaller than 20 pixels; simplify tolerance of about 1 pixel; `confidence` is the mean softmax probability of the class inside the polygon.

## Geo : http://geo:8002 (internal only)
| Method and path | Purpose |
| --- | --- |
| GET /health | liveness plus a `SELECT 1` database check |
| POST /analyses | body `{"name","source_filename","model_name","model_version","image_bounds":Polygon}` -> 201 analysis row |
| POST /analyses/{id}/polygons | body FeatureCollection -> 201 `{"inserted": n}` (bulk insert, one transaction) |
| GET /analyses, GET /analyses/{id} | read |
| GET /analyses/{id}/polygons | FeatureCollection |
| GET /polygons?bbox=&class_id=&analysis_id=&limit= | FeatureCollection using the GiST index (`geom && ST_MakeEnvelope(...)`) |
| GET /stats?analysis_id=&bbox= | `[{"class_id","class_name","polygon_count","area_ha"}]`. With `bbox`, a polygon is counted with its whole area if it intersects the box (no clipping). |

## Database schema (PostgreSQL 16 + PostGIS)
```sql
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE analyses (
  id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name           TEXT,
  source_filename TEXT,
  model_name     TEXT NOT NULL,
  model_version  TEXT NOT NULL,
  status         TEXT NOT NULL DEFAULT 'completed',
  image_bounds   geometry(Polygon, 4326),
  created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE predictions (
  id           BIGSERIAL PRIMARY KEY,
  analysis_id  UUID NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
  class_id     SMALLINT NOT NULL,
  class_name   TEXT NOT NULL,
  confidence   REAL,
  area_m2      DOUBLE PRECISION,
  geom         geometry(MultiPolygon, 4326) NOT NULL,
  created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX predictions_geom_gix ON predictions USING GIST (geom);
CREATE INDEX predictions_analysis_idx ON predictions (analysis_id);
CREATE INDEX predictions_class_idx ON predictions (class_id);
```
Area calculation: `ST_Area(geom::geography)` in square metres. Migrations are managed by Alembic in `services/geo/migrations/`; `infra/docker/postgis/init.sql` only enables the extension.

## Example bbox query used by the map
```sql
SELECT id, class_id, class_name, confidence, area_m2, ST_AsGeoJSON(geom)::json AS geometry
FROM predictions
WHERE geom && ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326)
  AND (:class_id IS NULL OR class_id = :class_id)
  AND (:analysis_id IS NULL OR analysis_id = :analysis_id)
LIMIT :limit;
```
