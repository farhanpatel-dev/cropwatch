"""Geo service: the only service that touches PostGIS. Internal (called by the gateway).

Endpoints (docs/api.md, "Geo : http://geo:8002"):
  GET  /health                    liveness plus SELECT 1
  POST /analyses                  create an analysis row -> 201
  POST /analyses/{id}/polygons    bulk insert a FeatureCollection -> 201 {"inserted": n}
  GET  /analyses, /analyses/{id}  read
  GET  /analyses/{id}/polygons    FeatureCollection
  GET  /polygons?bbox=&class_id=&analysis_id=&limit=
  GET  /stats?analysis_id=&bbox=  area per class (ST_Area on geography, never degrees)

Run locally (from services/geo, DATABASE_URL set):
  alembic upgrade head && uvicorn app.main:app --port 8002
"""

import json
import logging
import time
import uuid
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy import Connection, text

from app.db import get_engine
from app.schemas import AnalysisCreate, FeatureCollection, parse_bbox

log = logging.getLogger("geo")
REQUEST_ID_HEADER = "X-Request-ID"
MAX_LIMIT = 5000

ANALYSIS_COLUMNS = """
    id, name, source_filename, model_name, model_version, status,
    ST_AsGeoJSON(image_bounds)::json AS image_bounds, created_at
"""
FEATURE_COLUMNS = """
    id, analysis_id, class_id, class_name, confidence, area_m2,
    ST_AsGeoJSON(geom)::json AS geometry
"""


def get_conn():
    with get_engine().connect() as conn:
        yield conn


Conn = Annotated[Connection, Depends(get_conn)]


def analysis_dict(row) -> dict:
    return {
        "id": str(row.id),
        "name": row.name,
        "source_filename": row.source_filename,
        "model_name": row.model_name,
        "model_version": row.model_version,
        "status": row.status,
        "image_bounds": row.image_bounds,
        "created_at": row.created_at.isoformat(),
    }


def feature_collection(rows) -> dict:
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": r.geometry,
                "properties": {
                    "id": r.id,
                    "analysis_id": str(r.analysis_id),
                    "class_id": r.class_id,
                    "class_name": r.class_name,
                    "confidence": r.confidence,
                    "area_m2": r.area_m2,
                },
            }
            for r in rows
        ],
    }


def bbox_or_400(bbox: str | None) -> tuple[float, float, float, float] | None:
    if bbox is None:
        return None
    try:
        return parse_bbox(bbox)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def filters(
    analysis_id: uuid.UUID | None,
    bbox: tuple[float, float, float, float] | None,
    class_id: int | None = None,
) -> tuple[str, dict]:
    """WHERE clause and its parameters."""
    where, params = ["TRUE"], {}
    if analysis_id is not None:
        where.append("analysis_id = :analysis_id")
        params["analysis_id"] = analysis_id
    if class_id is not None:
        where.append("class_id = :class_id")
        params["class_id"] = class_id
    if bbox is not None:
        # ST_Intersects uses the GiST index (bounding-box test first), then the exact shape.
        where.append(
            "ST_Intersects(geom, ST_MakeEnvelope(:min_lon, :min_lat, :max_lon, :max_lat, 4326))"
        )
        params.update(zip(("min_lon", "min_lat", "max_lon", "max_lat"), bbox, strict=True))
    return " AND ".join(where), params


def create_app() -> FastAPI:
    app = FastAPI(title="CropWatch geo")

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        start = time.perf_counter()
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = rid
        log.info(
            "request_id=%s %s %s -> %s in %.0f ms",
            rid,
            request.method,
            request.url.path,
            response.status_code,
            (time.perf_counter() - start) * 1000,
        )
        return response

    @app.get("/health")
    def health(conn: Conn) -> dict:
        conn.execute(text("SELECT 1"))
        return {"status": "ok"}

    @app.post("/analyses", status_code=201)
    def create_analysis(body: AnalysisCreate, conn: Conn) -> dict:
        bounds = json.dumps(body.image_bounds.model_dump()) if body.image_bounds else None
        row = conn.execute(
            text(
                f"""
                INSERT INTO analyses (name, source_filename, model_name, model_version,
                                      image_bounds)
                VALUES (:name, :source_filename, :model_name, :model_version,
                        ST_SetSRID(ST_GeomFromGeoJSON(:bounds), 4326))
                RETURNING {ANALYSIS_COLUMNS}
                """
            ),
            {**body.model_dump(exclude={"image_bounds"}), "bounds": bounds},
        ).one()
        conn.commit()
        return analysis_dict(row)

    @app.get("/analyses")
    def list_analyses(
        conn: Conn, limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 100
    ) -> list[dict]:
        rows = conn.execute(
            text(f"SELECT {ANALYSIS_COLUMNS} FROM analyses ORDER BY created_at DESC LIMIT :n"),
            {"n": limit},
        )
        return [analysis_dict(r) for r in rows]

    def get_analysis_row(conn: Connection, analysis_id: uuid.UUID):
        row = conn.execute(
            text(f"SELECT {ANALYSIS_COLUMNS} FROM analyses WHERE id = :id"), {"id": analysis_id}
        ).one_or_none()
        if row is None:
            raise HTTPException(status_code=404, detail="analysis not found")
        return row

    @app.get("/analyses/{analysis_id}")
    def get_analysis(analysis_id: uuid.UUID, conn: Conn) -> dict:
        return analysis_dict(get_analysis_row(conn, analysis_id))

    @app.post("/analyses/{analysis_id}/polygons", status_code=201)
    def add_polygons(analysis_id: uuid.UUID, body: FeatureCollection, conn: Conn) -> dict:
        get_analysis_row(conn, analysis_id)
        rows = [
            {
                "analysis_id": analysis_id,
                "class_id": f.properties.class_id,
                "class_name": f.properties.class_name,
                "confidence": f.properties.confidence,
                "area_m2": f.properties.area_m2,
                "geom": json.dumps(f.geometry.model_dump()),
            }
            for f in body.features
        ]
        if rows:
            # One transaction: either every polygon is stored or none.
            conn.execute(
                text(
                    """
                    INSERT INTO predictions
                        (analysis_id, class_id, class_name, confidence, area_m2, geom)
                    VALUES (:analysis_id, :class_id, :class_name, :confidence, :area_m2,
                            ST_Multi(ST_SetSRID(ST_GeomFromGeoJSON(:geom), 4326)))
                    """
                ),
                rows,
            )
        conn.commit()
        return {"inserted": len(rows)}

    @app.get("/analyses/{analysis_id}/polygons")
    def analysis_polygons(analysis_id: uuid.UUID, conn: Conn) -> dict:
        get_analysis_row(conn, analysis_id)
        rows = conn.execute(
            text(f"SELECT {FEATURE_COLUMNS} FROM predictions WHERE analysis_id = :id ORDER BY id"),
            {"id": analysis_id},
        )
        return feature_collection(rows)

    @app.get("/polygons")
    def polygons(
        conn: Conn,
        bbox: str | None = None,
        class_id: int | None = None,
        analysis_id: uuid.UUID | None = None,
        limit: Annotated[int, Query(ge=1, le=MAX_LIMIT)] = 1000,
    ) -> dict:
        where, params = filters(analysis_id, bbox_or_400(bbox), class_id)
        rows = conn.execute(
            text(f"SELECT {FEATURE_COLUMNS} FROM predictions WHERE {where} ORDER BY id LIMIT :n"),
            {**params, "n": limit},
        )
        return feature_collection(rows)

    @app.get("/stats")
    def stats(
        conn: Conn, analysis_id: uuid.UUID | None = None, bbox: str | None = None
    ) -> list[dict]:
        # With bbox, a polygon counts with its whole area if it intersects the box (no clipping).
        where, params = filters(analysis_id, bbox_or_400(bbox))
        rows = conn.execute(
            text(
                f"""
                SELECT class_id, class_name, count(*) AS polygon_count,
                       sum(ST_Area(geom::geography)) / 10000.0 AS area_ha
                FROM predictions WHERE {where}
                GROUP BY class_id, class_name ORDER BY class_id
                """
            ),
            params,
        )
        return [
            {
                "class_id": r.class_id,
                "class_name": r.class_name,
                "polygon_count": r.polygon_count,
                "area_ha": round(float(r.area_ha), 4),
            }
            for r in rows
        ]

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
