"""Gateway: the only public API. Validates uploads, calls inference, then stores via geo.

Endpoints (docs/api.md, "Gateway (public)"), request flow in docs/architecture.md:
  GET  /health
  GET  /api/v1/classes
  GET  /api/v1/model-info              inference's /model-info (also wakes a sleeping inference)
  POST /api/v1/analyses                 upload -> inference /predict -> geo -> 201
  GET  /api/v1/analyses, /api/v1/analyses/{id}, /api/v1/analyses/{id}/polygons
  GET  /api/v1/polygons?bbox=&class_id=&analysis_id=&limit=
  GET  /api/v1/stats?analysis_id=&bbox=

Errors: inference 400/413 are passed on with their detail; inference unreachable or 5xx ->
502; inference slower than 60 s -> 504; geo errors -> 502 (404 and 400 are passed on).

Run locally (from services/gateway): uvicorn app.main:app --port 8000
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import PurePath
from typing import Annotated

import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import CLASSES, Settings

log = logging.getLogger("gateway")
REQUEST_ID_HEADER = "X-Request-ID"
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
NEEDS_BOUNDS = {".png", ".jpg", ".jpeg"}


def parse_bounds(text: str) -> list[float]:
    """'minLon,minLat,maxLon,maxLat' -> four floats; raises 400 with a clear message."""
    try:
        values = [float(v) for v in text.split(",")]
    except ValueError:
        values = []
    if len(values) != 4:
        raise HTTPException(400, "bounds must be four numbers: minLon,minLat,maxLon,maxLat")
    min_lon, min_lat, max_lon, max_lat = values
    if not (min_lon < max_lon and min_lat < max_lat):
        raise HTTPException(400, "bounds must be minLon,minLat,maxLon,maxLat with min < max")
    if not (-180 <= min_lon and max_lon <= 180 and -90 <= min_lat and max_lat <= 90):
        raise HTTPException(400, "bounds must be longitude -180..180 and latitude -90..90")
    return values


def bounds_polygon(b: list[float]) -> dict:
    """GeoJSON Polygon for [minLon, minLat, maxLon, maxLat], closed ring."""
    min_lon, min_lat, max_lon, max_lat = b
    ring = [
        [min_lon, min_lat],
        [max_lon, min_lat],
        [max_lon, max_lat],
        [min_lon, max_lat],
        [min_lon, min_lat],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def public_analysis(row: dict, stats: list | None = None) -> dict:
    """Geo analysis row -> the gateway's analysis shape (docs/api.md, POST /api/v1/analyses)."""
    out = {
        "id": row["id"],
        "name": row.get("name"),
        "source_filename": row.get("source_filename"),
        "status": row["status"],
        "created_at": row["created_at"],
        "model": {"name": row["model_name"], "version": row["model_version"]},
        "image_bounds": row.get("image_bounds"),
    }
    if stats is not None:
        out["stats"] = stats
    return out


def detail_of(response: httpx.Response) -> str:
    try:
        return str(response.json().get("detail", response.text))
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


def create_app(settings: Settings | None = None, transport: httpx.AsyncBaseTransport | None = None):
    """transport lets tests replace the network with fake inference and geo services."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async with httpx.AsyncClient(transport=transport) as client:
            app.state.http = client
            yield

    app = FastAPI(title="CropWatch gateway", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=[REQUEST_ID_HEADER],
    )

    @app.middleware("http")
    async def request_id(request: Request, call_next):
        rid = request.headers.get(REQUEST_ID_HEADER) or uuid.uuid4().hex
        request.state.request_id = rid
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

    async def geo(request: Request, method: str, path: str, **kwargs) -> httpx.Response:
        """Call the geo service; network errors and 5xx become 502, 400/404 pass through."""
        client: httpx.AsyncClient = request.app.state.http
        headers = {REQUEST_ID_HEADER: request.state.request_id}
        try:
            r = await client.request(
                method, f"{settings.geo_url}{path}", headers=headers, timeout=15, **kwargs
            )
        except httpx.HTTPError as e:
            log.warning("geo unreachable: %r", e)
            raise HTTPException(502, "geo service unreachable") from e
        if r.status_code in (400, 404):
            raise HTTPException(r.status_code, detail_of(r))
        if r.status_code >= 300:
            log.warning("geo %s %s -> %s: %s", method, path, r.status_code, r.text[:500])
            raise HTTPException(502, f"geo service error ({r.status_code})")
        return r

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/v1/classes")
    def classes() -> list[dict]:
        return CLASSES

    @app.get("/api/v1/model-info")
    async def model_info(request: Request) -> dict:
        """Which model is serving. The frontend calls this on page load, so on a host that
        scales to zero, inference starts waking up while the user picks an image."""
        client: httpx.AsyncClient = request.app.state.http
        try:
            r = await client.get(
                f"{settings.inference_url}/model-info",
                headers={REQUEST_ID_HEADER: request.state.request_id},
                timeout=settings.inference_timeout_s,
            )
        except httpx.HTTPError as e:
            raise HTTPException(502, "inference service unreachable") from e
        if r.status_code != 200:
            raise HTTPException(502, f"inference service error ({r.status_code})")
        return r.json()

    @app.post("/api/v1/analyses", status_code=201)
    async def create_analysis(
        request: Request,
        file: Annotated[UploadFile, File()],
        bounds: Annotated[str | None, Form()] = None,
        name: Annotated[str | None, Form()] = None,
    ) -> dict:
        filename = file.filename or "upload"
        ext = PurePath(filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(400, "file type must be .tif, .tiff, .png, .jpg or .jpeg")
        max_bytes = settings.max_upload_mb * 1024 * 1024
        data = await file.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise HTTPException(413, f"file is larger than {settings.max_upload_mb} MB")
        if not data:
            raise HTTPException(400, "empty file")
        if ext in NEEDS_BOUNDS:
            if not bounds or not bounds.strip():
                raise HTTPException(400, "bounds are required for PNG and JPEG images")
            bounds = ",".join(str(v) for v in parse_bounds(bounds))
        else:
            bounds = None  # a GeoTIFF is placed by its own CRS

        # 1. Inference
        client: httpx.AsyncClient = request.app.state.http
        form = {"bounds": bounds} if bounds else {}
        try:
            # httpx timeouts apply per network operation, so a total deadline is set here.
            async with asyncio.timeout(settings.inference_timeout_s):
                r = await client.post(
                    f"{settings.inference_url}/predict",
                    files={
                        "file": (filename, data, file.content_type or "application/octet-stream")
                    },
                    data=form,
                    headers={REQUEST_ID_HEADER: request.state.request_id},
                    timeout=None,
                )
        except (TimeoutError, httpx.TimeoutException) as e:
            raise HTTPException(504, "inference took too long; try a smaller image") from e
        except httpx.HTTPError as e:
            log.warning("inference unreachable: %r", e)
            raise HTTPException(502, "inference service unreachable") from e
        if r.status_code in (400, 413):
            raise HTTPException(r.status_code, detail_of(r))
        if r.status_code != 200:
            log.warning("inference -> %s: %s", r.status_code, r.text[:500])
            raise HTTPException(502, f"inference service error ({r.status_code})")
        result = r.json()

        # 2. Store in geo, then read the stats back
        analysis = (
            await geo(
                request,
                "POST",
                "/analyses",
                json={
                    "name": name or None,
                    "source_filename": filename,
                    "model_name": result["model"]["name"],
                    "model_version": str(result["model"]["version"]),
                    "image_bounds": bounds_polygon(result["image"]["bounds"]),
                },
            )
        ).json()
        aid = analysis["id"]
        await geo(request, "POST", f"/analyses/{aid}/polygons", json=result["polygons"])
        stats = (await geo(request, "GET", "/stats", params={"analysis_id": aid})).json()
        return public_analysis(analysis, stats)

    @app.get("/api/v1/analyses")
    async def list_analyses(request: Request, limit: int = 100) -> list[dict]:
        rows = (await geo(request, "GET", "/analyses", params={"limit": limit})).json()
        return [public_analysis(row) for row in rows]

    @app.get("/api/v1/analyses/{analysis_id}")
    async def get_analysis(request: Request, analysis_id: uuid.UUID) -> dict:
        row = (await geo(request, "GET", f"/analyses/{analysis_id}")).json()
        stats = (await geo(request, "GET", "/stats", params={"analysis_id": analysis_id})).json()
        return public_analysis(row, stats)

    @app.get("/api/v1/analyses/{analysis_id}/polygons")
    async def analysis_polygons(request: Request, analysis_id: uuid.UUID) -> dict:
        return (await geo(request, "GET", f"/analyses/{analysis_id}/polygons")).json()

    @app.get("/api/v1/polygons")
    async def polygons(
        request: Request,
        bbox: str | None = None,
        class_id: int | None = None,
        analysis_id: uuid.UUID | None = None,
        limit: int = 1000,
    ) -> dict:
        params = {"bbox": bbox, "class_id": class_id, "analysis_id": analysis_id, "limit": limit}
        params = {k: str(v) for k, v in params.items() if v is not None}
        return (await geo(request, "GET", "/polygons", params=params)).json()

    @app.get("/api/v1/stats")
    async def stats(
        request: Request, analysis_id: uuid.UUID | None = None, bbox: str | None = None
    ) -> list[dict]:
        params = {k: str(v) for k, v in {"analysis_id": analysis_id, "bbox": bbox}.items() if v}
        return (await geo(request, "GET", "/stats", params=params)).json()

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
