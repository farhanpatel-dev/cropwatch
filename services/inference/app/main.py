"""Inference service: image in, stress polygons (GeoJSON, EPSG:4326) out. Stateless.

Endpoints (docs/api.md, "Inference : http://inference:8001"):
  GET  /health       {"status": "ok"}
  GET  /model-info   name, version, MLflow run id, tile size, classes, preprocessing
  POST /predict      multipart `file` (or `image_url`), optional `bounds` -> model, image, polygons

Run locally (from services/inference, with MODEL_PATH and MODEL_INFO_PATH set):
  .venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8001
"""

import logging
import time
import urllib.error
import urllib.request
import uuid
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from app.config import Settings
from app.images import ImageError, read_image
from app.model import Segmenter
from app.polygons import image_bounds, polygonize

log = logging.getLogger("inference")
# Proposal: header name is not fixed in the docs yet. Reused if the gateway sends one.
REQUEST_ID_HEADER = "X-Request-ID"
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # same as the gateway's MAX_UPLOAD_MB default


def download(url: str, prefixes: tuple[str, ...]) -> bytes:
    """Fetch an image from an allowed URL (the gateway's S3 upload bucket, ADR 0004)."""
    if not prefixes or not any(url.startswith(p) for p in prefixes):
        raise HTTPException(status_code=400, detail="image_url is not allowed")
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read(MAX_DOWNLOAD_BYTES + 1)
    except (urllib.error.URLError, TimeoutError) as e:
        raise HTTPException(status_code=400, detail="could not download image_url") from e
    if len(data) > MAX_DOWNLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image is larger than 50 MB")
    return data


def load_segmenter(settings: Settings) -> Segmenter:
    seg = Segmenter.from_paths(
        settings.model_path, settings.model_info_path, overlap=settings.tile_overlap
    )
    if seg.info.tile_size != settings.tile_size:
        raise RuntimeError(
            f"TILE_SIZE={settings.tile_size} but the model was trained on "
            f"{seg.info.tile_size}; fix TILE_SIZE or the model files"
        )
    return seg


def create_app(segmenter: Segmenter | None = None, settings: Settings | None = None) -> FastAPI:
    """segmenter=None loads the model from env settings at startup (tests pass a fake)."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.segmenter = segmenter or load_segmenter(settings)
        info = app.state.segmenter.info
        log.info(
            "model loaded: %s version %s (run %s)", info.name, info.version, info.mlflow_run_id
        )
        yield

    app = FastAPI(title="CropWatch inference", lifespan=lifespan)

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
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/model-info")
    def model_info(request: Request) -> dict:
        info = request.app.state.segmenter.info
        return {
            "name": info.name,
            "version": info.version,
            "mlflow_run_id": info.mlflow_run_id,
            "input_channels": info.input_channels,
            "tile_size": info.tile_size,
            "classes": info.classes,
            "preprocessing": {"mean": info.mean, "std": info.std},
        }

    # Plain `def` (not async): FastAPI runs it in a worker thread, so a long prediction does
    # not block /health while it runs.
    @app.post("/predict")
    def predict(
        request: Request,
        file: Annotated[UploadFile | None, File()] = None,
        image_url: Annotated[str | None, Form()] = None,
        bounds: Annotated[str | None, Form()] = None,
    ) -> dict:
        seg: Segmenter = request.app.state.segmenter
        if file is not None:
            data = file.file.read()
        elif image_url:
            data = download(image_url, settings.image_url_prefixes)
        else:
            raise HTTPException(status_code=422, detail="send a file (or image_url)")
        try:
            placed = read_image(data, bounds)
        except ImageError as e:
            raise HTTPException(status_code=e.status_code, detail=str(e)) from e
        h, w = placed.rgb.shape[:2]
        classes, confidence = seg.predict(placed.rgb)
        names = {c["id"]: c["name"] for c in seg.info.classes}
        polygons = polygonize(classes, confidence, placed.transform, placed.crs, names)
        return {
            "model": {"name": seg.info.name, "version": seg.info.version},
            "image": {
                "width": w,
                "height": h,
                "crs": "EPSG:4326",
                "bounds": image_bounds(placed.transform, w, h, placed.crs),
            },
            "polygons": polygons,
        }

    return app


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
app = create_app()
