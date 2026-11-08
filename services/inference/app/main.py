"""Inference service: image in, stress polygons (GeoJSON, EPSG:4326) out. Stateless.

Endpoints (docs/api.md, "Inference : http://inference:8001"):
  GET  /health       {"status": "ok"}
  GET  /model-info   name, version, MLflow run id, tile size, classes, preprocessing
  POST /predict      multipart `file`, optional `bounds` -> model, image, polygons

Run locally (from services/inference, with MODEL_PATH and MODEL_INFO_PATH set):
  .venv\\Scripts\\python.exe -m uvicorn app.main:app --port 8001
"""

import logging
import time
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


def create_app(segmenter: Segmenter | None = None) -> FastAPI:
    """segmenter=None loads the model from env settings at startup (tests pass a fake)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.segmenter = segmenter or load_segmenter(Settings.from_env())
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
        file: Annotated[UploadFile, File()],
        bounds: Annotated[str | None, Form()] = None,
    ) -> dict:
        seg: Segmenter = request.app.state.segmenter
        try:
            placed = read_image(file.file.read(), bounds)
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
