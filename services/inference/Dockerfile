# Inference service image: FastAPI + ONNX Runtime, no PyTorch.
# Build from the repo root:  docker compose build inference
# The model files (models/model.onnx, model_info.json, classes.yaml) are copied in, so they
# must exist in services/inference/models/ before building (written by ml/scripts/export_onnx.py;
# not in Git). Without them the container starts and exits with a clear error.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# rasterio's wheel bundles GDAL, which still needs the system XML library libexpat
# (missing from the slim image: "ImportError: libexpat.so.1").
RUN apt-get update \
    && apt-get install -y --no-install-recommends libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first: this layer is reused as long as requirements.txt does not change.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/ app/
COPY models/ models/

# Run as a normal user, not root.
RUN useradd --create-home --uid 10001 appuser
USER appuser

ENV MODEL_PATH=/app/models/model.onnx \
    MODEL_INFO_PATH=/app/models/model_info.json \
    TILE_SIZE=256 \
    TILE_OVERLAP=32

EXPOSE 8001

HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8001/health')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8001"]
