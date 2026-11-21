# cropwatch
Crop-health monitoring from aerial farm images. Upload an RGB aerial image and CropWatch marks nutrient deficiency, drydown and weed clusters as polygons on a map, with the area per class. A U-Net (PyTorch, served with ONNX Runtime) finds the stressed pixels; a FastAPI gateway, an inference service and a geo service store the polygons in PostGIS; a React + Leaflet frontend shows them. Runs locally with Docker Compose; tested on aerial images only (not satellite).

## Run the whole stack
Needs Docker (Docker Desktop, or the free Colima on macOS: `brew install colima docker docker-compose && colima start --cpu 6 --memory 5`).
```bash
cp .env.example .env            # then set POSTGIS_HOST_PORT=55432 if a local PostgreSQL already uses 5432
docker compose up -d --build postgis inference geo gateway frontend
docker compose ps               # all should show (healthy)
```
Open http://localhost:3000 and click a sample field, or upload your own image. API docs: http://localhost:8000/docs.

| Service | Port | Folder |
| --- | --- | --- |
| frontend (React + Vite + react-leaflet, nginx) | 3000 (dev server 5173) | `frontend/` |
| gateway (public API) | 8000 | `services/gateway/` |
| inference (ONNX Runtime) | 8001 | `services/inference/` |
| geo (only service using the database) | 8002 | `services/geo/` |
| PostGIS | `POSTGIS_HOST_PORT` (5432) | `infra/docker/postgis/` |

Frontend development: `cd frontend && npm ci && npm run dev` (uses `VITE_API_URL`, default http://localhost:8000). Each Python service has its own venv and tests: `python3.11 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt && .venv/bin/pytest -q` inside its folder. The geo database tests need `TEST_DATABASE_URL` (see `services/geo/tests/conftest.py`).

## Machine learning and inference

The model labels every pixel of an RGB aerial farm image as background, `nutrient_deficiency`, `drydown` or `weed_cluster`. The inference service turns that into map polygons with an area in m2 and a confidence.

### Current model (v3)
U-Net with a pretrained ResNet34 encoder, MLflow run `6d9eee7f`. Scored on 18,334 validation images from fields not used in training. IoU is the overlap between predicted and true area (0 to 1); the baseline predicts background everywhere.

| | v3 | Baseline (all background) |
| --- | --- | --- |
| mIoU without background | 0.371 | 0.000 |
| mIoU with background | 0.440 | 0.181 |
| nutrient_deficiency / drydown / weed_cluster | 0.315 / 0.558 / 0.241 | 0 / 0 / 0 |
| background | 0.645 | 0.722 |

It misses about a third of nutrient-deficiency pixels and almost half of weed pixels, and marks 29.5% of healthy ground as stress. Details and limitations: [model card](docs/model-card.md). Design decisions: [ADRs](docs/adr/).

**Data:** [Agriculture-Vision 2021](https://github.com/SHI-Labs/Agriculture-Vision), RGB channels only. Its terms allow non-commercial use only, and that applies to the trained model too. Of the dataset's 9 labels we keep 3; the other 6 (double_plant, endrow, planter_skip, storm_damage, water, waterway) count as background.

### Folders
| Path | What |
| --- | --- |
| `ml/` | Training code (PyTorch), its own venv `ml/.venv`. Never used by the services |
| `ml/scripts/` | `download_agvision.py`, `train_full.py`, `export_onnx.py` |
| `ml/configs/classes.yaml` | The 4 classes (frozen) |
| `services/inference/` | FastAPI + ONNX Runtime service, port 8001, its own venv. No PyTorch |
| `services/inference/models/` | `model.onnx`, `model_info.json`, `classes.yaml` (written by the export; v3 is committed, 98 MB) |

Commands below are for PowerShell on Windows, run from the repo root.

### 1. Training environment
Needs Python 3.11 and an NVIDIA GPU. The pinned PyTorch build (2.14.1, CUDA 13.0) supports recent GPUs such as the RTX 50 series.
```powershell
py -3.11 -m venv ml\.venv
ml\.venv\Scripts\python.exe -m pip install -r ml\requirements.txt
ml\.venv\Scripts\python.exe -m pip install -e ml
ml\.venv\Scripts\python.exe -m pytest ml\tests -q
```
In `ml/` code, always `import torch` before mlflow, pandas or pyarrow. The other order fails on Windows with `WinError 1114` (an old DLL shipped with pyarrow).

### 2. Data (about 21 GB download)
See the header of `ml/scripts/download_agvision.py` for the full steps. It extracts only the train and val RGB images, masks, boundaries and labels into `ml/data/agvision2021/`.
```powershell
curl.exe -L -C - -o ml\data\Agriculture-Vision-2021.tar.gz <URL from the script>
ml\.venv\Scripts\python.exe ml\scripts\download_agvision.py --source ml\data\Agriculture-Vision-2021.tar.gz
```

### 3. Train (about 3 hours on an 8 GB laptop GPU)
Start MLflow first (`docker compose up -d mlflow`, then http://localhost:5000).
```powershell
$env:MLFLOW_TRACKING_URI = "http://localhost:5000"
ml\.venv\Scripts\python.exe ml\scripts\train_full.py --model unet-resnet34 --epochs 20 --lr 3e-4 `
    --weight-decay 0.01 --encoder-lr-scale 0.1 --colour-jitter 0.2 --class-weight-power 0.25
```
Checkpoints go to `ml/checkpoints/<run id>/`. At the end the best epoch is scored on all validation images; those `final_*` metrics in MLflow are the ones to report, always next to the `final_baseline_*` metrics. For a quick trial of speed and GPU memory first, add `--max-train-images 2000 --val-images 200 --epochs 1 --skip-final-val --experiment cropwatch-trial`.

### 4. Export to ONNX
```powershell
ml\.venv\Scripts\python.exe ml\scripts\export_onnx.py --run-id <MLflow run id> --version <n>
```
This writes the three model files to `services/inference/models/` and checks ONNX Runtime against PyTorch on 32 validation tiles. Only use the export if the check reports (almost) all pixels agreeing. `model_info.json` holds the normalisation mean/std; the service reads it from there, so the three files must always be replaced together.

### 5. Run the inference service
In Docker (the v3 model files are in Git, so a fresh clone builds without steps 1-4):
```powershell
docker compose up -d --build inference
docker compose ps            # inference should show (healthy)
```
Or locally, with its own venv:
```powershell
cd services\inference
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.venv\Scripts\python.exe -m pytest -q
$env:MODEL_PATH = "models\model.onnx"; $env:MODEL_INFO_PATH = "models\model_info.json"
.venv\Scripts\python.exe -m uvicorn app.main:app --port 8001
```
Try it at http://localhost:8001/docs, or:
```powershell
curl.exe -F "file=@field.jpg" -F "bounds=-123.2,49.1,-123.19,49.11" http://localhost:8001/predict
```
`bounds` (minLon,minLat,maxLon,maxLat) is required for PNG/JPEG; a GeoTIFF with a CRS is placed by its own georeferencing. Limits: 8-bit RGB, at most 6000 x 6000 pixels (a 6000 px image takes about 40 s on a 16-thread laptop CPU). The full contract is in [docs/api.md](docs/api.md).

If rasterio fails with `proj.db ... comes from another PROJ installation`, a program such as PostgreSQL has set `PROJ_LIB` / `GDAL_DATA` for the whole machine. The service clears them for its own process when `app` is imported; make sure your code imports `app` before rasterio.
