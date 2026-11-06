# Architecture and tech decisions

Copy of project file 02. Changes agreed on 2026-11-04: training runs on a local computer (no Colab); frontend hosted on Vercel.

## Components and ports
| Component | Port | Responsibility |
| --- | --- | --- |
| frontend (React, served by nginx in container; Vite dev server 5173) | 3000 | Upload form, Leaflet map, legend, stats |
| gateway (FastAPI) | 8000 | Only public API; validates uploads; orchestrates inference and geo |
| inference (FastAPI + ONNX Runtime) | 8001 | Image in, georeferenced polygons out. Stateless |
| geo (FastAPI + SQLAlchemy/GeoAlchemy2) | 8002 | Owns the database; stores and queries polygons |
| postgis (PostgreSQL 16 + PostGIS) | 5432 | Spatial database |
| mlflow | 5000 | Experiment tracking and model registry (SQLite backend + artifact volume) |
| redis (stretch) | 6379 | Job queue for async analysis |

## Request flow (POST /api/v1/analyses)
1. Browser sends multipart file (and bounds if needed) to the gateway.
2. Gateway validates type, size (max 50 MB) and bounds; keeps the file in memory or a temp path.
3. Gateway calls inference `POST /predict` (timeout 60 s).
4. Inference reads the image, normalises, cuts overlapping 256x256 tiles, runs the ONNX model, stitches the class mask, polygonizes with rasterio, simplifies, drops tiny shapes, reprojects to EPSG:4326, computes area and returns a GeoJSON FeatureCollection.
5. Gateway calls geo `POST /analyses`, then `POST /analyses/{id}/polygons`, then `GET /stats?analysis_id=`.
6. Gateway returns 201 with analysis id, model info, image bounds and stats.
7. Browser calls `GET /api/v1/analyses/{id}/polygons` (or bbox query) and draws the layer.

```
Browser -> gateway:8000 -> inference:8001   (predict)
                      \-> geo:8002 -> postgis:5432   (store/query)
ml/ (local computer) -> MLflow:5000 -> model.onnx -> services/inference/models/
```

## Rules
- Services communicate only via HTTP/JSON per 03-api-contract. No shared code imports.
- Only geo connects to PostGIS.
- Every service exposes GET /health and logs a request id.
- Config only from environment variables (.env.example). No secrets in Git.
- Model hand-off: `model.onnx` + `classes.yaml` (id, name, colour). Inference reports model name/version from `model_info.json` next to the file and takes its input normalisation (`preprocessing`: input_scale, per-channel mean and std from training) from the same file.

## Tech decisions (with alternatives)
| Decision | Choice | Reason | Alternatives at larger scale |
| --- | --- | --- | --- |
| DL library | PyTorch | Pythonic, standard in vision | TensorFlow, JAX |
| Model | U-Net (own implementation, then pretrained encoder via segmentation_models_pytorch) | Learn skip connections, then train faster | DeepLabV3+, SegFormer |
| Data | Agriculture-Vision subset (RGB channels only for the MVP) | Public, segmentation-ready; RGB so PNG/JPEG uploads work | Sentinel-2, commercial imagery |
| Training compute | Team's own local computer, logging to the Compose MLflow | Free, no session time-outs | Azure ML, SageMaker, Vertex AI |
| Tracking | MLflow | Open source, simple | Weights and Biases |
| Serving format | ONNX Runtime CPU | Small image, free hosts | Triton, TorchServe, GPU |
| API | FastAPI | Typed, auto docs | Django REST, Go/Java |
| DB | PostgreSQL + PostGIS (GiST index) | Spatial SQL; free on Neon | BigQuery GIS, Oracle Spatial |
| Frontend | React + Vite + react-leaflet | Simple map | MapLibre GL, deck.gl |
| Packaging | Docker, Compose | Reproducible | + image signing |
| Orchestration | Kubernetes on kind, Kustomize | Free local practice | AKS/EKS/GKE, Helm, Argo CD |
| CI/CD | GitHub Actions, GHCR | Free for public repos | GitLab CI, Azure DevOps |
| Cloud | Azure training lab only | No cost goal | AWS/GCP/Azure with IaC |

## Hosting decision
- Frontend: Vercel Hobby plan (free; personal, non-commercial use only; re-check Vercel's terms and limits before relying on them). Use Vercel's GitHub integration and set `VITE_API_URL` in the Vercel project settings.
- Backend: not hosted on Vercel (Vercel runs short-lived serverless functions, not long-running containers with ONNX Runtime and rasterio). Default: recorded demo of the local stack; a free backend host is optional (cut order item 2).

## MLflow tracking URI
- From the training script on the host computer: `MLFLOW_TRACKING_URI=http://localhost:5000` (Compose publishes port 5000).
- From inside a Compose container: `http://mlflow:5000` (the value in `.env.example`).

## Free hosting options (checked 2026-11-03; re-verify before use)
- Static frontend: GitHub Pages, Cloudflare Pages, Netlify. Vercel Hobby is free but personal, non-commercial only.
- Database: Neon free plan (PostGIS supported; roughly 0.5 GB per project per its FAQ; suspends when monthly compute hours are used up).
- Backend: Render free web service (512 MB RAM, sleeps after 15 min idle, ~1 min wake). Do not rely on Render's free Postgres (it expires).
- Whole stack on Kubernetes: Oracle Cloud Always Free Arm VM (up to 4 OCPU/24 GB; card needed for verification).
- Not recommended: Hugging Face Docker Spaces (need a paid plan for creators).
- Zero-risk demo: static site with a pre-computed GeoJSON result plus a recorded video.
- Azure: free account ($200 credit for 30 days; card needed for verification) or Azure for Students ($100 for 12 months; no card; school email). Training lab only; delete the resource group afterwards.

## Environment variables (.env.example)
```
POSTGRES_USER=cropwatch
POSTGRES_PASSWORD=change_me_locally
POSTGRES_DB=cropwatch
DATABASE_URL=postgresql+psycopg://cropwatch:change_me_locally@postgis:5432/cropwatch
INFERENCE_URL=http://inference:8001
GEO_URL=http://geo:8002
MODEL_PATH=/app/models/model.onnx
MODEL_INFO_PATH=/app/models/model_info.json
TILE_SIZE=256
TILE_OVERLAP=32
MAX_UPLOAD_MB=50
MLFLOW_TRACKING_URI=http://mlflow:5000
VITE_API_URL=http://localhost:8000
```

## Kubernetes mapping
Compose service -> Deployment + Service; published port -> Ingress or port-forward; .env -> ConfigMap + Secret; volume -> PersistentVolumeClaim; healthcheck -> readiness/liveness probes. Namespace: `cropwatch`. Overlays: `local-kind`, `azure-lab`.
