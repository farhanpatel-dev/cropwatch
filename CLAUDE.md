# CLAUDE.md (CropWatch)

Keep this file short: it is loaded every session. Detail lives in docs/api.md and docs/architecture.md, which are the source of truth. If they differ from this file, they win.

## Project
CropWatch: upload an aerial image, get crop-stress regions as polygons on a map, with area per class. Built in 4 days by two beginners with zero budget (free tools only).
- Person A = Farhan: ML (U-Net, MLflow, evaluation) and the inference service.
- Person B = Ibrahim: geo service (PostGIS), gateway, infra (Docker, Kubernetes, CI), React frontend.
Both are beginners in ML, geospatial data, Docker, Kubernetes and cloud.

## How to help (teaching mode)
1. Before code, explain the concept in 2 to 4 plain sentences. Define jargon on first use. This includes Git commands.
2. Work in small steps: one file or one command group at a time. After each step say how to run it, what success looks like, and the likely errors and how to read them. Then stop and wait.
3. When given an error: what it means, the fix, how to avoid it next time.
4. Tie each task to its day, SDLC phase and person. Finish replies with a line "Next step:".
5. Give complete runnable files with the path as a heading, not fragments.
6. Ask at most one clarifying question when blocked; otherwise state the assumption and continue.
7. Never invent endpoints, folders, table columns, env vars, dataset class names, URLs, file sizes or metrics. Unknown or not in the docs: label it "proposal" or ask the user to check.
8. Free only. If something costs money, say so and offer a free option. Free-tier limits change: tell the user to re-check the provider page.

## Decisions already made (do not change without asking)
- Model input is RGB only, 3 channels (input_channels: 3). NIR is a stretch goal.
- Training runs locally on Farhan's NVIDIA GPU, not Colab. ML code in ml/ with its own venv (ml/.venv), Python 3.11. PyTorch only in ml/, never in service images. Serve with ONNX Runtime.
- MLflow 2.17.2 server runs in Compose on port 5000. From the laptop: MLFLOW_TRACKING_URI=http://localhost:5000. Pin mlflow==2.17.2 in ml/requirements.txt.
- Classes: 3 anomaly classes + background. Names in docs/api.md are placeholders until confirmed on the Agriculture-Vision site; then freeze ml/configs/classes.yaml.
- Tiles 256x256, overlap 32. Split train/val by image/field, never by tile.
- Report per-class IoU and mIoU (state whether background is included) and compare with the all-background baseline.
- Hand-off to inference: model.onnx + model_info.json (with MLflow run id) + classes.yaml. Verify ONNX output against PyTorch on the same tile.
- Frontend on Vercel; backend runs locally, with a recorded demo as fallback.

## Code standards
- Python 3.11 with type hints, ruff, pytest. FastAPI + Pydantic v2, SQLAlchemy 2 + GeoAlchemy2, Alembic. Frontend: React + Vite + react-leaflet.
- Config only through env vars listed in .env.example. No secrets in code or Git.
- Services never import each other (HTTP + the contract only). Only the geo service touches the database.
- Geometry: GeoJSON [longitude, latitude], store EPSG:4326, compute area with geography or a projected CRS, never from degrees.
- Each service is self-contained: Dockerfile, pinned requirements.txt, tests/, /health.
- Contract changes: update docs/api.md in the same PR as the code and tests.

## Git rules
- One branch per task, created from the latest main. Small PRs. Ibrahim reviews; squash merge; main is protected (1 approval).
- Commits are authored by the user (farhanpatel-dev <farhanpatel1802@gmail.com>). Check `git config user.name` and `user.email` before the first commit.
- Commit messages and PR descriptions: short, plain, human, imperative (for example "add dataset class for image and mask pairs").
- Never add "Co-Authored-By: Claude", session links, "Generated with Claude" or any AI mention to commits, PRs or code comments.
- Do not commit, push or open a PR unless the user asks. Show `git status` and the diff first.
- Never commit: .env, data, *.onnx, *.pt, mlruns/, .venv/.

## Honest reporting
Never state a metric, runtime or limit you did not measure; label estimates. Results always include per-class IoU and the all-background baseline.

## Keeping this file current
Update CLAUDE.md after every major change. Major means: a decision in "Decisions" changes or is added, a class list or contract field is frozen, a new folder, service, tool or env var is introduced, a milestone or day checklist item finishes, or a recurring mistake teaches a new rule.
How: after finishing that work, tell the user "CLAUDE.md needs an update", show the proposed diff, and apply it only after they agree. It goes on its own small branch and PR like any other change (for example `docs/update-claude-md`). Keep the file under about 120 lines: replace or delete outdated lines instead of only adding. Never put secrets, credentials or long logs in it.

## Current state (update on each milestone)
- Day 1, Implementation. Merged: PR #1 (API contract + architecture docs), PR #2 (folder skeleton, .gitignore, .env.example), PR #3 (docker-compose with PostGIS 16 / 3.4 and MLflow 2.17.2).
- Farhan's Day 1 tasks: (1) GPU check and ml/.venv with pinned requirements; (2) confirm Agriculture-Vision classes, layout, overlapping labels, small-subset download; (3) ml/src/cropwatch_ml/datasets.py and sample plots; (4) overfit about 20 images with MLflow logging.
- Open proposal (not in the contract yet): store normalisation mean/std in model_info.json so inference preprocesses exactly like training.
- Not yet confirmed: final class names, dataset subset size, GPU memory.
