"""Settings from environment variables (names as in .env.example)."""

import os
from dataclasses import dataclass

# docs/api.md: the gateway waits at most 60 s for inference (INFERENCE_TIMEOUT_S), then 504.
INFERENCE_TIMEOUT_S = 60.0

# Frozen class list (docs/api.md "Classes", ml/configs/classes.yaml). Services never import
# each other, so the gateway keeps its own copy for GET /api/v1/classes.
CLASSES = [
    {"id": 0, "name": "background", "colour": "transparent"},
    {"id": 1, "name": "nutrient_deficiency", "colour": "#e6194b"},
    {"id": 2, "name": "drydown", "colour": "#f58231"},
    {"id": 3, "name": "weed_cluster", "colour": "#911eb4"},
]


@dataclass(frozen=True)
class Settings:
    inference_url: str
    geo_url: str
    max_upload_mb: int
    cors_origins: list[str]
    inference_timeout_s: float = INFERENCE_TIMEOUT_S
    # AWS hosting only (ADR 0004). Empty / false locally.
    upload_bucket: str = ""
    aws_region: str | None = None
    sign_requests: bool = False

    @classmethod
    def from_env(cls) -> "Settings":
        origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
        return cls(
            inference_url=os.environ.get("INFERENCE_URL", "http://inference:8001").rstrip("/"),
            geo_url=os.environ.get("GEO_URL", "http://geo:8002").rstrip("/"),
            max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "50")),
            inference_timeout_s=float(os.environ.get("INFERENCE_TIMEOUT_S") or INFERENCE_TIMEOUT_S),
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
            upload_bucket=os.environ.get("UPLOAD_BUCKET", ""),
            aws_region=os.environ.get("AWS_REGION"),
            sign_requests=os.environ.get("SIGN_REQUESTS", "").lower() in ("1", "true", "yes"),
        )
