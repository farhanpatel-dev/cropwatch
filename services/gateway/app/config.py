"""Settings from environment variables (names as in .env.example)."""

import os
from dataclasses import dataclass

# docs/api.md: the gateway waits at most 60 s for inference, then returns 504.
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

    @classmethod
    def from_env(cls) -> "Settings":
        origins = os.environ.get("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173")
        return cls(
            inference_url=os.environ.get("INFERENCE_URL", "http://inference:8001").rstrip("/"),
            geo_url=os.environ.get("GEO_URL", "http://geo:8002").rstrip("/"),
            max_upload_mb=int(os.environ.get("MAX_UPLOAD_MB", "50")),
            cors_origins=[o.strip() for o in origins.split(",") if o.strip()],
        )
