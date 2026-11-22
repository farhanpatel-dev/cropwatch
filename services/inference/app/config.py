"""Settings from environment variables (names as in .env.example; defaults fit the Docker image)."""

import os
from dataclasses import dataclass
from pathlib import Path

# docs/api.md, ADR 0003: larger images are rejected with 413. Lowered from 8000 to 6000 on
# 2026-11-08 and to 4000 on 2026-11-20: 6000 x 6000 took 74.6 s in Docker on an 8-core
# laptop (over the gateway's 60 s timeout); 4000 x 4000 took 26-31 s end to end.
MAX_PIXELS_SIDE = 4000


@dataclass(frozen=True)
class Settings:
    model_path: Path
    model_info_path: Path
    tile_size: int
    tile_overlap: int
    # Proposal (cloud hosting, ADR 0004): /predict may fetch the image from a URL instead of
    # an upload, but only from these prefixes (the S3 upload bucket). Empty: URLs refused.
    image_url_prefixes: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_path=Path(os.environ.get("MODEL_PATH", "/app/models/model.onnx")),
            model_info_path=Path(os.environ.get("MODEL_INFO_PATH", "/app/models/model_info.json")),
            tile_size=int(os.environ.get("TILE_SIZE", "256")),
            tile_overlap=int(os.environ.get("TILE_OVERLAP", "32")),
            image_url_prefixes=tuple(
                p.strip() for p in os.environ.get("IMAGE_URL_PREFIXES", "").split(",") if p.strip()
            ),
        )
