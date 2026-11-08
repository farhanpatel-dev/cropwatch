"""Settings from environment variables (names as in .env.example; defaults fit the Docker image)."""

import os
from dataclasses import dataclass
from pathlib import Path

# docs/api.md: larger images are rejected with 413. Lowered from 8000 on 2026-11-08: a
# 6000 x 6000 image took about 41 s on the dev laptop CPU; 8000 x 8000 would exceed the
# gateway's 60 s timeout.
MAX_PIXELS_SIDE = 6000


@dataclass(frozen=True)
class Settings:
    model_path: Path
    model_info_path: Path
    tile_size: int
    tile_overlap: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            model_path=Path(os.environ.get("MODEL_PATH", "/app/models/model.onnx")),
            model_info_path=Path(os.environ.get("MODEL_INFO_PATH", "/app/models/model_info.json")),
            tile_size=int(os.environ.get("TILE_SIZE", "256")),
            tile_overlap=int(os.environ.get("TILE_OVERLAP", "32")),
        )
