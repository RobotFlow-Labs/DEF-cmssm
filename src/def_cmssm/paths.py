from __future__ import annotations

from pathlib import Path

SHARED_INFRA_ROOT = Path("/mnt/forge-data/shared_infra")
SHARED_DATASETS_ROOT = Path("/mnt/forge-data/datasets")
SHARED_MODELS_ROOT = Path("/mnt/forge-data/models")


def ensure_path(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
