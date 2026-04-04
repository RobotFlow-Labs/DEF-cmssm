from __future__ import annotations

import enum
import os


class Backend(str, enum.Enum):
    CUDA = "cuda"
    MLX = "mlx"


def resolve_backend(explicit: str | None = None) -> Backend:
    """Resolve execution backend from explicit arg or ANIMA_BACKEND env var."""
    raw = (explicit or os.environ.get("ANIMA_BACKEND") or "cuda").strip().lower()
    if raw not in {Backend.CUDA.value, Backend.MLX.value}:
        raise ValueError(
            f"Unsupported backend '{raw}'. Use ANIMA_BACKEND in {{'cuda','mlx'}}."
        )
    return Backend(raw)
