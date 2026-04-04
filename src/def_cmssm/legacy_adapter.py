from __future__ import annotations

from pathlib import Path


LEGACY_IMPORT_FIXES: tuple[tuple[str, str], ...] = (
    ("from proposed.", "from models."),
    ("import proposed.", "import models."),
    ("backbone.covnextV2", "backbone.convnextV2"),
)


def patch_text(content: str) -> str:
    updated = content
    for src, dst in LEGACY_IMPORT_FIXES:
        updated = updated.replace(src, dst)
    return updated


def patch_file(path: Path) -> bool:
    original = path.read_text(encoding="utf-8")
    updated = patch_text(original)
    if updated == original:
        return False
    path.write_text(updated, encoding="utf-8")
    return True
