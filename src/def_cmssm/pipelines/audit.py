from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AuditResult:
    proposed_import_hits: int = 0
    absolute_path_hits: int = 0
    model_others_hits: int = 0


def _count_hits(text: str, token: str) -> int:
    return text.count(token)


def audit_cmssm_repo(repo_root: Path) -> AuditResult:
    result = AuditResult()
    for py_file in repo_root.rglob("*.py"):
        if "__pycache__" in py_file.parts:
            continue
        content = py_file.read_text(encoding="utf-8", errors="ignore")
        result.proposed_import_hits += _count_hits(content, "proposed.")
        result.absolute_path_hits += _count_hits(content, "/home/ubuntu/")
        result.absolute_path_hits += _count_hits(content, "/root/autodl-tmp/")
        result.model_others_hits += _count_hits(content, "model_others.")
    return result
