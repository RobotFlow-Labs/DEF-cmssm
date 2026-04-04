from __future__ import annotations

import argparse
from pathlib import Path

from def_cmssm.config import ModuleConfig
from def_cmssm.device import resolve_backend
from def_cmssm.pipelines.audit import audit_cmssm_repo


def run_scaffold(root: Path, backend: str | None = None) -> dict[str, str]:
    cfg = ModuleConfig()
    resolved = resolve_backend(backend)
    paths = cfg.paths(root)

    status = {
        "module": cfg.module_name,
        "backend": resolved.value,
        "cmssm_repo_exists": str(paths.cmssm_repo.exists()),
        "paper_pdf_exists": str((paths.papers_dir / f"{cfg.paper_id}.pdf").exists()),
    }
    if paths.cmssm_repo.exists():
        audit = audit_cmssm_repo(paths.cmssm_repo)
        status["audit.proposed_import_hits"] = str(audit.proposed_import_hits)
        status["audit.absolute_path_hits"] = str(audit.absolute_path_hits)
        status["audit.model_others_hits"] = str(audit.model_others_hits)
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="DEF-cmssm scaffold probe")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--backend", type=str, default=None)
    args = parser.parse_args()

    status = run_scaffold(root=args.root, backend=args.backend)
    for key, value in status.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
