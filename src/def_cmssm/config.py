from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class ModulePaths:
    root: Path
    cmssm_repo: Path = field(init=False)
    papers_dir: Path = field(init=False)
    kernels_dir: Path = field(init=False)
    benchmarks_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        self.cmssm_repo = self.root / "repositories" / "CMSSM"
        self.papers_dir = self.root / "papers"
        self.kernels_dir = self.root / "kernels"
        self.benchmarks_dir = self.root / "benchmarks"


@dataclass(slots=True)
class ModuleConfig:
    """Top-level scaffolding config for DEF-cmssm."""

    module_name: str = "DEF-cmssm"
    python: str = "3.11"
    package_manager: str = "uv"
    build_backend: str = "hatchling"
    paper_id: str = "2506.17869"
    paper_repo_url: str = "https://github.com/xiaodonguo/CMSSM"

    def paths(self, root: Path) -> ModulePaths:
        return ModulePaths(root=root)
