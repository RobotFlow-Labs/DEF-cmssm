#!/usr/bin/env python3
"""Minimal repeatable autopilot entrypoint for DEF-cmssm scaffolding phase."""

from __future__ import annotations

from pathlib import Path

from def_cmssm.pipelines.scaffold import run_scaffold


def main() -> None:
    status = run_scaffold(root=Path.cwd())
    print("[anima-autopilot] scaffold status")
    for key, value in status.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
