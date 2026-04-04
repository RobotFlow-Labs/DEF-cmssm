# TASK-001 — Foundation Scaffold
- Priority: P0
- Depends on: none

## Scope
Create the module package baseline with Python 3.11, hatchling, backend resolution, and autopilot probe.

## Acceptance
1. `pyproject.toml` present with `requires-python >=3.11`.
2. `src/def_cmssm/device.py` resolves `ANIMA_BACKEND`.
3. `scripts/anima_autopilot.py` runs scaffold status probe.
