# TASK-005 — Benchmark Harness Baseline
- Priority: P1
- Depends on: TASK-004

## Scope
Replace ad-hoc `fps.py` dependencies with module-owned benchmark harness.

## Acceptance
1. No dependency on `model_others` for baseline FPS benchmark.
2. Warmup/measure loops are configurable.
3. Metrics export to `benchmarks/` templates.
