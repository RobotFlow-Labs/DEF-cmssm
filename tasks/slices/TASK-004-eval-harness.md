# TASK-004 — Evaluation Harness Stabilization
- Priority: P1
- Depends on: TASK-003

## Scope
Stabilize eval entrypoints for 4 datasets and normalize output handling.

## Acceptance
1. Model output handling is consistent (`Tensor` contract).
2. Checkpoint path selection is argument/config driven.
3. Per-dataset metrics output is standardized.
