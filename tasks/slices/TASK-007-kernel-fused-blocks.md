# TASK-007 — Kernel Scaffold: `cm_ss2d_fused` and `cm_ssm_block`
- Priority: P1
- Depends on: TASK-006

## Scope
Build fused block scaffold around Mamba selective-scan call path.

## Acceptance
1. Wrapper interfaces defined and integration points documented.
2. Correctness comparison hooks exist vs reference path.
3. Throughput and memory deltas exported.
