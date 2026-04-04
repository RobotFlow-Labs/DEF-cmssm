# DEF-cmssm Task Roadmap

## Active Phase
- Phase: `M0 scaffold`
- Status: `in progress`
- Python baseline: `3.11`

## Task Slices
1. `TASK-001` Foundation packaging and backend abstraction.
2. `TASK-002` Upstream portability patching (imports/paths).
3. `TASK-003` Config unification and dataset registration expansion.
4. `TASK-004` Baseline eval harness hardening (CART/PST900/FMB/SUS).
5. `TASK-005` Baseline benchmark harness (latency/FPS/memory).
6. `TASK-006` CUDA kernel scaffold integration (`cm_interleave`).
7. `TASK-007` CUDA kernel scaffold integration (`cm_ss2d_fused`, `cm_ssm_block`).
8. `TASK-008` MLX fallback route and interface parity.
9. `TASK-009` Kernel benchmark + correctness reports.
10. `TASK-010` Release packaging and CI gates.

## Dependency Order
`001 -> 002 -> 003 -> 004 -> 005 -> 006 -> 007 -> 009 -> 010`

`004 -> 008 -> 009`

## Definition of Done (M0)
1. PRD exists and reflects audited code reality.
2. Task slices are actionable with acceptance criteria.
3. Scaffold package and autopilot script are present.
