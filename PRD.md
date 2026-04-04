# PRD — DEF-cmssm (Wave 8)

## 1. Project Identity
- Module: `DEF-cmssm`
- Focus: RGB-T semantic segmentation with cross-modal Mamba fusion
- Paper: `arXiv:2506.17869` (IROS 2025 acceptance stated by authors)
- Reference implementation: `repositories/CMSSM`
- Execution backends: `CUDA` and `MLX`
- Python baseline: `3.11`

## 2. Context and Source Audit
This PRD is grounded in:
- Local paper PDF: `papers/2506.17869.pdf`
- Local upstream code: `repositories/CMSSM` (source and configs audited)
- Upstream metadata: GitHub repository and release artifacts (`v1.0.1` checkpoints)

Observed code health in upstream:
1. Namespace drift (`proposed.*` imports still present in multiple model files).
2. Hardcoded absolute paths (`/home/ubuntu/...`, `/root/autodl-tmp/...`) in configs, encoders, train/eval scripts.
3. Optional comparison dependencies required unguarded in fusion stack (`model_others.*`).
4. Inconsistent model output assumptions (`predict`, `predict[0]`, `predict[-1]`) across scripts.
5. Dataset/config coverage mismatch (`FMB` and `SUS` scripts exist, dataset registry is only CART/PST900 in toolbox).
6. `ConvNeXtV2` import typo in encoder (`covnextV2` vs `convnextV2`).

## 3. Problem Statement
The module has strong research value but cannot be executed reproducibly in ANIMA infrastructure without a portability and stability scaffold. We need a robust foundation that:
- normalizes environment and paths,
- decouples optional baselines from core CM-SSM,
- supports dual compute (`ANIMA_BACKEND=cuda|mlx`),
- creates clean extension points for kernel optimization and benchmarking.

## 4. Product Goals
1. Make CM-SSM runnable from this module with minimal manual patching.
2. Establish dual-backend scaffolding so CUDA and MLX have explicit execution routes.
3. Define and start kernel-IP pipeline in `kernels/` and measurement pipeline in `benchmarks/`.
4. Provide implementation-ready tasks with acceptance criteria.

## 5. Non-Goals (Current Slice)
1. Full training reproduction on all datasets.
2. Final MLX selective-scan parity implementation.
3. Production TensorRT export.

## 6. Functional Requirements
1. The module shall use Python `3.11` and `uv` with `hatchling` packaging.
2. The module shall expose backend selection through `ANIMA_BACKEND`.
3. The module shall include a deterministic autopilot entrypoint for scaffold checks.
4. The module shall include a legacy adapter layer for systematic patching of upstream import/path issues.
5. The module shall include kernel scaffold files for the four IP targets:
- `cm_ss2d_fused`
- `cm_ssm_block`
- `cm_interleave`
- `dual_stream_batch`
6. The module shall include benchmark templates for:
- baseline accuracy/FPS
- profiling breakdown
- kernel delta reporting

## 7. Architecture (Scaffold Baseline)
- Package root: `src/def_cmssm`
- Core components:
- `device.py`: backend resolution
- `config.py`: module metadata and path model
- `paths.py`: shared infra path constants
- `legacy_adapter.py`: text patch primitives for upstream cleanup
- `pipelines/scaffold.py`: autopilot scaffold probe
- Entry script:
- `scripts/anima_autopilot.py`

## 8. Data and Model Contract (Initial)
- Upstream repo stays in-place under `repositories/CMSSM`.
- Paper artifact remains under `papers/2506.17869.pdf`.
- No model/dataset download in this scaffold slice.
- All dataset/model roots should transition to config-driven paths; no absolute user paths allowed.

## 9. Kernel IP Plan
1. Cross-modal interleave/de-interleave kernel (`cm_interleave`) as first low-risk CUDA unit.
2. SS2D wrapper fusion kernel (`cm_ss2d_fused`) wrapping `selective_scan_fn` pre/post.
3. CM_SSM block fusion (`cm_ssm_block`) reducing concat allocation overhead.
4. Batched dual-stream encoder optimization (`dual_stream_batch`) as a PyTorch-level optimization.

## 10. Benchmark Plan
1. Baseline report before replacement.
2. Per-kernel microbenchmark deltas.
3. End-to-end latency/FPS and memory delta after each optimization step.
4. Maintain identical semantic outputs under tolerance against reference path.

## 11. Milestones
1. M0: Scaffold + PRD + sliced tasks (this delivery).
2. M1: Upstream portability patch set (imports, paths, config unification).
3. M2: Baseline eval/fps harness stabilized on CUDA.
4. M3: Kernel integration loop (interleave -> SS2D -> block).
5. M4: MLX parity route bootstrapped with selective-scan fallback.

## 12. Risks and Mitigations
1. `mamba-ssm` build incompatibility across machines.
- Mitigation: pin version + explicit environment docs + wheel fallback policy.
2. Missing external comparison modules causing import crashes.
- Mitigation: guarded optional imports and feature flags.
3. MLX selective-scan performance/accuracy gap.
- Mitigation: phased fallback implementation with explicit regression thresholds.

## 13. Acceptance Criteria (Scaffold Slice)
1. `pyproject.toml` uses Python `>=3.11` and hatchling build.
2. Scaffold package exists under `src/def_cmssm` with backend/config/path primitives.
3. `scripts/anima_autopilot.py` exists and uses local module package.
4. PRD task slicing files exist under `tasks/slices`.
5. Kernel and benchmark template files exist and align with CM-SSM optimization targets.

## 14. Success Metrics
1. Zero hardcoded absolute paths in module-owned code.
2. Reproducible scaffold probe output across machines.
3. New contributors can identify what to implement next from `tasks/slices` without context loss.
