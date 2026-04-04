# CUDA Kernel Scaffold — DEF-cmssm

This directory holds module-owned CUDA optimization work for CM-SSM.

## Planned Kernels
1. `cm_interleave.cu`
2. `cm_ss2d_fused.cu`
3. `cm_ssm_block.cu`
4. `dual_stream_batch.py` (PyTorch-level optimization wrapper)

## Rules
1. Keep reference parity checks for each replacement.
2. Log benchmark deltas under `benchmarks/`.
3. Do not bind hardcoded paths in kernel wrappers.
