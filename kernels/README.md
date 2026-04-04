# Custom Kernels — DEF-cmssm
# CM-SSM: Cross-Modal State Space Modeling CUDA/MLX Kernels
# Architecture: Dual-stream encoder + Mamba (SS2D_rgbt) cross-modal fusion
# Following /anima-optimize-cuda-pipeline Phase 3

## Architecture-Specific Kernel Targets

NOTE: This repo depends on mamba-ssm which provides selective_scan_fn as a pre-compiled
CUDA kernel. Our custom kernels wrap mamba-ssm's core with fused pre/post-processing
for the cross-modal interleaving pattern.

### Kernel 1: Fused Cross-Modal SS2D (`cm_ss2d_fused.cu`)
**Bottleneck**: SS2D_rgbt runs ~25 ops per stage: 2× in_proj + 2× DWConv + interleave + selective_scan + de-interleave + 2× norm + 2× gate + 2× out_proj. Called 4 times (once per stage).
**Current**: PyTorch ops with 8+ intermediate B×D×2L tensors for interleave/de-interleave
**Target**: 2-3x speedup per SS2D_rgbt call

```
Input: rgb (B×H×W×C), thermal (B×H×W×C),
       in_proj1_weight (C×2D), in_proj2_weight (C×2D),
       conv2d1_weight (D×1×3×3), conv2d2_weight (D×1×3×3),
       x_proj_weight (4×(dt_rank+2*d_state)×D),
       dt_projs_weight (4×D×dt_rank), dt_projs_bias (4×D),
       A_logs (4D×d_state), Ds (4D),
       norm1_weight, norm2_weight,
       out_proj1_weight (D×C), out_proj2_weight (D×C)
Output: rgb_out (B×C×H×W), t_out (B×C×H×W)

Method: 3-phase fused kernel

  Phase 1: Projection + DWConv (custom CUDA)
    rgb_in = in_proj1(rgb_flat)  →  split → (rgb1, rgb2)
    t_in = in_proj2(t_flat)     →  split → (t1, t2)
    rgb1 = SiLU(DWConv3(reshape(rgb1)))  [B, D, H, W]
    t1 = SiLU(DWConv3(reshape(t1)))      [B, D, H, W]
    --- All 4 projections + 2 DWConvs in single fused kernel ---

  Phase 2: Cross-Modal Selective Scan (wraps mamba-ssm)
    hw_rgb = flatten(rgb1, H*W)     [B, D, L]
    wh_rgb = flatten(transpose(rgb1), H*W)
    hw_t = flatten(t1, H*W)
    wh_t = flatten(transpose(t1), H*W)

    INTERLEAVE (custom zero-copy):
      For direction d ∈ {hw, wh}:
        interleaved_d[b, d, 2*l] = rgb_d[b, d, l]
        interleaved_d[b, d, 2*l+1] = t_d[b, d, l]

    Forward + reverse: xs[b, 4, D, 2L]
    x_proj → dt, B_ssm, C_ssm parameters
    dt_projs → delta time

    selective_scan_fn(xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus=True)
      → out_y[B, 4, D, 2L]

    Sum 4 directions + DE-INTERLEAVE:
      y_total = y_hw + y_hw_inv + y_wh + y_wh_inv
      rgb_scan = y_total[:, 0::2, :]  (even indices)
      t_scan = y_total[:, 1::2, :]    (odd indices)

  Phase 3: Gated Output (custom CUDA)
    rgb_out = out_proj1(norm1(rgb_scan.view(B,H,W,D)) * SiLU(rgb2))
    t_out = out_proj2(norm2(t_scan.view(B,H,W,D)) * SiLU(t2))
    --- norm + gate + proj fused into single kernel ---

Key insight: Phase 2 calls mamba-ssm's selective_scan_fn internally.
Our kernel wraps it with fused Phases 1 and 3, eliminating intermediate allocations.
The interleave/de-interleave is custom zero-copy (biggest memory saving).
```

**Python wrapper**: `cm_ss2d_fused(rgb, thermal, weights_dict)` → (rgb_out, t_out)
**Save to**: `/mnt/forge-data/shared_infra/cuda_extensions/cm_ss2d_fused/`
**REUSABLE by**: Any cross-modal Mamba model (sigma-style, MDNet-style)

### Kernel 2: Fused CM_SSM Block (`cm_ssm_block.cu`)
**Bottleneck**: Each CM_SSM block: Conv3x3(cat) + SS2D_rgbt + residual + Conv1x1(cat) — 3 intermediate B×C×H×W allocations
**Target**: 1.5x speedup per block, 50% memory reduction

```
Input: rgb (B×C×H×W), thermal (B×C×H×W), block weights
Output: fused (B×C×H×W)

Method: Single mega-kernel
  1. left = BN(Conv3x3(cat(rgb, t))) + ReLU
     --- cat is implicit: kernel reads from 2 input pointers ---
  2. rgb_, t_ = cm_ss2d_fused(rgb.permute, t.permute)  [call Kernel 1]
  3. rgb_ = rgb_ + rgb  [residual, in-place]
  4. t_ = t_ + t        [residual, in-place]
  5. out = BN(Conv1x1(cat(left, rgb_, t_))) + ReLU
     --- cat is implicit: kernel reads from 3 input pointers ---

Savings: No materialized cat tensors (2C and 3C intermediates avoided)
4 stages × 1 block each = 4 calls total, saving 4 × 3 × B×C×H×W
```

**Python wrapper**: `cm_ssm_block(rgb, thermal, block_weights)` → fused_out
**Save to**: `/mnt/forge-data/shared_infra/cuda_extensions/cm_ssm_block/`

### Kernel 3: Zero-Copy Cross-Modal Interleave/De-interleave (`cm_interleave.cu`)
**Bottleneck**: Creating interleaved sequences requires torch.stack → view → cat → flip = 4 memory ops per direction, 2 directions, 4 stages = 32 memory operations
**Target**: 4x speedup for interleave, eliminate all intermediate allocations

```
Input (interleave):
  rgb_flat (B×D×L), t_flat (B×D×L)
Output:
  interleaved (B×D×2L) — zero-copy, single coalesced write

Kernel: interleave_forward
  For each thread (b, d, l):
    interleaved[b, d, 2*l] = rgb_flat[b, d, l]
    interleaved[b, d, 2*l+1] = t_flat[b, d, l]
  Coalesced write pattern: each warp writes 32 consecutive elements

Input (de-interleave):
  y (B×D×2L)
Output:
  rgb_out (B×D×L), t_out (B×D×L)

Kernel: deinterleave_forward
  For each thread (b, d, l):
    rgb_out[b, d, l] = y[b, d, 2*l]
    t_out[b, d, l] = y[b, d, 2*l+1]
  Coalesced read pattern from y

Also fused with flip for reverse direction:
  interleaved_rev[b, d, n] = interleaved[b, d, 2L-1-n]

Backward: simple transpose of forward — swap read/write patterns
```

**Python wrapper**: `cm_interleave(rgb_flat, t_flat)` → interleaved, `cm_deinterleave(y)` → (rgb_out, t_out)
**Save to**: `/mnt/forge-data/shared_infra/cuda_extensions/cm_interleave/`
**REUSABLE by**: ANY model that interleaves multi-modal sequences for joint processing

### Kernel 4: Batched Dual-Stream Encoder (`dual_stream_batch.py`)
**Not a CUDA kernel** — PyTorch-level optimization
**Target**: 1.3x encoder speedup

```
Method: Stack RGB and thermal into single larger batch
  rgb_batch = model_input[:B]    [B×3×H×W]
  t_batch = model_input[B:]      [B×3×H×W]
  stacked = cat(rgb_batch, t_batch, dim=0)  [2B×3×H×W]

  features = shared_backbone(stacked)  → 4-stage features for 2B batch
  f_rgb = [f[:B] for f in features]
  f_t = [f[B:] for f in features]

Requires: shared_weights=True in Encoder_RGBT_Efficientvit
Benefit: Single backbone call → better GPU utilization (especially at small B)
Risk: Different BN statistics for RGB vs thermal at train time
  → Use separate BN but shared conv weights
```

## MLX Metal Equivalents

### Critical: Mamba Selective Scan on MLX
The selective_scan_fn from mamba-ssm is CUDA-only. MLX port requires:

1. **`selective_scan_mlx.py`** — Pure MLX selective scan
   ```python
   # Sequential scan (simplest, O(N)):
   def selective_scan(x, dt, A, B, C, D):
       # x: (B, D, L), dt: (B, D, L)
       h = mx.zeros((B, D, d_state))
       outputs = []
       for l in range(L):
           dt_l = mx.softplus(dt[:, :, l])
           A_bar = mx.exp(A * dt_l.unsqueeze(-1))  # discretize
           B_bar = dt_l.unsqueeze(-1) * B[:, :, l].unsqueeze(1)
           h = A_bar * h + B_bar * x[:, :, l].unsqueeze(-1)
           y_l = (C[:, :, l].unsqueeze(1) * h).sum(-1) + D * x[:, :, l]
           outputs.append(y_l)
       return mx.stack(outputs, axis=-1)
   ```
   This is slow due to sequential nature. For MLX, consider:
   - Parallel associative scan (O(L log L)) with custom Metal kernel
   - Or accept slower MLX inference as dev-only

2. **`cm_ss2d_mlx.py`** — MLX cross-modal SS2D
   - Interleave: simple array indexing `mx.zeros((B,D,2*L)); out[:,::2]=rgb; out[:,1::2]=t`
   - DWConv: `mlx.nn.Conv2d(groups=D)`
   - in_proj/out_proj: `mlx.nn.Linear`

3. **`cm_ssm_block_mlx.py`** — MLX full block
   - Compose Conv3x3 + SS2D_rgbt_mlx + Conv1x1

4. **Weight conversion**: PyTorch state_dict → MLX npz

## Benchmark Targets

| Kernel | Baseline (ms) | Target (ms) | Speedup |
|--------|--------------|-------------|---------|
| SS2D_rgbt per stage (B=1, C=128, 60×80) | ~8.0 | ~3.0 | 2.7x |
| SS2D_rgbt per stage (B=1, C=256, 30×40) | ~5.0 | ~2.0 | 2.5x |
| CM_SSM block (B=1, C=128, 60×80) | ~12.0 | ~7.0 | 1.7x |
| Interleave+flip (B=1, D=256, L=4800) | ~1.5 | ~0.3 | 5x |
| De-interleave (B=1, D=256, L=4800) | ~0.8 | ~0.2 | 4x |
| **Full forward pass (EVit-B1, 480×640)** | **~65** | **~35** | **1.9x** |
| **Full forward pass (ConvNeXtV2-A, 480×640)** | **~55** | **~30** | **1.8x** |

## Memory Analysis

| Stage | Channels | Feature Size | SS2D d_inner | Sequence Length (2L) | SSM Memory |
|-------|----------|-------------|-------------|---------------------|-----------|
| 1 | 32 (B1) | 120×160 | 64 | 38,400 | ~10MB |
| 2 | 64 (B1) | 60×80 | 128 | 9,600 | ~5MB |
| 3 | 128 (B1) | 30×40 | 256 | 2,400 | ~2.5MB |
| 4 | 256 (B1) | 15×20 | 512 | 600 | ~1.3MB |

Stage 1 is the heaviest — 38,400 interleaved tokens × 4 directions × d_inner=64.
Fused interleave eliminates ~150MB of intermediate allocations across all stages.

## IP Notes

- **cm_ss2d_fused.cu** is the most novel — first fused cross-modal Mamba kernel. Wraps mamba-ssm's selective_scan with zero-copy interleave/de-interleave. Reusable by ANY cross-modal SSM architecture.
- **cm_interleave.cu** is the simplest but most widely reusable — any model that interleaves multi-modal sequences benefits. Patent-worthy as a general-purpose cross-modal sequence interleaving kernel.
- **mamba-ssm dependency**: Our kernels depend on mamba-ssm 1.0.1 for selective_scan_fn. If mamba-ssm API changes, our wrapper needs updating.
- All kernels stored at `/mnt/forge-data/shared_infra/cuda_extensions/`.

---
*Updated 2026-04-04 by ANIMA Research Agent*
