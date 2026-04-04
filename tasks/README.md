# Tasks — DEF-cmssm
# CM-SSM: Cross-Modal State Space Modeling for RGB-T Semantic Segmentation
# Paper: IROS 2025 | ArXiv: 2506.17869 | Journal: IEEE TASE (with KD)
# Author: Xiaodong Guo et al. (same as DEF-tuni)
# Repo: https://github.com/xiaodonguo/CMSSM
# Total: 11 PRDs | 72 hours estimated
# Critical Path: PRD-001 → PRD-002 → PRD-003 → PRD-005 → PRD-006 → PRD-008

---

## PRD-001: Environment Setup (6h) ⬜
**Priority**: P0 — blocking everything
**Dependencies**: None

### Steps
```bash
# 1. Clone repo
cd /mnt/forge-data/shared_infra/repos/
git clone https://github.com/xiaodonguo/CMSSM.git
cd CMSSM

# 2. Create uv env
uv venv .venv --python 3.10
source .venv/bin/activate

# 3. Install PyTorch cu128
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 4. Install Mamba SSM (CRITICAL — requires CUDA compilation)
# mamba-ssm 1.0.1 contains the selective_scan_fn CUDA kernel
uv pip install mamba-ssm==1.0.1
# If compilation fails, try: pip install mamba-ssm --no-build-isolation

# 5. Install causal-conv1d (CRITICAL — fast 1D causal convolution for Mamba)
uv pip install causal-conv1d==1.0.0

# 6. Install mmcv + mmengine
uv pip install mmcv==2.2.0 mmengine

# 7. Install additional deps
uv pip install timm einops fvcore ptflops jinja2

# 8. Download EfficientViT pretrained backbone weights
mkdir -p backbone/efficientvit/
wget -O backbone/efficientvit/efficientvit_b1_r288.pt \
    https://huggingface.co/mit-han-lab/efficientvit-b1-r288/resolve/main/efficientvit_b1_r288.pt

# 9. Download ConvNeXtV2 pretrained backbone weights
mkdir -p backbone/convnextV2/
# ConvNeXtV2-atto from Facebook Research
wget -O backbone/convnextV2/convnextv2_atto.pt \
    https://dl.fbaipublicfiles.com/convnext/convnextv2/im1k/convnextv2_atto_1k_224_ema.pt

# 10. Download CM-SSM pre-trained weights (4 checkpoints)
mkdir -p pretrained/
wget -O pretrained/CART.pth https://github.com/xiaodonguo/CMSSM/releases/download/v1.0.1/CART.pth
wget -O pretrained/PST900.pth https://github.com/xiaodonguo/CMSSM/releases/download/v1.0.1/PST900.pth
wget -O pretrained/SUS.pth https://github.com/xiaodonguo/CMSSM/releases/download/v1.0.1/SUS.pth
wget -O pretrained/FMB.pth https://github.com/xiaodonguo/CMSSM/releases/download/v1.0.1/FMB.pth

# 11. Fix absolute paths in encoder files
# models/encoder/Efficientvit.py has hardcoded paths like:
#   "/home/ubuntu/code/backbone/efficientvit/efficientvit_b1_r288.pt"
# Update to: "./backbone/efficientvit/efficientvit_b1_r288.pt"
# Also fix ConvNeXtV2.py paths

# 12. Fix import paths
# model_EVit.py uses "proposed." prefix — update to "models."
# model_Conv.py uses "proposed." prefix — update to "models."
# fusion.py imports from "model_others.RGB_T.*" — these are comparison methods
#   May need to stub or comment out: CMX, MAINet, MDNet, sigma imports
```

### CRITICAL: mamba-ssm Installation
mamba-ssm requires CUDA to compile selective_scan_fn. If on a machine without CUDA:
- Use prebuilt wheels: `pip install mamba-ssm --find-links ...`
- Or install from source with `CUDA_HOME` set
- Version 1.0.1 is required (newer versions may have API changes)

### Acceptance Criteria
- [ ] `python -c "import torch; print(torch.cuda.is_available())"` → True
- [ ] `python -c "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print('OK')"` → OK
- [ ] `python -c "from causal_conv1d import causal_conv1d_fn; print('OK')"` → OK
- [ ] `python -c "from models.CM_SSM import Model; print('OK')"` → OK (after import fixes)
- [ ] `python -c "from backbone.MedMamba import SS2D; print('OK')"` → OK
- [ ] All 4 pre-trained weights downloaded to pretrained/
- [ ] EfficientViT and ConvNeXtV2 backbone weights downloaded
- [ ] Absolute paths fixed in all encoder files

---

## PRD-002: Dataset Download & Preparation (5h) ⬜
**Priority**: P0 — blocking evaluation
**Dependencies**: PRD-001

### Datasets (4 total)
```bash
DATASET_ROOT=/mnt/forge-data/datasets/rgbt/

# 1. FMB (Freiburg Multi-modal Benchmark) — 14 classes, urban driving
# Shared with DEF-tuni, DEF-rtfdnet — may already exist
mkdir -p $DATASET_ROOT/FMB
# Download from SegMiF repo: https://github.com/JinyuanLiu-CV/SegMiF

# 2. PST900 (Penn State Thermal) — 5 classes, off-road/indoor
# Shared with DEF-tuni, DEF-rtfdnet
mkdir -p $DATASET_ROOT/PST900
# Download from: https://github.com/ShreyasSkandanS/pst900_thermal_rgb

# 3. CART (Caltech Aerial RGB-T) — 12 classes, aerial
# Shared with DEF-tuni
mkdir -p $DATASET_ROOT/CART
# Download from: https://github.com/aerorobotics/caltech-aerial-rgbt-dataset

# 4. SUS (Wild Scene — author's own dataset)
# NEW — not shared with other modules
mkdir -p $DATASET_ROOT/SUS
# Download from: https://github.com/xiaodonguo/SUS_dataset
```

### Config Updates
```bash
# Update configs/CART.json → data_root
# Update configs/PST900.json → data_root
# Create configs/FMB.json (may not exist in repo)
# Create configs/SUS.json (may not exist in repo)
# Check train_FMB.py and train_SUS.py for inline config paths
```

### IMPORTANT: Input Format
- CM-SSM takes **3-channel thermal** input (unlike TUNI which converts to 1ch)
- Both RGB and thermal are standard 3-channel images
- Resolution: 480×640 (CART, PST900) or 512×640 (FMB per model_Conv.py __main__)

### Acceptance Criteria
- [ ] FMB dataset accessible, correct train/test split
- [ ] PST900 dataset accessible
- [ ] CART dataset accessible
- [ ] SUS dataset accessible (author's own dataset)
- [ ] All config files updated with correct paths
- [ ] Symlinks created for shared datasets (FMB, PST900, CART) with DEF-tuni/DEF-rtfdnet
- [ ] Sample images verified: RGB (H×W×3), Thermal (H×W×3)

---

## PRD-003: Evaluation Baseline — All 4 Datasets (6h) ⬜
**Priority**: P0 — establishes ground truth
**Dependencies**: PRD-001, PRD-002

### Steps
```bash
# Run evaluation on each dataset with pre-trained weights

# 1. CART evaluation (12 classes, EfficientViT-B1)
python evaluate_CART.py \
    --checkpoint pretrained/CART.pth \
    --gpu 0
# Expected: 75.1 mIoU

# 2. PST900 evaluation (5 classes, EfficientViT-B1)
python evaluate_pst900.py \
    --checkpoint pretrained/PST900.pth \
    --gpu 0
# Expected: 85.9 mIoU

# 3. SUS evaluation (author's dataset, ConvNeXtV2-A)
python evaluate_SUS.py \
    --checkpoint pretrained/SUS.pth \
    --gpu 0
# Expected: 82.5 mIoU

# 4. FMB evaluation (14 classes, ConvNeXtV2-A)
python evaluate_FMB.py \
    --checkpoint pretrained/FMB.pth \
    --gpu 0
# Expected: 60.7 mIoU

# 5. Measure model stats
python -c "
from models.CM_SSM import Model
from ptflops import get_model_complexity_info
model = Model(mode='b1', inputs='rgbt', fusion_mode='CM-SSM', n_class=12).eval().cuda()
flops, params = get_model_complexity_info(model, (3, 480, 640), as_strings=True)
print(f'FLOPs: {flops}, Params: {params}')
"
```

### Acceptance Criteria
- [ ] CART mIoU within 1% of 75.1 (paper value)
- [ ] PST900 mIoU within 1% of 85.9 (paper value)
- [ ] SUS mIoU within 1% of 82.5 (paper value)
- [ ] FMB mIoU within 1% of 60.7 (paper value)
- [ ] Per-class IoU tables saved for all 4 datasets
- [ ] GFLOPs and parameter counts recorded for both backbone configs
- [ ] Results saved to benchmarks/

---

## PRD-004: FPS Benchmarking (4h) ⬜
**Priority**: P1 — speed comparison with TUNI
**Dependencies**: PRD-001

### Steps
```bash
# 1. GPU FPS benchmark
python fps.py --input_size 480 640 --warmup 100 --iterations 1000

# 2. Both backbone variants
# EfficientViT-B1 (CART/PST900 config)
python -c "
import torch, time
from models.CM_SSM import Model
model = Model(mode='b1', inputs='rgbt', fusion_mode='CM-SSM', n_class=12).eval().cuda()
rgb = torch.randn(1, 3, 480, 640).cuda()
t = torch.randn(1, 3, 480, 640).cuda()
# warmup
for _ in range(100): model(rgb, t)
torch.cuda.synchronize()
start = time.time()
for _ in range(1000): model(rgb, t)
torch.cuda.synchronize()
fps = 1000 / (time.time() - start)
print(f'EfficientViT-B1 CM-SSM: {fps:.1f} FPS')
"

# ConvNeXtV2-atto (FMB/SUS config)
# Same benchmark with model_Conv.py

# 3. Compare fusion modes (the repo supports many!)
# CM-SSM vs M-SSM vs add vs max vs cat vs CMX vs sigma vs CDA vs TSFA
for mode in CM-SSM M-SSM add max cat; do
    python benchmark_fusion.py --fusion_mode $mode --backbone b1
done

# 4. Compare with TUNI on same hardware
# TUNI: expected 60+ FPS (lighter architecture)
# CM-SSM: expected 20-40 FPS (Mamba overhead)
```

### Acceptance Criteria
- [ ] FPS measured for EfficientViT-B1 + CM-SSM at 480×640
- [ ] FPS measured for ConvNeXtV2-atto + CM-SSM at 480×640
- [ ] Fusion mode comparison: CM-SSM vs alternatives (at least 5 modes)
- [ ] Comparison with DEF-tuni FPS on same hardware
- [ ] Latency (ms) with std deviation
- [ ] Results saved to benchmarks/

---

## PRD-005: CUDA Profiling — Mamba Bottleneck Analysis (5h) ⬜
**Priority**: P1 — identifies kernel optimization targets
**Dependencies**: PRD-001, PRD-003

### Profiling Strategy
```bash
# 1. Nsight Systems trace
nsys profile --trace=cuda,nvtx --output=cmssm_profile \
    python fps.py --input_size 480 640 --iterations 100

# 2. PyTorch profiler
python -c "
import torch
from torch.profiler import profile, record_function, ProfilerActivity
from models.CM_SSM import Model

model = Model(mode='b1', inputs='rgbt', fusion_mode='CM-SSM', n_class=12).eval().cuda()
rgb = torch.randn(1, 3, 480, 640).cuda()
t = torch.randn(1, 3, 480, 640).cuda()

with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
    with record_function('full_forward'):
        out = model(rgb, t)

print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=30))
prof.export_chrome_trace('cmssm_trace.json')
"
```

### Expected Bottlenecks
1. **selective_scan_fn** — The Mamba core. Called on 2*L length sequences (interleaved RGB+T). This is already a fused CUDA kernel from mamba-ssm, but the cross-modal interleaving adds overhead.
2. **Cross-modal interleave/de-interleave** — `torch.stack → view → cat → flip` pattern creates ~8 intermediate tensors per CM_SSM block, 4 blocks total.
3. **in_proj + conv2d** — Two separate Linear + DWConv paths for RGB and thermal. Could batch.
4. **Dual encoder streams** — Two full EfficientViT/ConvNeXtV2 forwards. Independent but wasteful GPU utilization.
5. **Conv3x3 + Conv1x1 wrapper ops** — Small ops with kernel launch overhead.

### Acceptance Criteria
- [ ] Nsight Systems trace analyzed
- [ ] Time breakdown: encoder_rgb vs encoder_t vs CM_SSM_fusion × 4 vs decoder
- [ ] Per-CM_SSM breakdown: conv1 vs SS2D_rgbt vs conv2
- [ ] Within SS2D_rgbt: in_proj vs conv2d vs interleave vs selective_scan vs de-interleave vs norm+gate vs out_proj
- [ ] Memory bandwidth utilization measured
- [ ] Intermediate tensor allocation quantified
- [ ] Profile saved to benchmarks/profiling/

---

## PRD-006: Custom CUDA Kernels — 4 Targets (16h) ⬜
**Priority**: P1 — core IP generation
**Dependencies**: PRD-005

### Kernel 1: Fused Cross-Modal Interleave + Selective Scan (8h)
```
File: kernels/cuda/cm_ss2d_fused.cu
Save to: /mnt/forge-data/shared_infra/cuda_extensions/cm_ss2d_fused/

THE BIG KERNEL — fuses the entire SS2D_rgbt forward:

Input: rgb_feat (B×H×W×C), thermal_feat (B×H×W×C), all SSM weights
Output: rgb_out (B×C×H×W), t_out (B×C×H×W)

Method: Fuse ~25 ops into 3 phases:
  Phase 1: Projection + Local Conv (fuseable)
    1. in_proj1: rgb → (rgb1, rgb2) via Linear split
    2. in_proj2: t → (t1, t2) via Linear split
    3. conv2d1(rgb1) → SiLU [depthwise conv + activation]
    4. conv2d2(t1) → SiLU

  Phase 2: Cross-Modal Selective Scan (call mamba_ssm kernel internally)
    5. Flatten + create hw/wh scans for RGB and thermal
    6. INTERLEAVE: stack(rgb_hwwh, t_hwwh) → (B, 2, d_inner, 2*L)
       - Custom CUDA: zero-copy interleave without materializing intermediates
    7. Forward + reverse scans → xs (B, 4, d_inner, 2*L)
    8. x_proj: einsum for dt, B, C parameters
    9. dt_projs: delta time projection
    10. selective_scan_fn on (B, 4*d_inner, 2*L) sequences
    11. Sum 4 directions + de-interleave (custom: even→RGB, odd→thermal)

  Phase 3: Gated Output (fuseable)
    12. norm1(rgb_scan) * SiLU(rgb2) → out_proj1 → rgb_out
    13. norm2(t_scan) * SiLU(t2) → out_proj2 → t_out

Target: 2-3x speedup over PyTorch implementation
Key savings: Eliminate 8+ intermediate tensor allocations for interleave/de-interleave

Note: selective_scan_fn is already CUDA-optimized by mamba-ssm.
Our kernel wraps it with fused pre/post-processing.
```

### Kernel 2: Fused CM_SSM Block (4h)
```
File: kernels/cuda/cm_ssm_block.cu
Save to: /mnt/forge-data/shared_infra/cuda_extensions/cm_ssm_block/

Input: rgb (B×C×H×W), t (B×C×H×W), block weights
Output: fused (B×C×H×W)

Method: Fuse entire CM_SSM forward:
  1. left = Conv3x3(cat(rgb, t)) + BN + ReLU [left branch]
  2. rgb_, t_ = SS2D_rgbt(rgb, t) [call Kernel 1]
  3. rgb_ = rgb_ + rgb [residual]
  4. t_ = t_ + t [residual]
  5. out = Conv1x1(cat(left, rgb_, t_)) + BN + ReLU [merge]

Eliminate: 3 × B×C×H×W intermediate allocations (left, rgb_, t_)
Target: 1.5x speedup per CM_SSM block
```

### Kernel 3: Optimized Cross-Modal Interleave (2h)
```
File: kernels/cuda/cm_interleave.cu
Save to: /mnt/forge-data/shared_infra/cuda_extensions/cm_interleave/

Input: rgb_flat (B×D×L), t_flat (B×D×L)
Output: interleaved (B×D×2L) — alternating RGB, thermal tokens

Method: Single kernel, zero intermediate allocation
  For each (b, d, l):
    interleaved[b, d, 2*l] = rgb_flat[b, d, l]
    interleaved[b, d, 2*l+1] = t_flat[b, d, l]

Reverse (de-interleave):
  rgb_out[b, d, l] = y[b, d, 2*l]
  t_out[b, d, l] = y[b, d, 2*l+1]

Simple but called 4× per forward (once per CM_SSM stage).
Currently: torch.stack → view → cat → flip = 4 memory operations
Target: single coalesced memory operation → 4x speedup for interleave step

REUSABLE by: Any cross-modal Mamba model that interleaves modalities
```

### Kernel 4: Batched Dual-Stream Encoder (2h)
```
File: kernels/cuda/dual_stream_batch.cu
Save to: /mnt/forge-data/shared_infra/cuda_extensions/dual_stream_batch/

Input: rgb (B×3×H×W), t (B×3×H×W)
Output: f_rgb[4], f_t[4] (4-stage features each)

Method: Batch RGB and thermal into single larger batch
  stacked = cat(rgb, t, dim=0) → (2B×3×H×W)
  features = backbone(stacked)  → 4-stage features for 2B batch
  f_rgb = features[:B], f_t = features[B:]

Not a CUDA kernel — more of a PyTorch optimization.
Avoids: 2 separate encoder forwards with underutilized GPU.
Requires: shared weights between RGB and thermal encoders (share_weights=True)
Target: 1.3x encoder speedup when GPU has spare SM capacity
```

### Acceptance Criteria
- [ ] Kernel 1 (cm_ss2d_fused) compiles and passes unit tests
- [ ] Kernel 2 (cm_ssm_block) achieves 1.5x+ speedup
- [ ] Kernel 3 (cm_interleave) achieves 4x+ for interleave step
- [ ] Kernel 4 (dual_stream_batch) benchmarked
- [ ] All kernels have Python wrappers via torch.utils.cpp_extension
- [ ] Gradients verified against PyTorch reference (atol=1e-4)
- [ ] mIoU identical before and after kernel replacement
- [ ] Kernels stored in shared_infra for cross-module reuse

---

## PRD-007: MLX Port (8h) ⬜
**Priority**: P1 — dual-compute mandatory
**Dependencies**: PRD-003

### Key Challenge: Mamba on MLX
```python
# mamba-ssm is CUDA-only (selective_scan_fn is a CUDA kernel)
# MLX port requires reimplementing the selective scan in pure MLX

# Option A: Pure MLX selective scan
# Implement S6 (selective state space) using mlx.core operations:
#   - discretize A, B using dt
#   - sequential scan: h[t] = A_bar * h[t-1] + B_bar * x[t]
#   - y[t] = C[t] * h[t] + D * x[t]
# This is inherently sequential but MLX can vectorize over batch/channel dims

# Option B: Use mlx-mamba if available
# Check if community has ported mamba-ssm to MLX

# Option C: Parallel scan algorithm
# Implement parallel associative scan for O(L log L) complexity
# mlx supports custom Metal kernels for this
```

### Steps
```bash
# 1. Port backbone (EfficientViT or ConvNeXtV2)
# Both are standard ConvNet architectures — straightforward port to mlx.nn

# 2. Port SS2D_rgbt to MLX
# - in_proj: mlx.nn.Linear
# - conv2d: mlx.nn.Conv2d(groups=d_inner) — depthwise
# - selective_scan: Custom MLX implementation (KEY CHALLENGE)
# - interleave/de-interleave: mlx array indexing

# 3. Port CM_SSM block
# - Conv3x3 + BN + ReLU: standard mlx.nn

# 4. Port decoder (Decoder_MLP)
# - Same as TUNI port — already done for DEF-tuni

# 5. Weight conversion
python convert_weights.py --src pretrained/CART.pth --dst pretrained/CART_mlx.npz
```

### Acceptance Criteria
- [ ] Full model runs on MLX (Mac Studio M-series)
- [ ] Selective scan implemented in MLX (pure or Metal kernel)
- [ ] CART mIoU matches CUDA within 0.5%
- [ ] Weight conversion script works
- [ ] device.py abstraction layer functional
- [ ] `ANIMA_BACKEND=mlx python eval.py` works end-to-end
- [ ] MLX inference FPS measured

---

## PRD-008: Training from Scratch (6h) ⬜
**Priority**: P1 — validate training pipeline
**Dependencies**: PRD-001, PRD-002

### Steps
```bash
# Train on CART dataset (smallest, fastest convergence)
python train_CART.py \
    --backbone b1 \
    --fusion_mode CM-SSM \
    --n_class 12 \
    --epochs 300 \
    --batch_size 8 \
    --lr 6e-5 \
    --gpu 0

# Train on PST900
python train_PST900.py \
    --backbone b1 \
    --fusion_mode CM-SSM \
    --n_class 5 \
    --epochs 300

# Train with Knowledge Distillation (journal extension)
python train_CART_KD.py \
    --teacher_ckpt pretrained/CART_teacher.pth \
    --student_backbone b0 \
    --epochs 300
```

### Acceptance Criteria
- [ ] CART training converges to within 2% of paper mIoU (75.1)
- [ ] PST900 training converges to within 2% of paper mIoU (85.9)
- [ ] Training time per epoch recorded
- [ ] GPU memory usage recorded
- [ ] Loss curves saved (CE + Dice)
- [ ] KD training pipeline functional (if teacher weights available)

---

## PRD-009: Fusion Mode Ablation & Comparison (4h) ⬜
**Priority**: P2 — understanding architecture
**Dependencies**: PRD-003

### The repo supports 8+ fusion modes — compare all:
```bash
# All fusion modes available in Fusion_Module:
# 1. CM-SSM (proposed — Mamba cross-modal)
# 2. M-SSM (Mamba per-modal, no cross-modal)
# 3. add (element-wise addition)
# 4. max (element-wise max)
# 5. cat (concatenation + conv)
# 6. CMX (cross-modal attention from CMX paper)
# 7. sigma (cross-modal Mamba from Sigma paper)
# 8. CDA (channel-domain attention)
# 9. TSFA (from MAINet)
# 10. MDFusion (from MDNet — uses SS_Conv_SSM)

# Compare all on CART dataset (fastest):
for mode in CM-SSM M-SSM add max cat CMX sigma CDA TSFA; do
    python train_CART.py --fusion_mode $mode --epochs 300 --save_prefix ${mode}
    python evaluate_CART.py --checkpoint results/${mode}_best.pth
done
```

### Acceptance Criteria
- [ ] All 8+ fusion modes evaluated on CART
- [ ] mIoU comparison table
- [ ] FPS comparison (some modes are much faster)
- [ ] CM-SSM vs M-SSM ablation (cross-modal vs per-modal Mamba)
- [ ] Accuracy-speed Pareto plot

---

## PRD-010: RGB-T Family Integration & Comparison (3h) ⬜
**Priority**: P2 — cross-module analysis
**Dependencies**: PRD-003

### Cross-Module Comparison
```bash
# Compare CM-SSM with other RGB-T modules on shared datasets:
# 1. DEF-tuni — TUNI (unified encoder, 27 FPS real-time)
# 2. DEF-rtfdnet — RTFDNet (dual-stream SegFormer, robustness)
# 3. DEF-hypsam — HyPSAM (RGB-T SOD + SAM)

# CMSSM unique advantage: linear complexity via Mamba (O(N) vs O(N²))
# TUNI unique advantage: real-time speed (27 FPS Jetson)
# RTFDNet unique advantage: modality degradation robustness
```

### Same Author (TUNI + CMSSM) Analysis
- Both by Xiaodong Guo — designed as complementary approaches
- TUNI = real-time focus (unified encoder, attention-based)
- CMSSM = accuracy focus (dual-stream encoder, Mamba-based)
- Shared datasets: FMB, PST900, CART
- Can share encoder weights? (both support EfficientViT)

### Acceptance Criteria
- [ ] Comparison table: CMSSM vs TUNI vs RTFDNet on FMB + PST900
- [ ] Speed comparison on same hardware
- [ ] Accuracy-speed tradeoff plot (Pareto frontier)
- [ ] CMSSM + TUNI complementary analysis documented

---

## PRD-011: Edge Deployment — Jetson (5h) ⬜
**Priority**: P2 — edge demo (lower priority than TUNI for edge)
**Dependencies**: PRD-003, PRD-006

### Challenge: mamba-ssm on Jetson
```bash
# mamba-ssm requires CUDA compilation — verify it compiles on Jetson's CUDA version
# Jetson Orin NX runs CUDA 11.4-12.x via JetPack

# ONNX export is HARD for Mamba (selective_scan is not standard ONNX op)
# Options:
# 1. Custom ONNX op registration for selective_scan
# 2. TorchScript export with custom op
# 3. PyTorch → TensorRT with custom plugin for selective_scan
# 4. Run in PyTorch directly on Jetson (simplest, slower)

# For demo: option 4 (PyTorch on Jetson) is fastest path
# For production: option 3 (TRT with custom plugin)
```

### Acceptance Criteria
- [ ] mamba-ssm compiles on Jetson Orin NX
- [ ] Inference runs in PyTorch on Jetson
- [ ] FPS measured (expect 5-15 FPS, slower than TUNI)
- [ ] Memory footprint < 6GB

---

## Build Plan Summary

| PRD | Task | Hours | Status | Depends On |
|-----|------|-------|--------|-----------|
| 001 | Environment Setup | 6h | ⬜ | — |
| 002 | Dataset Download | 5h | ⬜ | 001 |
| 003 | Evaluation Baseline | 6h | ⬜ | 001, 002 |
| 004 | FPS Benchmarking | 4h | ⬜ | 001 |
| 005 | CUDA Profiling | 5h | ⬜ | 001, 003 |
| 006 | Custom CUDA Kernels | 16h | ⬜ | 005 |
| 007 | MLX Port | 8h | ⬜ | 003 |
| 008 | Training from Scratch | 6h | ⬜ | 001, 002 |
| 009 | Fusion Mode Ablation | 4h | ⬜ | 003 |
| 010 | RGB-T Family Integration | 3h | ⬜ | 003 |
| 011 | Edge Deployment Jetson | 5h | ⬜ | 003, 006 |
| **TOTAL** | | **72h** | | |

## Critical Path
```
PRD-001 (env + mamba-ssm) → PRD-002 (data) → PRD-003 (baseline) → PRD-005 (profile) → PRD-006 (kernels)
                                              ↘ PRD-004 (FPS) — parallel with 005
                                              ↘ PRD-007 (MLX) — parallel with 005/006
                                              ↘ PRD-008 (training) — parallel with 005
                                              ↘ PRD-009 (ablation) — parallel with 005
                                              ↘ PRD-010 (integration) — after 003
```

---
*Updated 2026-04-04 by ANIMA Research Agent*
