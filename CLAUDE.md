# DEF-cmssm — CM-SSM: Cross-Modal State Space Modeling for RGB-T Semantic Segmentation
# Wave 8 Defense Module
# Paper: "Cross-modal State Space Modeling for Real-time RGB-Thermal Wild Scene Semantic Segmentation"
# Extended: "Cross-modal State Space Modeling and Terrain-specific Knowledge Distillation for RGB-Thermal Semantic Segmentation"
# Authors: Xiaodong Guo et al. (SAME AUTHOR AS DEF-tuni)
# Published: IROS 2025 | ArXiv: 2506.17869
# Journal extension: submitted to IEEE TASE (with terrain-specific KD)
# Repo: https://github.com/xiaodonguo/CMSSM
# Domain: RGB-T Semantic Segmentation via Mamba State Space Models
# Product Stack: ORACLE (through-wall safety) / ATLAS (fleet AV night ops) / NEMESIS (terrain nav)
# Related: DEF-tuni (same author, complementary), DEF-rtfdnet (shared datasets), DEF-hypsam, DEF-rgbtcc

## Status: ⬜ Not Started

## Paper Summary
CM-SSM is an RGB-T semantic segmentation framework that uses **Mamba state space models** for cross-modal feature fusion. Unlike transformer-based fusion that uses quadratic attention, CM-SSM achieves **linear complexity** by interleaving RGB and thermal tokens in a 4-directional selective scan (Mamba S6). The key innovation is `SS2D_rgbt` — a cross-modal variant of the SS2D (Selective Scan 2D) block that jointly processes RGB and thermal sequences through shared state space dynamics. Uses dual-stream backbones (EfficientViT-B1 or ConvNeXtV2-A) with CM-SSM fusion at all 4 stages. Accepted at IROS 2025 with journal extension at IEEE TASE adding terrain-specific knowledge distillation.

## Architecture
- **Dual-stream Encoder**: Two independent backbone networks
  - Option A: `Encoder_RGBT_Efficientvit` — dual EfficientViT (b0/b1/b2/b3/l1/l2)
  - Option B: `Encoder_RGBT_ConvNeXt` — dual ConvNeXtV2 (atto/femto/pico/nano/tiny/base)
  - Both output 4-stage features: `f_rgb[4]`, `f_t[4]`

- **CM-SSM Fusion Module** (KEY INNOVATION — `fuison_strategy/fusion.py`):
  - 4 × `CM_SSM` modules (one per encoder stage)
  - Each `CM_SSM(in_c)`:
    1. **Left branch**: `conv1 = Conv3x3(2C→C) + BN + ReLU` on `cat(rgb, t)`
    2. **SS2D_rgbt branch** (THE CORE):
       - Interleaves RGB and thermal tokens for joint state space modeling
       - Produces refined `rgb_` and `t_` with cross-modal context
       - Residual: `rgb_ = rgb_ + rgb`, `t_ = t_ + t`
    3. **Merge**: `conv2 = Conv1x1(3C→C) + BN + ReLU` on `cat(left, rgb_, t_)`

- **SS2D_rgbt** (Cross-Modal Selective Scan — `fusion.py`, the 23KB core):
  - **Parameters**:
    - `d_model` = channel dim per stage
    - `d_state` = 16 (SSM hidden state dimension)
    - `d_conv` = 3 (local convolution kernel)
    - `expand` = 2 (d_inner = 2 * d_model)
    - `dt_rank` = ceil(d_model / 16) (timestep projection rank)
    - K = 4 scanning directions
  - **Forward pass**:
    1. `in_proj1(rgb)` → split into `rgb1` (gate) + `rgb2` (residual) — each d_inner
    2. `in_proj2(t)` → split into `t1` (gate) + `t2` (residual) — each d_inner
    3. `conv2d1(rgb1)` → SiLU activation — local conv on RGB
    4. `conv2d2(t1)` → SiLU activation — local conv on thermal
    5. **forward_corev0** (CROSS-MODAL SELECTIVE SCANNING):
       a. Flatten: rgb (B, d_inner, H*W), thermal (B, d_inner, H*W)
       b. Create hw and wh scans for each modality → (B, 2, d_inner, L)
       c. **INTERLEAVE**: stack rgb_hwwh and t_hwwh along last dim → (B, 2, d_inner, 2*L)
          - This is the key: RGB and thermal tokens alternate in the sequence
       d. Create forward + reverse scans → xs (B, 4, d_inner, 2*L) — 4 directions
       e. Project: `x_proj_weight` → dt, B, C state space parameters
       f. `selective_scan_fn` (Mamba core) on interleaved sequence → out_y (B, 4, d_inner, 2*L)
       g. Sum 4 directional scans: y = y1 + y2 + y3 + y4
       h. **DE-INTERLEAVE**: `rgb1 = y[:, 0::2, :]` (even tokens = RGB), `t1 = y[:, 1::2, :]` (odd tokens = thermal)
    6. rgb_out = `out_proj1(norm1(rgb1) * SiLU(rgb2))` — gated output
    7. t_out = `out_proj2(norm2(t1) * SiLU(t2))` — gated output
  - **State space parameters** (S4D initialization):
    - A_logs: S4D real initialization, log-space, (K*d_inner, d_state)
    - Ds: skip connection parameter, ones init, (K*d_inner,)
    - dt_projs: timestep projection with softplus, (K, d_inner, dt_rank)

- **Decoder**: `Decoder_MLP` — same lightweight MLP head as TUNI
  - Input channels match backbone: e.g., [32, 64, 128, 256] for EfficientViT-B1
  - Embed dim: 256 (B1) or 128 (ConvNeXtV2-atto)

- **Backbone configurations** (from model files):
  - EfficientViT-B0: channels [16, 32, 64, 128], emb 128
  - EfficientViT-B1: channels [32, 64, 128, 256], emb 256 (**default for CART/PST900**)
  - EfficientViT-B2: channels [48, 96, 192, 384], emb 256
  - ConvNeXtV2-atto: channels [40, 80, 160, 320], emb 128 (**default for FMB/SUS**)
  - ConvNeXtV2-femto: channels [48, 96, 192, 384], emb 256
  - ConvNeXtV2-pico: channels [64, 128, 256, 512], emb 768

- **Knowledge Distillation** (journal extension — `train_CART_KD.py`):
  - Terrain-specific KD: larger teacher → smaller student
  - KD loss added to standard CE + Dice loss

- **Thermal input**: 3-channel (same as RGB), NOT single-channel like TUNI

## Datasets Used
- **FMB** (Freiburg Multi-modal Benchmark) — 14 classes, urban driving (ConvNeXtV2-A)
- **PST900** (Penn State Thermal) — 5 classes, off-road/indoor (EfficientViT-B1)
- **CART** (Caltech Aerial RGB-T) — 12 classes, aerial (EfficientViT-B1)
- **SUS** (author's own dataset) — wild scene segmentation (ConvNeXtV2-A)
- Input resolution: 480×640 (H×W)
- Shared with DEF-tuni (FMB, PST900, CART), DEF-rtfdnet (FMB, PST900)

## Key Results (from README — to reproduce)
| Backbone | Dataset | mIoU (%) | Weights |
|----------|---------|----------|---------|
| EfficientViT-B1 | CART | 75.1 | releases/v1.0.1/CART.pth |
| EfficientViT-B1 | PST900 | 85.9 | releases/v1.0.1/PST900.pth |
| ConvNeXtV2-A | SUS | 82.5 | releases/v1.0.1/SUS.pth |
| ConvNeXtV2-A | FMB | 60.7 | releases/v1.0.1/FMB.pth |

## Dependencies
- Python 3.9+ (we use 3.10)
- PyTorch 2.0+ + CUDA
- **mamba-ssm 1.0.1** (CRITICAL — Mamba selective scan CUDA kernels)
- **causal-conv1d 1.0.0** (CRITICAL — fast 1D causal convolution)
- mmcv 2.2.0
- timm, einops, fvcore, ptflops
- EfficientViT pretrained weights (from backbone/efficientvit/)
- ConvNeXtV2 pretrained weights (from backbone/convnextV2/)

## Repo Structure
```
CMSSM/
├── README.md
├── models/
│   ├── CM_SSM.py                    — Main model (EfficientViT + CM-SSM fusion)
│   ├── model_EVit.py                — EfficientViT variant (older import paths)
│   ├── model_Conv.py                — ConvNeXtV2 variant
│   ├── model_mit.py                 — SegFormer MiT variant
│   ├── encoder/
│   │   ├── Efficientvit.py          — Dual EfficientViT encoder
│   │   ├── ConvNeXtV2.py            — Dual ConvNeXtV2 encoder
│   │   ├── Segformer.py             — SegFormer encoder
│   │   ├── Swin.py                  — Swin Transformer encoder
│   │   └── DFormer.py               — DFormer encoder
│   ├── decoder/
│   │   ├── MLP.py                   — Decoder_MLP (default, lightweight)
│   │   ├── MLP_plus.py              — Enhanced MLP decoder
│   │   ├── DeepLabV3.py             — DeepLabV3+ decoder
│   │   ├── FSN.py                   — FSN decoder
│   │   └── Hamburger.py             — Hamburger decoder
│   └── fuison_strategy/
│       └── fusion.py                — ALL FUSION MODES (23KB, THE CORE)
│                                      Contains: CM_SSM, M_SSM, SS2D_rgbt,
│                                      Demo1, MDFusion, + comparison methods
│                                      (CMX, sigma, CDA, TSFA, cat, add, max)
├── backbone/
│   ├── MedMamba.py                  — SS2D (vanilla Mamba) backbone
│   ├── efficientvit/                — EfficientViT backbone code
│   ├── convnextV2/                  — ConvNeXtV2 backbone code
│   └── SegFormer/                   — SegFormer backbone code
├── Loss/
│   ├── dice.py                      — Dice loss
│   └── functional.py               — Loss utilities
├── configs/
│   ├── base_cfg.py                  — Base training config
│   ├── CART.json                    — CART dataset config
│   └── PST900.json                  — PST900 dataset config
├── train_FMB.py                     — FMB training script
├── train_PST900.py                  — PST900 training script
├── train_CART.py                    — CART training script
├── train_SUS.py                     — SUS training script
├── train_CART_KD.py                 — CART with Knowledge Distillation (journal ext.)
├── evaluate_FMB.py                  — FMB evaluation
├── evaluate_pst900.py               — PST900 evaluation
├── evaluate_CART.py                 — CART evaluation
├── evaluate_SUS.py                  — SUS evaluation
├── fps.py                           — FPS benchmarking
└── toolbox/                         — Training/eval utilities
```

## Build Requirements
- [ ] Clone repo: `git clone https://github.com/xiaodonguo/CMSSM.git`
- [ ] Create uv env: `uv venv .venv --python 3.10`
- [ ] Install PyTorch cu128: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128`
- [ ] Install mamba-ssm: `uv pip install mamba-ssm==1.0.1` (requires CUDA compilation)
- [ ] Install causal-conv1d: `uv pip install causal-conv1d==1.0.0`
- [ ] Install mmcv + mmengine: `uv pip install mmcv==2.2.0 mmengine`
- [ ] Install extras: `uv pip install timm einops fvcore ptflops`
- [ ] Download EfficientViT pretrained weights (b1_r288.pt)
- [ ] Download ConvNeXtV2 pretrained weights (atto)
- [ ] Download pre-trained CM-SSM weights (4 checkpoints from releases v1.0.1)
- [ ] Download datasets (FMB, PST900, CART, SUS)
- [ ] Fix absolute paths in encoder files (hardcoded /home/ubuntu/code/ paths)
- [ ] Run evaluation on all 4 datasets
- [ ] Measure FPS
- [ ] Profile training and inference
- [ ] Build custom CUDA kernels
- [ ] Port to MLX
- [ ] Dual-compute validation

## CUDA Kernel Targets
1. **Fused SS2D_rgbt** — THE BIG ONE: cross-modal selective scan with interleaved tokens. Currently calls mamba_ssm selective_scan_fn on 2*L length sequences. Fuse: in_proj → conv2d → cross-modal interleave → selective_scan → de-interleave → norm → gate → out_proj into single kernel.
2. **Fused CM_SSM Block** — Each CM_SSM does: cat→conv3x3 (left) + SS2D_rgbt (right) + cat→conv1x1 (merge). 3 branches with intermediate B×C×H×W allocations. Fuse into single mega-kernel.
3. **Optimized Cross-Modal Interleave/De-interleave** — The interleaving pattern (stack→view→cat→flip) is ~8 memory operations per block. Custom CUDA kernel to avoid materializing intermediates.
4. **Fused Dual-Stream Encoder** — RGB and thermal encoders are independent but identical architecture. Batch the two streams into single kernel calls for better GPU utilization.

## Defense Marketplace Value
CM-SSM brings **linear-complexity** cross-modal fusion to RGB-T segmentation — O(N) vs O(N²) for transformer attention. This means better scaling to high-resolution inputs (crucial for defense: wide-area surveillance, high-res satellite imagery). The Mamba backbone enables efficient sequence modeling of long visual sequences without the memory bottleneck of attention. Combined with TUNI (same author — real-time focus) and RTFDNet (robustness focus), provides a complete RGB-T segmentation toolkit. IROS 2025 publication demonstrates robotics-community acceptance. The terrain-specific KD in the journal extension is directly applicable to military terrain classification.

## Package Manager: uv (NEVER pip)
## Python: >= 3.10
## Torch: cu128 index
## Git prefix: [DEF-cmssm]

## Reference Implementation
Source code at: repositories/CMSSM/
Read ALL files in repositories/CMSSM/ before building. This is the paper's reference implementation.
