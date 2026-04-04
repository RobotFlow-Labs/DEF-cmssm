# Benchmarks — DEF-cmssm
# CM-SSM: Cross-Modal State Space Modeling for RGB-T Semantic Segmentation
# Paper: IROS 2025 | ArXiv: 2506.17869 | Journal: IEEE TASE
# Backbones: EfficientViT-B1 (CART/PST900) + ConvNeXtV2-atto (FMB/SUS)
# Input: 480×640 (H×W), RGB 3ch + Thermal 3ch

## Paper Baseline Results (to reproduce)

### Published Results (from README)
| Backbone | Dataset | Classes | mIoU (%) | Weights |
|----------|---------|---------|----------|---------|
| EfficientViT-B1 | CART | 12 | 75.1 | v1.0.1/CART.pth |
| EfficientViT-B1 | PST900 | 5 | 85.9 | v1.0.1/PST900.pth |
| ConvNeXtV2-A | SUS | — | 82.5 | v1.0.1/SUS.pth |
| ConvNeXtV2-A | FMB | 14 | 60.7 | v1.0.1/FMB.pth |

### Our Reproduction
| Backbone | Dataset | mIoU (%) | Params (M) | GFLOPs | FPS (GPU) |
|----------|---------|----------|-----------|--------|-----------|
| EfficientViT-B1 | CART | TBD | TBD | TBD | TBD |
| EfficientViT-B1 | PST900 | TBD | TBD | TBD | TBD |
| ConvNeXtV2-A | SUS | TBD | TBD | TBD | TBD |
| ConvNeXtV2-A | FMB | TBD | TBD | TBD | TBD |

### Per-Class IoU — CART (12 classes, EfficientViT-B1)
| Class | IoU (%) |
|-------|---------|
| (12 rows) | TBD |

### Per-Class IoU — FMB (14 classes, ConvNeXtV2-A)
| Class | IoU (%) |
|-------|---------|
| (14 rows) | TBD |

## Fusion Mode Comparison (KEY ABLATION)

### CART Dataset — EfficientViT-B1 Backbone
| Fusion Mode | mIoU (%) | Params (M) | GFLOPs | FPS | Type |
|-------------|---------|-----------|--------|-----|------|
| CM-SSM (proposed) | TBD | TBD | TBD | TBD | Cross-modal Mamba |
| M-SSM | TBD | TBD | TBD | TBD | Per-modal Mamba |
| add | TBD | TBD | TBD | TBD | Element-wise add |
| max | TBD | TBD | TBD | TBD | Element-wise max |
| cat | TBD | TBD | TBD | TBD | Concat + Conv |
| CMX | TBD | TBD | TBD | TBD | Cross-modal attention |
| sigma | TBD | TBD | TBD | TBD | Cross-modal Mamba (Sigma) |
| CDA | TBD | TBD | TBD | TBD | Channel-domain attention |
| TSFA | TBD | TBD | TBD | TBD | MAINet fusion |

### CM-SSM vs M-SSM Ablation (cross-modal vs per-modal Mamba)
| Fusion | CART mIoU | PST900 mIoU | FMB mIoU | FPS | Delta |
|--------|----------|------------|---------|-----|-------|
| CM-SSM | TBD | TBD | TBD | TBD | baseline |
| M-SSM | TBD | TBD | TBD | TBD | TBD |

## FPS / Latency

### EfficientViT-B1 + CM-SSM
| Hardware | FPS | Latency (ms) | GPU Memory (GB) |
|----------|-----|-------------|----------------|
| RTX 6000 Pro Blackwell | TBD | TBD | TBD |
| Mac Studio M-series (MLX) | TBD | TBD | TBD |
| Jetson Orin NX (PyTorch) | TBD | TBD | TBD |

### ConvNeXtV2-atto + CM-SSM
| Hardware | FPS | Latency (ms) | GPU Memory (GB) |
|----------|-----|-------------|----------------|
| RTX 6000 Pro Blackwell | TBD | TBD | TBD |
| Mac Studio M-series (MLX) | TBD | TBD | TBD |

### Backbone Comparison (same CM-SSM fusion)
| Backbone | Channels | Params (M) | GFLOPs | FPS | CART mIoU |
|----------|----------|-----------|--------|-----|----------|
| EfficientViT-B0 | [16,32,64,128] | TBD | TBD | TBD | TBD |
| EfficientViT-B1 | [32,64,128,256] | TBD | TBD | TBD | TBD |
| EfficientViT-B2 | [48,96,192,384] | TBD | TBD | TBD | TBD |
| ConvNeXtV2-atto | [40,80,160,320] | TBD | TBD | TBD | TBD |
| ConvNeXtV2-femto | [48,96,192,384] | TBD | TBD | TBD | TBD |

## CUDA Profiling — SS2D_rgbt Breakdown

### Per-Stage Time — EfficientViT-B1 + CM-SSM (B=1, 480×640)
| Stage | Channels | Feature Size | d_inner | Seq Len (2L) | SS2D_rgbt (ms) | CM_SSM Total (ms) |
|-------|----------|-------------|---------|-------------|---------------|-------------------|
| 1 | 32 | 120×160 | 64 | 38,400 | TBD | TBD |
| 2 | 64 | 60×80 | 128 | 9,600 | TBD | TBD |
| 3 | 128 | 30×40 | 256 | 2,400 | TBD | TBD |
| 4 | 256 | 15×20 | 512 | 600 | TBD | TBD |

### Within SS2D_rgbt — Detailed Breakdown (Stage 2, C=64)
| Operation | Time (ms) | % of SS2D_rgbt | Kernel Launches |
|-----------|----------|---------------|----------------|
| in_proj1 + in_proj2 (2× Linear) | TBD | TBD | 2 |
| conv2d1 + conv2d2 (2× DWConv) | TBD | TBD | 2 |
| SiLU activations | TBD | TBD | 2 |
| Interleave (stack+view+cat) | TBD | TBD | ~4 |
| Flip (reverse directions) | TBD | TBD | 1 |
| x_proj (einsum) | TBD | TBD | 1 |
| dt_projs (einsum) | TBD | TBD | 1 |
| **selective_scan_fn** | **TBD** | **TBD (expect dominant)** | **1** |
| De-interleave + direction sum | TBD | TBD | ~4 |
| norm1 + norm2 (2× LayerNorm) | TBD | TBD | 2 |
| SiLU gates | TBD | TBD | 2 |
| out_proj1 + out_proj2 | TBD | TBD | 2 |
| **SS2D_rgbt Total** | **TBD** | **100%** | **~24** |

### Full Forward Pass Breakdown
| Component | Time (ms) | % of Forward |
|-----------|----------|-------------|
| Encoder RGB (EfficientViT-B1) | TBD | TBD |
| Encoder Thermal (EfficientViT-B1) | TBD | TBD |
| CM_SSM Stage 1 | TBD | TBD |
| CM_SSM Stage 2 | TBD | TBD |
| CM_SSM Stage 3 | TBD | TBD |
| CM_SSM Stage 4 | TBD | TBD |
| Decoder MLP | TBD | TBD |
| Bilinear upsample 4× | TBD | TBD |
| **Total** | **TBD** | **100%** |

## Kernel Optimization Impact

| Kernel | Baseline (ms) | Optimized (ms) | Speedup |
|--------|--------------|----------------|---------|
| SS2D_rgbt stage 2 (C=64) | TBD | TBD | TBD (target 2.7x) |
| SS2D_rgbt stage 4 (C=256) | TBD | TBD | TBD (target 2.5x) |
| CM_SSM block stage 2 | TBD | TBD | TBD (target 1.7x) |
| Interleave (all stages) | TBD | TBD | TBD (target 5x) |
| **Full forward pass (EVit-B1)** | **TBD** | **TBD** | **TBD (target 1.9x)** |
| **Full forward pass (ConvV2-A)** | **TBD** | **TBD** | **TBD (target 1.8x)** |

## Training Performance

| Config | Dataset | Epochs | Time/epoch | GPU Memory (GB) | Best mIoU | Best Epoch |
|--------|---------|--------|-----------|----------------|----------|-----------|
| EVit-B1 + CM-SSM | CART | 300 | TBD | TBD | TBD | TBD |
| EVit-B1 + CM-SSM | PST900 | 300 | TBD | TBD | TBD | TBD |
| ConvV2-A + CM-SSM | FMB | 300 | TBD | TBD | TBD | TBD |
| ConvV2-A + CM-SSM | SUS | 300 | TBD | TBD | TBD | TBD |

## Knowledge Distillation Impact (Journal Extension)
| Teacher | Student | Dataset | Student mIoU (no KD) | Student mIoU (with KD) | Delta |
|---------|---------|---------|---------------------|----------------------|-------|
| EVit-B1 | EVit-B0 | CART | TBD | TBD | TBD |

## Dual-Compute Validation

| Backend | Dataset | mIoU | FPS | Inference (ms) |
|---------|---------|------|-----|---------------|
| CUDA (RTX 6000 Pro) | CART | TBD | reference | TBD |
| CUDA (RTX 6000 Pro) | FMB | TBD | reference | TBD |
| MLX (Mac Studio) | CART | TBD (within 0.5%) | TBD | TBD |
| MLX (Mac Studio) | FMB | TBD (within 0.5%) | TBD | TBD |
| Jetson Orin NX | CART | TBD (within 1%) | TBD | TBD |

## Cross-Module RGB-T Comparison (shared datasets)

| Module | Method | Fusion Type | FMB mIoU | PST900 mIoU | CART mIoU | Params (M) | FPS (GPU) |
|--------|--------|------------|---------|------------|----------|-----------|-----------|
| DEF-cmssm | CM-SSM (EVit-B1) | Mamba SSM | 60.7 | 85.9 | 75.1 | TBD | TBD |
| DEF-tuni | TUNI (384_2242) | Attention | TBD | TBD | TBD | TBD | TBD |
| DEF-rtfdnet | RTFDNet (MiT-B4) | CLIP align | TBD | TBD | N/A | TBD | TBD |
| DEF-hypsam | HyPSAM | SAM-based | TBD | TBD | N/A | TBD | TBD |

### Complexity Analysis (Linear vs Quadratic)
| Method | Attention Complexity | Seq Length (120×160) | Theoretical FLOPs |
|--------|---------------------|---------------------|-------------------|
| CM-SSM | O(N) — Mamba | 19,200 × 2 = 38,400 | TBD |
| TUNI Global | O(49²) — Pooled | 49 (fixed pool) | TBD |
| RTFDNet SR-Attn | O(N/r²) — Reduced | N/64 (sr_ratio=8) | TBD |
| Full Attention | O(N²) — Quadratic | 19,200 | TBD |

## Hardware & Methodology
- RTX 6000 Pro Blackwell (training + evaluation)
- Mac Studio M-series (MLX local dev)
- Jetson Orin NX (edge deployment)
- Input: 480×640, RGB (3ch) + Thermal (3ch)
- FPS measurement: 100 warmup + 1000 iterations, mean ± std
- GFLOPs: ptflops (get_model_complexity_info) + fvcore verification
- Results stored as JSON: `results_*.json`

---
*Updated 2026-04-04 by ANIMA Research Agent*
