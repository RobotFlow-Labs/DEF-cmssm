# NEXT_STEPS — DEF-cmssm
## Last Updated: 2026-04-04
## Status: TRAINING IN PROGRESS — PST900 eval verified, exports complete
## MVP Readiness: 75%
## Total PRDs: 11 (72 hours estimated)
## Critical Path: PRD-001 ✅ ��� PRD-002 ✅ → PRD-003 ✅ → PRD-005 → PRD-006

---

### Completed
- [x] **PRD-001**: Environment setup — torch 2.11+cu128, mamba-ssm 2.3.1, causal-conv1d 1.6.1
- [x] **PRD-002**: Dataset download — PST900, FMB, CART on disk at /mnt/forge-data/datasets/wave8/
- [x] **PRD-003**: Evaluation baseline — PST900 verified: **85.87% mIoU** (paper: 85.9%)
- [x] Reference code patched: fusion.py (stripped broken imports), Efficientvit.py (configurable paths)
- [x] ANIMA training pipeline: config-driven, checkpointing, early stopping, warmup+cosine
- [x] Full export pipeline: pth → safetensors → ONNX → TRT FP16 → TRT FP32
- [x] HuggingFace push: ilessio-aiflowlab/DEF-cmssm (all checkpoints + PST900 exports)

### In Progress
- [ ] **PRD-008**: Training PST900 from scratch — epoch 25/200, mIoU 0.7542 (GPU 3, batch_size=6)

### TODO
- [ ] **PRD-004**: FPS benchmarking — both backbone configs
- [ ] **PRD-005**: CUDA profiling — Mamba bottleneck analysis
- [ ] **PRD-006**: Custom CUDA kernels (4 targets)
- [ ] **PRD-007**: MLX port (selective_scan challenge)
- [ ] **PRD-009**: Fusion mode ablation
- [ ] **PRD-010**: RGB-T family integration
- [ ] **PRD-011**: Edge deployment Jetson
- [ ] Evaluate CART, FMB, SUS (need to create CART splits)
- [ ] Train CART, FMB, SUS variants
- [ ] Docker serving infrastructure

### Key Results
| Dataset | Backbone | Paper mIoU | Our mIoU | Status |
|---------|----------|-----------|----------|--------|
| PST900  | EfficientViT-B1 | 85.9% | 85.87% | ✅ Verified |
| CART    | EfficientViT-B1 | 75.1% | — | Needs splits |
| FMB     | ConvNeXtV2-A | 60.7% | — | Needs ConvNeXtV2 |
| SUS     | ConvNeXtV2-A | 82.5% | — | Needs ConvNeXtV2 |

### Exports (PST900)
| Format | Size | Path |
|--------|------|------|
| pth | 49MB | /mnt/artifacts-datai/exports/DEF-cmssm/pst900/model.pth |
| safetensors | 49MB | /mnt/artifacts-datai/exports/DEF-cmssm/pst900/model.safetensors |
| ONNX | 105MB | /mnt/artifacts-datai/exports/DEF-cmssm/pst900/model.onnx |
| TRT FP16 | 34MB | /mnt/artifacts-datai/exports/DEF-cmssm/pst900/model_fp16.trt |
| TRT FP32 | 62MB | /mnt/artifacts-datai/exports/DEF-cmssm/pst900/model_fp32.trt |

### Datasets Available (DO NOT download)
- PST900: /mnt/forge-data/datasets/wave8/PST900/ (6.1GB)
- FMB: /mnt/forge-data/datasets/wave8/FMB/ (2.2GB)
- CART: /mnt/forge-data/datasets/wave8/CART/ (4.5GB)
- MFNet: /mnt/forge-data/datasets/wave8/mfnet_rgbt/ (212MB)
- RGBT-CC: /mnt/forge-data/datasets/RGBT-CC/ (908MB)

---
*Updated 2026-04-04 by ANIMA Autopilot Agent*
