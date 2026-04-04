# NEXT_STEPS — DEF-cmssm
## Last Updated: 2026-04-04
## Status: ENRICHED — Ready for build
## MVP Readiness: 0%
## Total PRDs: 11 (72 hours estimated)
## Critical Path: PRD-001 → PRD-002 → PRD-003 → PRD-005 → PRD-006

---

### Immediate Next Actions
1. Clone repo: `git clone https://github.com/xiaodonguo/CMSSM.git`
2. Create uv env: `uv venv .venv --python 3.10`
3. Install PyTorch cu128
4. **Install mamba-ssm 1.0.1** (CRITICAL — requires CUDA compilation of selective_scan_fn)
5. **Install causal-conv1d 1.0.0** (CRITICAL — fast 1D causal conv for Mamba)
6. Install mmcv 2.2.0, timm, einops, fvcore, ptflops
7. Download EfficientViT-B1 + ConvNeXtV2-atto backbone weights
8. Download 4 CM-SSM pre-trained checkpoints from releases v1.0.1
9. Download 4 datasets (FMB, PST900, CART, SUS)
10. Fix absolute paths in encoder files (hardcoded /home/ubuntu/ paths)
11. Fix import paths (`proposed.` → `models.`)
12. Stub comparison method imports (CMX, MAINet, MDNet, sigma)
13. Run evaluation on all 4 datasets — reproduce paper results

### What This Module Does
CM-SSM uses Mamba state space models for cross-modal RGB-T feature fusion. The key innovation is `SS2D_rgbt` which interleaves RGB and thermal tokens into a single sequence and processes them jointly through a 4-directional selective scan (Mamba S6). This achieves O(N) complexity vs O(N²) for transformer attention, enabling efficient fusion of long visual sequences. Uses dual-stream backbones (EfficientViT or ConvNeXtV2) with CM-SSM fusion at 4 stages. Same author as DEF-tuni — designed as the accuracy-focused complement to TUNI's speed focus. IROS 2025 paper with journal extension at IEEE TASE adding terrain-specific knowledge distillation.

### Key Results to Reproduce
- CART: 75.1 mIoU (EfficientViT-B1)
- PST900: 85.9 mIoU (EfficientViT-B1)
- SUS: 82.5 mIoU (ConvNeXtV2-A)
- FMB: 60.7 mIoU (ConvNeXtV2-A)
- Fusion mode ablation: CM-SSM vs M-SSM vs add/max/cat/CMX/sigma/CDA/TSFA

### TODO (by PRD)
- [ ] **PRD-001**: Environment setup — mamba-ssm 1.0.1 + causal-conv1d + weights (6h)
- [ ] **PRD-002**: Dataset download — FMB + PST900 + CART + SUS (5h)
- [ ] **PRD-003**: Evaluation baseline — all 4 datasets, reproduce paper results (6h)
- [ ] **PRD-004**: FPS benchmarking — both backbone configs, compare with TUNI (4h)
- [ ] **PRD-005**: CUDA profiling — Mamba bottleneck analysis, selective_scan overhead (5h)
- [ ] **PRD-006**: Custom CUDA kernels — 4 targets: cm_ss2d_fused, cm_ssm_block, cm_interleave, dual_stream_batch (16h)
- [ ] **PRD-007**: MLX port — KEY CHALLENGE: Mamba selective scan on MLX (8h)
- [ ] **PRD-008**: Training from scratch — validate training pipeline (6h)
- [ ] **PRD-009**: Fusion mode ablation — compare 8+ fusion modes (4h)
- [ ] **PRD-010**: RGB-T family integration — compare with TUNI/RTFDNet/HyPSAM (3h)
- [ ] **PRD-011**: Edge deployment Jetson — mamba-ssm on Jetson (5h)

### Blockers
- **mamba-ssm compilation**: Requires CUDA development headers. May fail on systems without nvcc.
- **Import path issues**: Repo has mixed import prefixes (`proposed.`, `models.`, `model_others.`). Needs cleanup.
- **Comparison method dependencies**: fusion.py imports from `model_others.RGB_T.*` (CMX, MAINet, MDNet, sigma) — need stubs or comment out.

### Datasets/Models Needed
- FMB (~2GB) — shared with DEF-tuni, DEF-rtfdnet
- PST900 (~1GB) — shared with DEF-tuni, DEF-rtfdnet
- CART (~2GB) — shared with DEF-tuni
- SUS (~1-2GB) — author's own dataset, CMSSM-specific
- EfficientViT-B1 pretrained (~50MB)
- ConvNeXtV2-atto pretrained (~15MB)
- CM-SSM checkpoints ×4 (~200MB total)
- Total: ~7-9GB datasets + ~300MB models

### Kernel IP Targets (shared across ANIMA)
1. **Fused Cross-Modal SS2D** → NOVEL: first fused cross-modal Mamba kernel. Reusable by ANY cross-modal SSM model (2-3x speedup)
2. **Zero-Copy Interleave/De-interleave** → General-purpose cross-modal sequence interleaving (4-5x speedup for interleave step)
3. **Fused CM_SSM Block** → Eliminates intermediate cat allocations (1.5x)
4. **Batched Dual-Stream Encoder** → PyTorch-level optimization (1.3x)

### Related Modules
- **DEF-tuni** — SAME AUTHOR (Xiaodong Guo), real-time focus, complementary to CM-SSM
- **DEF-rtfdnet** — RGB-T dual-stream SegFormer, robustness focus, shares FMB/PST900
- **DEF-hypsam** — RGB-T SOD + SAM, shares thermal processing
- **DEF-rgbtcc** — RGB-T crowd counting, shares thermal sensors

### MLX Port Challenge
The selective_scan_fn from mamba-ssm is CUDA-only. MLX port requires reimplementing the selective scan — either as a sequential scan in pure MLX (simple but slow) or as a parallel associative scan with a custom Metal kernel (fast but complex). This is the biggest technical risk in the MLX port.

---
*Updated 2026-04-04 by ANIMA Research Agent*
