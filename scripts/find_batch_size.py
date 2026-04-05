"""Auto-detect optimal batch size for CM-SSM training on target GPU."""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1] / "repositories" / "CMSSM"
sys.path.insert(0, str(_REPO_ROOT))


def find_optimal_batch(
    target_util: float = 0.65,
    max_batch: int = 32,
    input_size: tuple = (3, 480, 640),
    n_classes: int = 5,
):
    """Binary search for optimal batch size targeting given VRAM utilization."""
    os.environ.setdefault(
        "CMSSM_WEIGHTS_DIR",
        str(Path(__file__).resolve().parents[1] / "weights" / "backbone"),
    )

    from toolbox import get_model

    device = torch.device("cuda")
    total_mem = torch.cuda.get_device_properties(0).total_mem

    cfg = {
        "model_name": "b1_CM-SSM",
        "n_classes": n_classes,
        "inputs": "rgbt",
        "embed_dim": 128,
    }
    model = get_model(cfg).to(device).train()

    optimal = 1
    for bs in range(1, max_batch + 1):
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()

        rgb = None
        t = None
        label = None
        out = None
        loss = None

        try:
            rgb = torch.randn(bs, *input_size, device=device)
            t = torch.randn(bs, *input_size, device=device)
            label = torch.randint(
                0, n_classes, (bs, input_size[1], input_size[2]), device=device
            )

            out = model(rgb, t)
            loss = torch.nn.functional.cross_entropy(out, label)
            loss.backward()

            peak = torch.cuda.max_memory_allocated()
            util = peak / total_mem

            print(
                f"  batch_size={bs:3d} -> "
                f"{peak / 1e9:.2f}GB / {total_mem / 1e9:.2f}GB = {util * 100:.1f}%"
            )

            if util <= 0.80:
                optimal = bs
            if util > target_util:
                break

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  batch_size={bs:3d} -> OOM")
                break
            raise
        finally:
            del rgb, t, label, out, loss
            model.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    del model
    torch.cuda.empty_cache()

    print(f"\n[BATCH] Optimal batch_size={optimal} (target {target_util * 100:.0f}% VRAM)")
    return optimal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=0.65)
    parser.add_argument("--max-batch", type=int, default=32)
    parser.add_argument("--n-classes", type=int, default=5)
    args = parser.parse_args()
    find_optimal_batch(
        target_util=args.target, max_batch=args.max_batch, n_classes=args.n_classes
    )
