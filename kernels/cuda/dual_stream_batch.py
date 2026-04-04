"""PyTorch-level scaffold for dual-stream encoder batching optimization."""

from __future__ import annotations

import torch


def dual_stream_batch_forward(backbone, rgb: torch.Tensor, thermal: torch.Tensor):
    """Batch RGB/T in one forward and split features back.

    Expects backbone output shape as list[Tensor] where batch is first dim.
    """
    batch = rgb.shape[0]
    stacked = torch.cat([rgb, thermal], dim=0)
    features = backbone(stacked)
    rgb_feats = [f[:batch] for f in features]
    thermal_feats = [f[batch:] for f in features]
    return rgb_feats, thermal_feats
