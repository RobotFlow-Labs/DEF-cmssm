"""ANIMA DEF-cmssm evaluation pipeline."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2] / "repositories" / "CMSSM"
sys.path.insert(0, str(_REPO_ROOT))

from toolbox import get_dataset, get_model, setup_seed
from toolbox.metrics_CART import averageMeter
from toolbox.metrics_CART import runningScore as runningScore_CART
from toolbox.metrics_PST900 import runningScore as runningScore_PST900


def _remap_checkpoint_keys(state_dict: dict) -> dict:
    """Remap legacy checkpoint keys to match current model architecture."""
    clean = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "")
        # Fusion module naming: fusion_fusion.X → fusion_module.fusion.X
        if nk.startswith("fusion_fusion."):
            nk = "fusion_module.fusion." + nk[len("fusion_fusion."):]
        # EfficientViT backbone version mismatch:
        # Old: context_main.X → New: context_module.main.X
        # Old: local_main.X → New: local_module.main.X
        nk = nk.replace(".context_main.", ".context_module.main.")
        nk = nk.replace(".local_main.", ".local_module.main.")
        clean[nk] = v
    return clean


def evaluate(args):
    with open(args.config, "r") as fp:
        cfg = json.load(fp)

    setup_seed(42)
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Model
    model = get_model(cfg)

    # Load checkpoint
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    # Handle DataParallel prefix + legacy key remapping
    clean_sd = _remap_checkpoint_keys(state_dict)

    # Auto-detect embed_dim from checkpoint
    if "decoder.linear_c4.proj.bias" in clean_sd:
        ckpt_emb = clean_sd["decoder.linear_c4.proj.bias"].shape[0]
        model_emb = model.decoder.linear_c4.proj.bias.shape[0]
        if ckpt_emb != model_emb:
            print(f"[INFO] Rebuilding decoder: embed_dim {model_emb} → {ckpt_emb}")
            from models.decoder.MLP import Decoder_MLP
            channels = [model.decoder.linear_c1.proj.weight.shape[1],
                       model.decoder.linear_c2.proj.weight.shape[1],
                       model.decoder.linear_c3.proj.weight.shape[1],
                       model.decoder.linear_c4.proj.weight.shape[1]]
            model.decoder = Decoder_MLP(in_channels=channels, embed_dim=ckpt_emb,
                                       num_classes=cfg["n_classes"])

    missing, unexpected = model.load_state_dict(clean_sd, strict=False)
    print(f"[LOAD] {len(missing)} missing, {len(unexpected)} unexpected keys")
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[MODEL] {cfg['model_name']} — {n_params:.2f}M params")

    # Dataset
    datasets = get_dataset(cfg)
    testset = datasets[-1]  # last split is test
    test_loader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    print(f"[DATA] {cfg['dataset']} test: {len(testset)} samples")

    # Evaluate
    # Use dataset-appropriate metrics (CART skips first 2 classes, PST900/FMB/SUS uses all)
    dataset_name = cfg["dataset"].strip().lower()
    MetricsClass = runningScore_CART if dataset_name == "cart" else runningScore_PST900
    running_metrics = MetricsClass(cfg["n_classes"], ignore_index=cfg.get("id_unlabel"))
    loss_meter = averageMeter()
    criterion = nn.CrossEntropyLoss().to(device)

    with torch.no_grad():
        for i, sample in enumerate(test_loader):
            image = sample["image"].to(device)
            thermal = sample["thermal"].to(device)
            label = sample["label"].to(device)

            predict = model(image, thermal)
            loss = criterion(predict, label)
            loss_meter.update(loss.item())

            pred_np = predict.max(1)[1].cpu().numpy()
            label_np = label.cpu().numpy()
            running_metrics.update(label_np, pred_np)

    scores, cls_iu, cls_acc = running_metrics.get_scores()
    print(f"\n{'='*50}")
    print(f"Dataset: {cfg['dataset']}")
    print(f"Model: {cfg['model_name']}")
    print(f"{'='*50}")
    print(f"Pixel Acc:  {scores['pixel_acc: ']:.4f}")
    print(f"Mean Acc:   {scores['class_acc: ']:.4f}")
    print(f"mIoU:       {scores['mIou: ']:.4f}")
    print(f"FW IoU:     {scores['fwIou: ']:.4f}")
    print(f"F1-Score:   {scores['F1-Score: ']:.4f}")
    print(f"Test Loss:  {loss_meter.avg:.4f}")
    print(f"\nPer-class IoU:")
    for cls_id, iou in cls_iu.items():
        print(f"  Class {cls_id}: {iou:.4f}")
    print(f"{'='*50}")

    return scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEF-cmssm evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    evaluate(args)
