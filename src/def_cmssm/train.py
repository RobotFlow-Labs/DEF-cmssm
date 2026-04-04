"""ANIMA DEF-cmssm training pipeline — config-driven, ANIMA standards compliant."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

# Add reference repo to path
_REPO_ROOT = Path(__file__).resolve().parents[2] / "repositories" / "CMSSM"
sys.path.insert(0, str(_REPO_ROOT))

from toolbox import get_dataset, get_logger, get_model, Ranger, setup_seed
from toolbox.metrics_CART import averageMeter
from toolbox.metrics_CART import runningScore as runningScore_CART
from toolbox.metrics_PST900 import runningScore as runningScore_PST900
from Loss.dice import DiceLoss


# ---------- Dataset-specific loss configs ----------
DATASET_LOSS_WEIGHTS = {
    "cart": {
        "n_classes": 12,
        "class_weight": [50.2527, 50.4935, 4.8389, 6.3680, 24.0135,
                         26.3811, 9.7799, 14.6093, 16.8741, 2.7478, 49.2211, 50.2928],
    },
    "pst900": {
        "n_classes": 5,
        "class_weight": [1.4537, 44.2457, 31.6650, 46.4071, 30.1391],
    },
    "fmb": {
        "n_classes": 14,
        "class_weight": None,  # uses default CE
    },
    "sus": {
        "n_classes": 9,
        "class_weight": [1.5105, 16.6591, 29.4238, 34.6315, 40.0845,
                         41.4357, 47.9794, 45.3725, 44.9000],
    },
}


class TrainLoss(nn.Module):
    def __init__(self, n_classes: int, class_weight=None):
        super().__init__()
        if class_weight is not None:
            w = torch.tensor(class_weight, dtype=torch.float32)
            self.ce = nn.CrossEntropyLoss(weight=w)
        else:
            self.ce = nn.CrossEntropyLoss()
        self.dice = DiceLoss(mode="multiclass", ignore_index=-1)

    def forward(self, pred, target):
        return self.ce(pred, target) + self.dice(pred, target)


class CheckpointManager:
    def __init__(self, save_dir: Path, keep_top_k: int = 2, metric: str = "val_miou", mode: str = "max"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k
        self.mode = mode
        self.history: list[tuple[float, Path]] = []

    def save(self, state: dict, metric_value: float, step: int):
        path = self.save_dir / f"checkpoint_step{step:06d}.pth"
        torch.save(state, path)
        self.history.append((metric_value, path))
        self.history.sort(key=lambda x: x[0], reverse=(self.mode == "max"))
        while len(self.history) > self.keep_top_k:
            _, old_path = self.history.pop()
            old_path.unlink(missing_ok=True)
        best_val, best_path = self.history[0]
        shutil.copy2(best_path, self.save_dir / "best.pth")


class EarlyStopping:
    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = float("-inf") if mode == "max" else float("inf")
        self.counter = 0

    def step(self, metric: float) -> bool:
        improved = (metric > self.best + self.min_delta) if self.mode == "max" \
                   else (metric < self.best - self.min_delta)
        if improved:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-7):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            scale = self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            scale = 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg["lr"] = max(self.min_lr, base_lr * scale)

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state):
        self.current_step = state["current_step"]


def check_gpu_memory(max_util: float = 0.80):
    for i in range(torch.cuda.device_count()):
        total = torch.cuda.get_device_properties(i).total_mem
        used = torch.cuda.memory_allocated(i)
        util = used / total
        if util > max_util:
            raise RuntimeError(
                f"GPU {i} at {util*100:.1f}% VRAM — exceeds {max_util*100:.0f}% cap. "
                f"Reduce batch_size or enable gradient checkpointing."
            )


def run(args):
    with open(args.config, "r") as fp:
        cfg = json.load(fp)

    setup_seed(cfg.get("seed", 42))

    # GPU setup
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if int(x) >= 0]
    if gpu_ids:
        torch.cuda.set_device(gpu_ids[0])

    dataset_name = cfg["dataset"].strip().lower()
    loss_cfg = DATASET_LOSS_WEIGHTS.get(dataset_name, {"n_classes": cfg["n_classes"], "class_weight": None})

    # Logging
    module_name = "DEF-cmssm"
    log_dir = Path("/mnt/artifacts-datai/logs") / module_name
    log_dir.mkdir(parents=True, exist_ok=True)
    run_name = f'{time.strftime("%Y%m%d_%H%M")}_{dataset_name}_{cfg["model_name"]}'
    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir)
    logger = get_logger(str(run_dir))

    # Checkpoint dir
    ckpt_dir = Path("/mnt/artifacts-datai/checkpoints") / module_name / run_name
    ckpt_manager = CheckpointManager(ckpt_dir, keep_top_k=2, metric="val_miou", mode="max")

    # Model
    model = get_model(cfg)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"[CONFIG] {args.config}")
    logger.info(f"[MODEL] {cfg['model_name']} — {n_params:.2f}M parameters")
    logger.info(f"[DATA] dataset={dataset_name}, n_classes={cfg['n_classes']}")

    model.to(gpu_ids[0])
    if len(gpu_ids) > 1:
        model = nn.DataParallel(model, gpu_ids)

    # Data
    datasets = get_dataset(cfg)
    trainset = datasets[0]
    valset = datasets[1]
    batch_size = cfg.get("ims_per_gpu", 8)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=cfg.get("num_workers", 4), pin_memory=True, drop_last=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False,
                            num_workers=cfg.get("num_workers", 4), pin_memory=True)
    logger.info(f"[BATCH] batch_size={batch_size}")
    logger.info(f"[DATA] train={len(trainset)} samples, val={len(valset)} samples")

    # Optimizer + scheduler
    lr = cfg.get("lr_start", 1e-4)
    epochs = cfg.get("epochs", 300)
    optimizer = Ranger(model.parameters(), lr=lr, weight_decay=cfg.get("weight_decay", 5e-4))

    total_steps = epochs * len(train_loader)
    warmup_steps = int(total_steps * 0.05)
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps, total_steps)

    scaler = amp.GradScaler()
    train_criterion = TrainLoss(cfg["n_classes"], loss_cfg.get("class_weight")).cuda()
    val_criterion = nn.CrossEntropyLoss().cuda()
    early_stopping = EarlyStopping(patience=20, mode="max")

    # Resume
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=f"cuda:{gpu_ids[0]}")
        if "model" in ckpt:
            model_state = ckpt["model"]
            start_epoch = ckpt.get("epoch", 0)
            optimizer.load_state_dict(ckpt["optimizer"])
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
        else:
            model_state = ckpt
        if hasattr(model, "module"):
            model.module.load_state_dict(model_state, strict=False)
        else:
            model.load_state_dict(model_state, strict=False)
        logger.info(f"[RESUME] from {args.resume}, epoch={start_epoch}")

    logger.info(f"[TRAIN] {epochs} epochs, lr={lr}, optimizer=Ranger")
    logger.info(f"[CKPT] save every epoch, keep best 2")
    logger.info(f"[GPU] {len(gpu_ids)} GPU(s): {gpu_ids}")

    # Metrics
    train_loss_meter = averageMeter()
    test_loss_meter = averageMeter()
    MetricsClass = runningScore_CART if dataset_name == "cart" else runningScore_PST900
    running_metrics = MetricsClass(cfg["n_classes"], ignore_index=cfg.get("id_unlabel"))
    best_miou = 0.0
    global_step = start_epoch * len(train_loader)

    for ep in range(start_epoch, epochs):
        model.train()
        train_loss_meter.reset()

        for i, sample in enumerate(train_loader):
            optimizer.zero_grad()
            image = sample["image"].cuda()
            thermal = sample["thermal"].cuda()
            label = sample["label"].cuda()

            with amp.autocast():
                predict = model(image, thermal)
                loss = train_criterion(predict, label)

            # NaN detection
            if torch.isnan(loss):
                logger.info("[FATAL] Loss is NaN — stopping training")
                return

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_loss_meter.update(loss.item())
            global_step += 1

        torch.cuda.empty_cache()

        # Validation
        with torch.no_grad():
            model.eval()
            running_metrics.reset()
            test_loss_meter.reset()

            for sample in val_loader:
                image = sample["image"].cuda()
                thermal = sample["thermal"].cuda()
                label = sample["label"].cuda()
                predict = model(image, thermal)
                loss = val_criterion(predict, label)
                test_loss_meter.update(loss.item())
                pred_np = predict.max(1)[1].cpu().numpy()
                label_np = label.cpu().numpy()
                running_metrics.update(label_np, pred_np)

        scores = running_metrics.get_scores()[0]
        test_miou = scores["mIou: "]
        test_macc = scores["class_acc: "]
        train_loss = train_loss_meter.avg
        test_loss = test_loss_meter.avg
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch [{ep+1:3d}/{epochs}] "
            f"loss={train_loss:.4f}/{test_loss:.4f} "
            f"mIoU={test_miou:.4f} mPA={test_macc:.4f} "
            f"lr={current_lr:.2e}"
        )

        # Checkpoint
        state = {
            "model": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": ep + 1,
            "step": global_step,
            "miou": test_miou,
            "config": cfg,
        }
        ckpt_manager.save(state, test_miou, global_step)

        if test_miou > best_miou:
            best_miou = test_miou
            logger.info(f"  *** New best mIoU: {best_miou:.4f} ***")

        # Early stopping
        if early_stopping.step(test_miou):
            logger.info(f"[EARLY STOP] No improvement for {early_stopping.patience} epochs")
            break

    logger.info(f"[DONE] Training complete. Best mIoU: {best_miou:.4f}")
    logger.info(f"[CKPT] Best checkpoint: {ckpt_dir / 'best.pth'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEF-cmssm training")
    parser.add_argument("--config", type=str, required=True, help="JSON config file")
    parser.add_argument("--gpu_ids", type=str, default="0", help="GPU IDs")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    args = parser.parse_args()
    run(args)
