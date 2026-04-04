#!/usr/bin/env bash
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WEIGHTS_DIR="${MODULE_DIR}/weights"
BACKBONE_DIR="${WEIGHTS_DIR}/backbone"
CKPT_DIR="${WEIGHTS_DIR}/checkpoints"
DATASET_DIR="/mnt/forge-data/datasets/wave8"

mkdir -p "$BACKBONE_DIR" "$CKPT_DIR" "$DATASET_DIR"

echo "=== Downloading EfficientViT backbone weights ==="
# EfficientViT-B1 pretrained (from MIT HAN Lab)
if [ ! -f "$BACKBONE_DIR/efficientvit_b1_r288.pt" ]; then
    wget -q --show-progress -O "$BACKBONE_DIR/efficientvit_b1_r288.pt" \
        "https://huggingface.co/han-cai/efficientvit-cls/resolve/main/b1-r288.pt" || \
    echo "[WARN] EfficientViT-B1 download failed — try manual download"
fi

echo "=== Downloading CM-SSM pretrained checkpoints ==="
RELEASE_URL="https://github.com/xiaodonguo/CMSSM/releases/download/v1.0.1"
for ds in CART PST900 FMB SUS; do
    if [ ! -f "$CKPT_DIR/${ds}.pth" ]; then
        wget -q --show-progress -O "$CKPT_DIR/${ds}.pth" "${RELEASE_URL}/${ds}.pth" || \
        echo "[WARN] ${ds}.pth download failed"
    fi
done

echo "=== Asset download complete ==="
echo "Backbone weights: $BACKBONE_DIR"
echo "Checkpoints: $CKPT_DIR"
echo ""
echo "DATASETS: You need to manually download CART, PST900, FMB, SUS datasets."
echo "Place them under: $DATASET_DIR/"
echo "Expected structure:"
echo "  $DATASET_DIR/CART/rgbt_splits/rgb_train.txt"
echo "  $DATASET_DIR/PST900/train.txt"
