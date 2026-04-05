"""ANIMA DEF-cmssm serving node — RGB-T semantic segmentation."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

# Add reference repo to path
_REPO_ROOT = Path(__file__).resolve().parents[2] / "repositories" / "CMSSM"
sys.path.insert(0, str(_REPO_ROOT))

try:
    from anima_serve.node import AnimaNode
except ImportError:
    # Fallback for development without anima_serve
    class AnimaNode:
        def __init__(self, *a, **kw): pass
        def setup_inference(self): pass
        def process(self, data): pass
        def get_status(self) -> dict: return {}


# Dataset configs for class names
DATASET_CLASSES = {
    "pst900": ["Background", "Fire-Extinguisher", "Backpack", "Hand-Drill", "Survivor"],
    "cart": ["Unlabeled", "Road", "Sidewalk", "Building", "Wall", "Fence",
             "Pole", "Vegetation", "Terrain", "Sky", "Person", "Car"],
    "fmb": list(range(14)),
}


class CMSSMNode(AnimaNode):
    """CM-SSM RGB-T segmentation inference node."""

    def setup_inference(self):
        """Load model and weights."""
        os.environ["CMSSM_WEIGHTS_DIR"] = str(
            Path(__file__).resolve().parents[2] / "weights" / "backbone"
        )

        from toolbox import get_model

        dataset = os.environ.get("ANIMA_DATASET", "pst900")
        n_classes = {"pst900": 5, "cart": 12, "fmb": 14, "sus": 9}.get(dataset, 5)

        cfg = {
            "model_name": "b1_CM-SSM",
            "n_classes": n_classes,
            "inputs": "rgbt",
            "embed_dim": 128,
        }
        self.model = get_model(cfg)
        self.n_classes = n_classes
        self.dataset = dataset
        self.class_names = DATASET_CLASSES.get(dataset, [str(i) for i in range(n_classes)])

        # Load weights
        weight_path = os.environ.get(
            "ANIMA_WEIGHT_PATH",
            str(
                Path(__file__).resolve().parents[2]
                / "weights"
                / "checkpoints"
                / "PST900.pth"
            ),
        )
        ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
        state = ckpt if not isinstance(ckpt, dict) or "model" not in ckpt else ckpt["model"]
        clean = {}
        for k, v in state.items():
            nk = k.replace("module.", "")
            if nk.startswith("fusion_fusion."):
                nk = "fusion_module.fusion." + nk[len("fusion_fusion."):]
            nk = nk.replace(".context_main.", ".context_module.main.")
            nk = nk.replace(".local_main.", ".local_module.main.")
            clean[nk] = v
        self.model.load_state_dict(clean, strict=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

        self.rgb_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.thermal_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.449, 0.449, 0.449], [0.226, 0.226, 0.226]),
        ])

        print(f"[CMSSM] Model loaded: {dataset}, {n_classes} classes, device={self.device}")

    def process(self, input_data: dict) -> dict:
        """Run inference on RGB + thermal pair.

        Args:
            input_data: dict with 'rgb' (np.ndarray HxWx3) and 'thermal' (np.ndarray HxWx3)

        Returns:
            dict with 'segmentation' (np.ndarray HxW) and 'classes' (list of class names)
        """
        rgb_np = input_data["rgb"]
        thermal_np = input_data["thermal"]

        from PIL import Image
        rgb_pil = Image.fromarray(rgb_np) if isinstance(rgb_np, np.ndarray) else rgb_np
        thermal_pil = (
            Image.fromarray(thermal_np) if isinstance(thermal_np, np.ndarray) else thermal_np
        )

        rgb_t = self.rgb_transform(rgb_pil).unsqueeze(0).to(self.device)
        thermal_t = self.thermal_transform(thermal_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(rgb_t, thermal_t)
            pred = output.argmax(dim=1).squeeze(0).cpu().numpy()

        return {
            "segmentation": pred,
            "classes": self.class_names,
            "n_classes": self.n_classes,
        }

    def get_status(self) -> dict:
        return {
            "model_loaded": self.model is not None,
            "dataset": self.dataset,
            "n_classes": self.n_classes,
            "device": str(self.device),
        }


if __name__ == "__main__":
    node = CMSSMNode()
    node.setup_inference()
    # Quick test
    dummy_rgb = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    dummy_thermal = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    result = node.process({"rgb": dummy_rgb, "thermal": dummy_thermal})
    print(f"Output shape: {result['segmentation'].shape}, classes: {result['classes']}")
