"""ANIMA DEF-cmssm export pipeline: pth -> safetensors -> ONNX -> TRT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2] / "repositories" / "CMSSM"
sys.path.insert(0, str(_REPO_ROOT))

from toolbox import get_model


def export_safetensors(state_dict: dict, output_path: Path):
    from safetensors.torch import save_file
    save_file(state_dict, str(output_path))
    print(f"[EXPORT] safetensors -> {output_path} ({output_path.stat().st_size / 1e6:.1f}MB)")


def export_onnx(model: torch.nn.Module, output_path: Path, input_size=(480, 640)):
    model.eval()
    device = next(model.parameters()).device
    rgb = torch.randn(1, 3, *input_size, device=device)
    t = torch.randn(1, 3, *input_size, device=device)

    torch.onnx.export(
        model, (rgb, t), str(output_path),
        input_names=["rgb", "thermal"],
        output_names=["segmentation"],
        dynamic_axes={
            "rgb": {0: "batch"},
            "thermal": {0: "batch"},
            "segmentation": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"[EXPORT] ONNX -> {output_path} ({output_path.stat().st_size / 1e6:.1f}MB)")


def export_trt(onnx_path: Path, output_dir: Path):
    """Use shared TRT toolkit for conversion."""
    trt_script = Path("/mnt/forge-data/shared_infra/trt_toolkit/export_to_trt.py")
    if not trt_script.exists():
        print("[WARN] TRT toolkit not found, skipping TRT export")
        return

    import subprocess
    for precision in ["fp16", "fp32"]:
        out_path = output_dir / f"model_{precision}.trt"
        cmd = [
            sys.executable, str(trt_script),
            "--onnx", str(onnx_path),
            "--output", str(out_path),
            "--precision", precision,
        ]
        print(f"[EXPORT] TRT {precision} -> {out_path}")
        subprocess.run(cmd, check=True)


def run(args):
    with open(args.config, "r") as fp:
        cfg = json.load(fp)

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load model
    model = get_model(cfg)
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    else:
        state_dict = ckpt
    # Remap legacy checkpoint keys
    clean_sd = {}
    for k, v in state_dict.items():
        nk = k.replace("module.", "")
        if nk.startswith("fusion_fusion."):
            nk = "fusion_module.fusion." + nk[len("fusion_fusion."):]
        nk = nk.replace(".context_main.", ".context_module.main.")
        nk = nk.replace(".local_main.", ".local_module.main.")
        clean_sd[nk] = v

    # Auto-detect embed_dim
    if "decoder.linear_c4.proj.bias" in clean_sd:
        ckpt_emb = clean_sd["decoder.linear_c4.proj.bias"].shape[0]
        model_emb = model.decoder.linear_c4.proj.bias.shape[0]
        if ckpt_emb != model_emb:
            from models.decoder.MLP import Decoder_MLP
            channels = [model.decoder.linear_c1.proj.weight.shape[1],
                       model.decoder.linear_c2.proj.weight.shape[1],
                       model.decoder.linear_c3.proj.weight.shape[1],
                       model.decoder.linear_c4.proj.weight.shape[1]]
            model.decoder = Decoder_MLP(in_channels=channels, embed_dim=ckpt_emb,
                                       num_classes=cfg["n_classes"])

    model.load_state_dict(clean_sd, strict=False)
    model.to(device)
    model.eval()

    dataset_name = cfg["dataset"].strip().lower()
    output_dir = Path("/mnt/artifacts-datai/exports/DEF-cmssm") / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save clean pth
    pth_path = output_dir / "model.pth"
    torch.save(clean_sd, pth_path)
    print(f"[EXPORT] pth -> {pth_path} ({pth_path.stat().st_size / 1e6:.1f}MB)")

    # 2. safetensors
    st_path = output_dir / "model.safetensors"
    export_safetensors(clean_sd, st_path)

    # 3. ONNX
    onnx_path = output_dir / "model.onnx"
    try:
        export_onnx(model, onnx_path)
    except Exception as e:
        print(f"[WARN] ONNX export failed: {e}")
        onnx_path = None

    # 4. TRT
    if onnx_path and onnx_path.exists():
        try:
            export_trt(onnx_path, output_dir)
        except Exception as e:
            print(f"[WARN] TRT export failed: {e}")

    print(f"\n[DONE] All exports saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DEF-cmssm export")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    run(args)
