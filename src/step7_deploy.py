"""
step7_deploy.py
───────────────
Two deployment modes:

  gradio  (default) — interactive web app for real-time inference
  onnx              — export model to ONNX for edge / production use

Usage
─────
    python -m src.step7_deploy --mode gradio
    python -m src.step7_deploy --mode onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.config import CFG, PATHS, ensure_dirs
from src.step3_unet_model import ResUNet


# ── helpers ───────────────────────────────────────────────────────────────────

def _preprocess(img: Image.Image) -> torch.Tensor:
    """Convert PIL image to [1,1,H,W] float32 tensor."""
    arr = (
        np.asarray(
            img.convert("L").resize((CFG.image_size, CFG.image_size), Image.BILINEAR),
            dtype=np.float32,
        )
        / 255.0
    )
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)   # [1,1,H,W]


def _to_pil(arr01: np.ndarray) -> Image.Image:
    """Convert [0,1] float array to 8-bit PIL image."""
    return Image.fromarray(
        np.clip(arr01 * 255.0, 0, 255).astype(np.uint8), mode="L"
    )


# ── predictor ─────────────────────────────────────────────────────────────────

class Predictor:
    def __init__(self, device: str = CFG.device):
        ensure_dirs()
        self.device = torch.device(device)
        self.model  = ResUNet(
            CFG.in_channels, CFG.out_channels, CFG.base_channels
        ).to(self.device)
        ckpt = torch.load(
            PATHS.ckpt_dir / "best_model.pt", map_location=self.device
        )
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        print(
            f"Model loaded  "
            f"(epoch {ckpt.get('epoch','?')}, "
            f"val_loss={ckpt.get('val_loss',0):.5f})"
        )

    @torch.no_grad()
    def infer(self, image: Image.Image) -> tuple[Image.Image, Image.Image]:
        """
        Run inference on a single hologram image.

        Returns
        ───────
        (amplitude_pil, phase_pil)  — both 8-bit grayscale PIL images
        """
        x     = _preprocess(image).to(self.device)
        pred  = self.model(x)
        amp   = pred[:, 0:1].clamp(0, 1)[0, 0].cpu().numpy()
        phase = torch.tanh(pred[:, 1:2])[0, 0].cpu().numpy()   # [−1,1]
        return _to_pil(amp), _to_pil((phase + 1.0) * 0.5)


# ── Gradio ────────────────────────────────────────────────────────────────────

def launch_gradio() -> None:
    import gradio as gr

    predictor = Predictor()

    with gr.Blocks(title="Hologram Reconstruction — Res-AttU-Net") as demo:
        gr.Markdown(
            """
            ## Single Hologram Reconstruction
            **Deep Learning (Res-AttU-Net with Attention Gates)**
            *B.Tech Final Year Project — NIT Delhi, EE Dept.*

            Upload a grayscale hologram and click **Reconstruct** to obtain
            the predicted amplitude map and phase map.
            """
        )
        with gr.Row():
            inp = gr.Image(type="pil", label="Input Hologram", image_mode="L")
            btn = gr.Button("Reconstruct", variant="primary")
        with gr.Row():
            out_amp   = gr.Image(type="pil", label="Reconstructed Amplitude")
            out_phase = gr.Image(type="pil", label="Reconstructed Phase")

        btn.click(
            fn=lambda img: predictor.infer(img),
            inputs=inp,
            outputs=[out_amp, out_phase],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


# ── ONNX export ────────────────────────────────────────────────────────────────

def export_onnx(onnx_path: Path) -> None:
    device = torch.device("cpu")
    model  = ResUNet(CFG.in_channels, CFG.out_channels, CFG.base_channels).to(device)
    ckpt   = torch.load(PATHS.ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, 1, CFG.image_size, CFG.image_size, device=device)
    torch.onnx.export(
        model, dummy, onnx_path.as_posix(),
        input_names=["hologram"],
        output_names=["amp_phase"],
        dynamic_axes={"hologram": {0: "batch"}, "amp_phase": {0: "batch"}},
        opset_version=17,
    )
    print(f"ONNX model exported → {onnx_path}")


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["gradio", "onnx"], default="gradio",
        help="gradio: launch web UI | onnx: export to ONNX",
    )
    parser.add_argument("--onnx-name", type=str, default="holo_resunet_att.onnx")
    args = parser.parse_args()

    ensure_dirs()
    if args.mode == "gradio":
        launch_gradio()
    else:
        export_onnx(PATHS.onnx_dir / args.onnx_name)
