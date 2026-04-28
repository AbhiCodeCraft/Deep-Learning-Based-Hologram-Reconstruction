"""
step6_test.py
─────────────
Quantitative evaluation: Res-AttU-Net vs TWO classical baselines.

This comparison is the scientific core of the project.  The claim is
that the deep learning approach outperforms classical numerical
reconstruction on every metric.

Baselines
─────────
Both use the same physical recording geometry as step1:
  U_h = √I  (weak-object / intensity-only estimate at the hologram plane)

  1. Fresnel (paraxial) back-propagation:
       U_obj ≈ IFFT[ FFT(U_h) · conj(H_Fresnel(z)) ]
     Classic and fast; valid for small propagation angles.
     Suffers from the twin-image artefact.

  2. ASM back-propagation:
       U_obj ≈ IFFT[ FFT(U_h) · conj(H_ASM(z)) ]
     Exact Helmholtz solution; evanescent waves zeroed.
     Same twin-image limitation as Fresnel but more accurate for
     large angles.

Both baselines use the exact per-sample z stored in manifest.jsonl.

Metrics (reported per method)
──────────────────────────────
  Amplitude PSNR (dB)  ↑   20·log10(1/√MSE)
  Amplitude SSIM       ↑   Structural Similarity Index
  Phase RMSE (rad)     ↓   √mean(Δφ_wrapped²)

Outputs
───────
  outputs/logs/test_metrics.json     — full numeric summary
  outputs/predictions/               — 24 visual comparison panels
  Printed comparison table
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.config import CFG, PATHS, ensure_dirs
from src.propagation import asm_reconstruct, fresnel_reconstruct
from src.step2_dataset import collect_triplets, make_dataloaders
from src.step3_unet_model import ResUNet
from src.step4_loss_metrics import phase_rmse_np, psnr_np, ssim_np
from src.utils import save_json


# ── helpers ───────────────────────────────────────────────────────────────────

def _save_gray(path: Path, arr: np.ndarray) -> None:
    Image.fromarray(
        np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L"
    ).save(path)


def _save_comparison_panel(
    path: Path,
    holo:          np.ndarray,
    true_amp:      np.ndarray,
    unet_amp:      np.ndarray,
    fresnel_amp:   np.ndarray,
    asm_amp:       np.ndarray,
    true_phase:    np.ndarray,
    unet_phase:    np.ndarray,
    fresnel_phase: np.ndarray,
    asm_phase:     np.ndarray,
) -> None:
    """
    Save a 9-panel horizontal strip:
    hologram | true_amp | unet_amp | fresnel_amp | asm_amp |
    (blank)  | true_ph  | unet_ph  | fresnel_ph  | asm_ph
    """
    def _norm(a): return np.clip(a * 255.0, 0, 255).astype(np.uint8)
    row_amp   = [holo, true_amp, unet_amp, fresnel_amp, asm_amp]
    row_phase = [
        np.zeros_like(holo),                # spacer under hologram col
        (true_phase    + 1.0) * 0.5,
        (unet_phase    + 1.0) * 0.5,
        (fresnel_phase + 1.0) * 0.5,
        (asm_phase     + 1.0) * 0.5,
    ]
    top    = np.concatenate([_norm(x) for x in row_amp],   axis=1)
    bottom = np.concatenate([_norm(x) for x in row_phase], axis=1)
    Image.fromarray(np.vstack([top, bottom]), mode="L").save(path)


def _mean(lst: List[float]) -> float:
    return float(np.mean(lst))


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs()
    device = torch.device(CFG.device)

    # ── data ──────────────────────────────────────────────────────────────
    samples = collect_triplets(
        PATHS.sim_holo_dir, PATHS.sim_amp_dir, PATHS.sim_phase_dir
    )
    _, _, test_loader = make_dataloaders(samples)

    # ── model ─────────────────────────────────────────────────────────────
    model = ResUNet(CFG.in_channels, CFG.out_channels, CFG.base_channels).to(device)
    ckpt  = torch.load(PATHS.ckpt_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"Loaded best_model.pt  "
        f"(epoch {ckpt.get('epoch','?')}, val_loss={ckpt.get('val_loss',0):.5f})"
    )

    # ── metric containers ─────────────────────────────────────────────────
    m: Dict[str, List[float]] = {
        f"{p}_{k}": []
        for p in ("unet", "fresnel", "asm")
        for k in ("amp_psnr", "amp_ssim", "phase_rmse")
    }

    lam   = CFG.wavelength_m
    pitch = CFG.pixel_pitch_m

    n_saved = 0

    # ── evaluation loop ───────────────────────────────────────────────────
    with torch.no_grad():
        for x, y, z in tqdm(test_loader, desc="Evaluating"):
            x   = x.to(device, non_blocking=True)
            pred = model(x)

            pred_amp   = pred[:, 0:1].clamp(0, 1).cpu().numpy()
            pred_phase = torch.tanh(pred[:, 1:2]).cpu().numpy()   # [−1,1]
            true_amp   = y[:, 0:1].numpy()
            true_phase = y[:, 1:2].numpy()                        # [−1,1]
            z_np       = z.numpy()                                # [B,1]

            bsz = pred_amp.shape[0]
            for i in range(bsz):
                ta  = true_amp[i, 0]
                tp  = true_phase[i, 0]         # ∈ [−1, 1]
                pa  = pred_amp[i, 0]
                pp  = pred_phase[i, 0]         # ∈ [−1, 1]
                zi  = float(z_np[i, 0])

                # phase in radians for RMSE
                tp_rad = tp * np.pi
                pp_rad = pp * np.pi

                # ── U-Net metrics ──────────────────────────────────────────
                m["unet_amp_psnr"].append(psnr_np(pa, ta))
                m["unet_amp_ssim"].append(ssim_np(pa, ta))
                m["unet_phase_rmse"].append(phase_rmse_np(pp_rad, tp_rad))

                # ── Fresnel baseline ───────────────────────────────────────
                holo_np = x[i, 0].detach().cpu().numpy()
                fa, fp  = fresnel_reconstruct(holo_np, zi, lam, pitch)
                fp_rad  = fp * np.pi
                m["fresnel_amp_psnr"].append(psnr_np(fa, ta))
                m["fresnel_amp_ssim"].append(ssim_np(fa, ta))
                m["fresnel_phase_rmse"].append(phase_rmse_np(fp_rad, tp_rad))

                # ── ASM baseline ───────────────────────────────────────────
                aa, ap  = asm_reconstruct(holo_np, zi, lam, pitch)
                ap_rad  = ap * np.pi
                m["asm_amp_psnr"].append(psnr_np(aa, ta))
                m["asm_amp_ssim"].append(ssim_np(aa, ta))
                m["asm_phase_rmse"].append(phase_rmse_np(ap_rad, tp_rad))

                # ── save visual comparison (first 24) ──────────────────────
                if n_saved < 24:
                    _save_comparison_panel(
                        PATHS.pred_dir / f"{n_saved:04d}_panel.png",
                        holo_np, ta, pa, fa, aa,
                        tp, pp, fp, ap,
                    )
                    _save_gray(PATHS.pred_dir / f"{n_saved:04d}_true_amp.png",    ta)
                    _save_gray(PATHS.pred_dir / f"{n_saved:04d}_unet_amp.png",    pa)
                    _save_gray(PATHS.pred_dir / f"{n_saved:04d}_fresnel_amp.png", fa)
                    _save_gray(PATHS.pred_dir / f"{n_saved:04d}_asm_amp.png",     aa)
                    _save_gray(PATHS.pred_dir / f"{n_saved:04d}_true_phase.png",
                               (tp + 1.0) * 0.5)
                    _save_gray(PATHS.pred_dir / f"{n_saved:04d}_unet_phase.png",
                               (pp + 1.0) * 0.5)
                    _save_gray(PATHS.pred_dir / f"{n_saved:04d}_fresnel_phase.png",
                               (fp + 1.0) * 0.5)
                    _save_gray(PATHS.pred_dir / f"{n_saved:04d}_asm_phase.png",
                               (ap + 1.0) * 0.5)
                    n_saved += 1

    # ── summary ───────────────────────────────────────────────────────────
    summary = {
        "ResUNet_Att": {
            "amp_PSNR_dB":   _mean(m["unet_amp_psnr"]),
            "amp_SSIM":      _mean(m["unet_amp_ssim"]),
            "phase_RMSE_rad": _mean(m["unet_phase_rmse"]),
        },
        "Fresnel_backprop": {
            "amp_PSNR_dB":   _mean(m["fresnel_amp_psnr"]),
            "amp_SSIM":      _mean(m["fresnel_amp_ssim"]),
            "phase_RMSE_rad": _mean(m["fresnel_phase_rmse"]),
        },
        "ASM_backprop": {
            "amp_PSNR_dB":   _mean(m["asm_amp_psnr"]),
            "amp_SSIM":      _mean(m["asm_amp_ssim"]),
            "phase_RMSE_rad": _mean(m["asm_phase_rmse"]),
        },
    }

    # ── print comparison table ────────────────────────────────────────────
    W = 58
    print("\n" + "═" * W)
    print(f"{'Metric':<22} {'ResUNet':>10} {'Fresnel':>10} {'ASM':>10}")
    print("─" * W)
    for label, key, fmt, arrow in [
        ("Amp PSNR (dB) ↑",  "amp_PSNR_dB",    ".3f", "↑"),
        ("Amp SSIM ↑",       "amp_SSIM",        ".4f", "↑"),
        ("Phase RMSE (rad) ↓","phase_RMSE_rad", ".4f", "↓"),
    ]:
        vals = [summary[m][key] for m in ("ResUNet_Att",
                                           "Fresnel_backprop",
                                           "ASM_backprop")]
        print(f"{label:<22} {vals[0]:>10{fmt}} {vals[1]:>10{fmt}} {vals[2]:>10{fmt}}")
    print("═" * W)

    save_json(PATHS.logs_dir / "test_metrics.json", summary)
    print(f"\nResults saved → {PATHS.logs_dir / 'test_metrics.json'}")
    print(f"Visual panels → {PATHS.pred_dir}  ({n_saved} panels)")


if __name__ == "__main__":
    main()
