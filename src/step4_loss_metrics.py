"""
step4_loss_metrics.py
──────────────────────
Training loss function and evaluation metrics.

Composite Loss
──────────────
  L = α · (MSE_amp + MSE_phase)
    + β · (1 − SSIM_amp)
    + γ · mean(1 − cos(φ_pred − φ_true))
    + δ · L1_gradient_amp

  Term              Weight   Purpose
  ────────────────  ───────  ─────────────────────────────────────────────
  MSE (amp+phase)   α = 1.0  Pixel-level fidelity for both outputs
  1 − SSIM_amp      β = 0.3  Structural / perceptual quality of amplitude
  Phase cosine      γ = 0.5  Rotation-invariant phase error; handles wrapping
  L1 gradient       δ = 0.1  Preserves fine edges and sharp boundaries

Phase cosine loss:
  Converts phase from [−1,1] → [−π,π], then computes
  mean(1 − cos(φ_pred − φ_true)).  This is invariant to global phase
  offsets and handles the circular nature of phase correctly.

Numpy metrics (used in step6_test.py)
──────────────────────────────────────
  psnr_np        Peak Signal-to-Noise Ratio (dB) on amplitude
  ssim_np        Structural Similarity on amplitude
  phase_rmse_np  RMS phase error (radians), computed on wrapped difference
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── individual loss components ────────────────────────────────────────────────

def ssim_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
    window: int = 3,
) -> torch.Tensor:
    """Differentiable 1 − SSIM using a uniform pooling window."""
    p  = window // 2
    mu_x  = F.avg_pool2d(pred,   window, stride=1, padding=p)
    mu_y  = F.avg_pool2d(target, window, stride=1, padding=p)
    sx    = F.avg_pool2d(pred   * pred,   window, 1, p) - mu_x * mu_x
    sy    = F.avg_pool2d(target * target, window, 1, p) - mu_y * mu_y
    sxy   = F.avg_pool2d(pred   * target, window, 1, p) - mu_x * mu_y
    num   = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    denom = (mu_x ** 2 + mu_y ** 2 + c1) * (sx + sy + c2) + 1e-8
    return 1.0 - (num / denom).mean()


def phase_cosine_loss(
    pred_phase: torch.Tensor,
    true_phase: torch.Tensor,
) -> torch.Tensor:
    """
    Rotation-invariant phase loss.
    Both tensors expected in [−1, 1]  (= phase / π).
    """
    pred = pred_phase * torch.pi
    true = true_phase * torch.pi
    return (1.0 - torch.cos(pred - true)).mean()


def gradient_loss(
    pred:   torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """
    L1 loss on spatial finite differences.
    Encourages the model to reproduce sharp edges and fine structures.
    """
    gx_p = pred[:, :, :, 1:]   - pred[:, :, :, :-1]
    gy_p = pred[:, :, 1:, :]   - pred[:, :, :-1, :]
    gx_t = target[:, :, :, 1:] - target[:, :, :, :-1]
    gy_t = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


# ── composite loss ────────────────────────────────────────────────────────────

def composite_loss(
    pred:            torch.Tensor,   # [B, 2, H, W]
    target:          torch.Tensor,   # [B, 2, H, W]
    alpha_mse:       float = 1.0,
    beta_ssim:       float = 0.3,
    gamma_phase_cos: float = 0.5,
    delta_grad:      float = 0.1,
) -> tuple[torch.Tensor, dict]:
    """
    Parameters
    ──────────
    pred   : ch0 = raw amplitude logits, ch1 = raw phase logits
    target : ch0 = amplitude ∈ [0,1],    ch1 = phase ∈ [−1,1]

    Returns (total_loss, parts_dict)
    """
    pred_amp   = pred[:, 0:1].clamp(0.0, 1.0)
    pred_phase = torch.tanh(pred[:, 1:2])          # ∈ [−1, 1]

    true_amp   = target[:, 0:1]
    true_phase = target[:, 1:2]

    l_mse_amp   = F.mse_loss(pred_amp,   true_amp)
    l_mse_phase = F.mse_loss(pred_phase, true_phase)
    l_ssim      = ssim_loss(pred_amp, true_amp.clamp(0.0, 1.0))
    l_pcos      = phase_cosine_loss(pred_phase, true_phase)
    l_grad      = gradient_loss(pred_amp, true_amp)

    total = (
        alpha_mse       * (l_mse_amp + l_mse_phase)
        + beta_ssim     * l_ssim
        + gamma_phase_cos * l_pcos
        + delta_grad    * l_grad
    )

    parts = {
        "mse_amp":   l_mse_amp.item(),
        "mse_phase": l_mse_phase.item(),
        "ssim":      l_ssim.item(),
        "phase_cos": l_pcos.item(),
        "grad":      l_grad.item(),
        "total":     total.item(),
    }
    return total, parts


# ── numpy evaluation metrics (used in step6_test.py) ─────────────────────────

def psnr_np(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio in dB. Returns 99.0 if MSE < 1e-12."""
    mse = float(np.mean((pred - target) ** 2))
    if mse < 1e-12:
        return 99.0
    return float(20.0 * np.log10(max_val / np.sqrt(mse)))


def ssim_np(
    pred:   np.ndarray,
    target: np.ndarray,
    c1: float = 0.01 ** 2,
    c2: float = 0.03 ** 2,
) -> float:
    """Global SSIM (single-window approximation, fast)."""
    mx, my = pred.mean(), target.mean()
    vx, vy = pred.var(), target.var()
    cov    = float(((pred - mx) * (target - my)).mean())
    num    = (2 * mx * my + c1) * (2 * cov + c2)
    denom  = (mx ** 2 + my ** 2 + c1) * (vx + vy + c2) + 1e-12
    return float(num / denom)


def phase_rmse_np(
    pred_phase_rad:  np.ndarray,
    true_phase_rad:  np.ndarray,
) -> float:
    """
    Root-mean-square phase error in radians.
    Uses np.angle(exp(j·Δφ)) to handle phase wrapping correctly.
    """
    diff = np.angle(np.exp(1j * (pred_phase_rad - true_phase_rad)))
    return float(np.sqrt(np.mean(diff ** 2)))
