"""
config.py
─────────
Single source of truth for all paths, physical parameters, and training
hyper-parameters.  Every other module imports from here.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


# ── directory layout ─────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Paths:
    project_root:      Path = Path(__file__).resolve().parents[1]
    data_dir:          Path = project_root / "data"
    sim_holo_dir:      Path = data_dir / "simulated" / "holograms"
    sim_amp_dir:       Path = data_dir / "simulated" / "targets_amp"
    sim_phase_dir:     Path = data_dir / "simulated" / "targets_phase"
    # manifest.jsonl  → one JSON line per sample: {"file": "...", "z_m": <float>}
    # Required by step2 (DataLoader) and step6 (classical baselines) for
    # per-sample propagation distance.
    sim_manifest_path: Path = data_dir / "simulated" / "manifest.jsonl"

    outputs_dir: Path = project_root / "outputs"
    ckpt_dir:    Path = outputs_dir / "checkpoints"
    pred_dir:    Path = outputs_dir / "predictions"
    fig_dir:     Path = outputs_dir / "figures"
    onnx_dir:    Path = outputs_dir / "onnx"
    logs_dir:    Path = outputs_dir / "logs"


# ── training hyper-parameters ────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainConfig:
    seed:        int = 42
    device:      str = "cuda" if torch.cuda.is_available() else "cpu"

    image_size:  int = 256
    num_workers: int = 2

    train_split: float = 0.80
    val_split:   float = 0.10
    test_split:  float = 0.10

    # ── model ────────────────────────────────────────────────────────────────
    in_channels:   int = 1
    out_channels:  int = 2
    base_channels: int = 64   # [64,128,256,512,1024] — ~31 M params with AttGates

    # ── optimiser ────────────────────────────────────────────────────────────
    batch_size:   int   = 8       # ↓ to 4 if GPU OOM on free Kaggle T4
    epochs:       int   = 80
    lr:           float = 2e-4
    weight_decay: float = 1e-5
    grad_clip:    float = 1.0

    # ── composite loss weights ────────────────────────────────────────────────
    # L = α·(MSE_amp + MSE_phase) + β·(1-SSIM_amp) + γ·phase_cosine + δ·L1_grad
    alpha_mse:       float = 1.0
    beta_ssim:       float = 0.3
    gamma_phase_cos: float = 0.5
    delta_grad:      float = 0.1   # spatial-gradient sharpness penalty

    # ── training control ──────────────────────────────────────────────────────
    save_every:          int = 5
    early_stop_patience: int = 12

    # ── physical parameters (ASM simulation + classical baselines) ────────────
    # Green laser, typical sCMOS camera pixel pitch
    wavelength_m:      float = 532e-9    # 532 nm
    pixel_pitch_m:     float = 3.45e-6   # 3.45 µm
    # Propagation distance sampled per sample (Gaussian, clipped at min)
    # Range covers near-field DHM setups (5 mm – ~50 mm)
    propagation_m:     float = 25e-3     # mean 25 mm
    propagation_m_std: float = 10e-3     # std  10 mm
    propagation_m_min: float = 5e-3      # hard floor 5 mm

    # ── Weights & Biases ──────────────────────────────────────────────────────
    # Disable by setting env var:  export WANDB_DISABLED=1
    wandb_project:  str        = "hologram-resunet"
    wandb_run_name: str | None = None     # auto-generated if None


PATHS = Paths()
CFG   = TrainConfig()


def ensure_dirs() -> None:
    """Create all output directories (idempotent)."""
    for p in [
        PATHS.sim_holo_dir,
        PATHS.sim_amp_dir,
        PATHS.sim_phase_dir,
        PATHS.sim_manifest_path.parent,
        PATHS.ckpt_dir,
        PATHS.pred_dir,
        PATHS.fig_dir,
        PATHS.onnx_dir,
        PATHS.logs_dir,
    ]:
        p.mkdir(parents=True, exist_ok=True)


def cfg_as_dict() -> dict:
    """Return CFG as a plain dict (for W&B / JSON logging)."""
    return asdict(CFG)
