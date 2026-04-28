"""
step5_train.py
──────────────
Training loop for the Res-AttU-Net.

Features
────────
• AdamW + CosineAnnealingLR
• Mixed-precision AMP (torch.amp)
• Gradient clipping
• Early stopping
• Best-model + periodic epoch checkpoints
• Optional Weights & Biases experiment tracking
    Disable:  export WANDB_DISABLED=1   (before running)
    Login:    wandb login               (first time only)

Run
───
    python -m src.step5_train
"""
from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.config import CFG, PATHS, cfg_as_dict, ensure_dirs
from src.step2_dataset import collect_triplets, make_dataloaders
from src.step3_unet_model import ResUNet
from src.step4_loss_metrics import composite_loss
from src.utils import AverageMeter, save_json, seed_everything


# ── W&B helper ────────────────────────────────────────────────────────────────

def _init_wandb():
    """Initialise W&B. Returns run object or None if disabled/unavailable."""
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true", "yes"):
        print("[W&B] disabled via environment variable.")
        return None
    try:
        import wandb
        run = wandb.init(
            project=CFG.wandb_project,
            name=CFG.wandb_run_name,
            config=cfg_as_dict(),
        )
        print(f"[W&B] run: {run.url}")
        return run
    except Exception as exc:
        print(f"[W&B] not available ({exc}) — training without logging.")
        return None


# ── validation helper ─────────────────────────────────────────────────────────

def evaluate(
    model:  torch.nn.Module,
    loader,
    device: torch.device,
) -> dict:
    """Return average loss components over the loader."""
    model.eval()
    meters = {k: AverageMeter()
              for k in ("total", "mse_amp", "mse_phase", "ssim", "phase_cos", "grad")}
    with torch.no_grad():
        for x, y, _z in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            pred = model(x)
            _, parts = composite_loss(
                pred, y,
                alpha_mse=CFG.alpha_mse,
                beta_ssim=CFG.beta_ssim,
                gamma_phase_cos=CFG.gamma_phase_cos,
                delta_grad=CFG.delta_grad,
            )
            n = x.size(0)
            for k in meters:
                meters[k].update(parts.get(k, parts["total"]), n)
    return {k: m.avg for k, m in meters.items()}


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ensure_dirs()
    seed_everything(CFG.seed)

    device = torch.device(CFG.device)
    print(f"Device : {device}")

    # ── data ──────────────────────────────────────────────────────────────
    samples = collect_triplets(
        PATHS.sim_holo_dir, PATHS.sim_amp_dir, PATHS.sim_phase_dir
    )
    if len(samples) < 100:
        raise RuntimeError(
            f"Only {len(samples)} samples found. "
            "Run step1_simulate_dataset.py first (need ≥ 100)."
        )
    print(f"Dataset: {len(samples)} samples  (80 / 10 / 10 split)")

    train_loader, val_loader, _ = make_dataloaders(samples)

    # ── model ─────────────────────────────────────────────────────────────
    model = ResUNet(
        in_channels=CFG.in_channels,
        out_channels=CFG.out_channels,
        base_channels=CFG.base_channels,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model  : Res-AttU-Net  |  params = {n_params:,}")

    # ── optimiser + scheduler ─────────────────────────────────────────────
    optimizer = AdamW(model.parameters(),
                      lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)
    scaler    = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    wb = _init_wandb()
    if wb is not None:
        try:
            import wandb
            wb.watch(model, log="gradients", log_freq=100)
        except Exception:
            pass

    # ── training state ────────────────────────────────────────────────────
    best_val = float("inf")
    wait     = 0
    history  = {
        "train_loss": [], "val_loss": [],
        "val_ssim": [], "val_phase_cos": [],
    }

    for epoch in range(1, CFG.epochs + 1):
        # ── train ─────────────────────────────────────────────────────────
        model.train()
        train_meter = AverageMeter()

        pbar = tqdm(train_loader,
                    desc=f"Epoch {epoch:03d}/{CFG.epochs}", leave=False)
        for x, y, _z in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,
                                 enabled=(device.type == "cuda")):
                pred       = model(x)
                loss, parts = composite_loss(
                    pred, y,
                    alpha_mse=CFG.alpha_mse,
                    beta_ssim=CFG.beta_ssim,
                    gamma_phase_cos=CFG.gamma_phase_cos,
                    delta_grad=CFG.delta_grad,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_meter.update(loss.item(), x.size(0))
            pbar.set_postfix(
                loss=f"{train_meter.avg:.4f}",
                mse=f"{parts['mse_amp']:.4f}",
            )

        # ── validate ──────────────────────────────────────────────────────
        val_metrics = evaluate(model, val_loader, device)
        val_loss    = val_metrics["total"]
        scheduler.step()
        lr_now = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_meter.avg)
        history["val_loss"].append(val_loss)
        history["val_ssim"].append(val_metrics.get("ssim", 0.0))
        history["val_phase_cos"].append(val_metrics.get("phase_cos", 0.0))

        print(
            f"Epoch {epoch:03d} | "
            f"train={train_meter.avg:.5f} | "
            f"val={val_loss:.5f} | "
            f"ssim={val_metrics.get('ssim', 0):.4f} | "
            f"pcos={val_metrics.get('phase_cos', 0):.4f} | "
            f"lr={lr_now:.2e}"
        )

        # ── W&B log ───────────────────────────────────────────────────────
        if wb is not None:
            try:
                wb.log(
                    {
                        "epoch":          epoch,
                        "train/loss":     train_meter.avg,
                        "val/loss":       val_loss,
                        "val/ssim":       val_metrics.get("ssim", 0.0),
                        "val/phase_cos":  val_metrics.get("phase_cos", 0.0),
                        "val/mse_amp":    val_metrics.get("mse_amp", 0.0),
                        "val/mse_phase":  val_metrics.get("mse_phase", 0.0),
                        "val/grad":       val_metrics.get("grad", 0.0),
                        "lr":             lr_now,
                    },
                    step=epoch,
                )
            except Exception:
                pass

        # ── checkpoint: best ──────────────────────────────────────────────
        if val_loss < best_val:
            best_val = val_loss
            wait     = 0
            torch.save(
                {
                    "epoch":               epoch,
                    "model_state_dict":    model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss":            val_loss,
                    "cfg":                 cfg_as_dict(),
                },
                PATHS.ckpt_dir / "best_model.pt",
            )
            print(f"  ✓ best_model.pt  (val={best_val:.6f})")
        else:
            wait += 1

        # ── checkpoint: periodic ──────────────────────────────────────────
        if epoch % CFG.save_every == 0:
            torch.save(
                {
                    "epoch":            epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss":         val_loss,
                    "cfg":              cfg_as_dict(),
                },
                PATHS.ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

        # ── early stopping ────────────────────────────────────────────────
        if wait >= CFG.early_stop_patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    save_json(PATHS.logs_dir / "train_history.json", history)
    print(f"\nTraining complete.  Best val loss = {best_val:.6f}")
    if wb is not None:
        try:
            wb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    main()
