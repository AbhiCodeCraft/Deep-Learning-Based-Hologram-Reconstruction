"""
step2_dataset.py
────────────────
PyTorch Dataset + DataLoaders for hologram reconstruction.

Key features
────────────
• Per-sample propagation distance z loaded from manifest.jsonl and
  returned as a third element — (x, y, z).  z is needed by step6_test.py
  for classical back-propagation at the exact recorded distance.

• Training split receives full augmentation; val/test splits receive none.

Augmentation (training only)
─────────────────────────────
Spatial transforms are applied IDENTICALLY to both the hologram input
and the amplitude/phase targets so that spatial correspondence is
preserved.  Photometric transforms are applied to the hologram ONLY,
simulating varying illumination and sensor conditions.

  Spatial  : random horizontal flip, vertical flip, small rotation [±12°]
  Photometric (hologram only):
    — Gaussian additive sensor noise  σ ∈ [0.005, 0.030]
    — Random intensity scale          s ∈ [0.85, 1.15]   (illumination jitter)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import InterpolationMode

from src.config import CFG, PATHS


# ── I/O helpers ───────────────────────────────────────────────────────────────

def _read_grayscale(path: Path, image_size: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    img = Image.open(path).convert("L").resize(
        (image_size, image_size), Image.BILINEAR
    )
    return np.asarray(img, dtype=np.float32) / 255.0


def collect_triplets(
    holo_dir: Path,
    amp_dir: Path,
    phase_dir: Path,
) -> List[Tuple[Path, Path, Path]]:
    """Return sorted list of matched (hologram, amplitude, phase) path triplets."""
    holo_map  = {p.name: p for p in holo_dir.glob("*")  if p.is_file()}
    amp_map   = {p.name: p for p in amp_dir.glob("*")   if p.is_file()}
    phase_map = {p.name: p for p in phase_dir.glob("*") if p.is_file()}
    common    = sorted(set(holo_map) & set(amp_map) & set(phase_map))
    return [(holo_map[n], amp_map[n], phase_map[n]) for n in common]


def load_z_manifest(manifest_path: Path) -> Dict[str, float]:
    """
    Load per-sample propagation distances from manifest.jsonl.
    Returns {filename: z_metres}.  Empty dict if file missing.
    """
    if not manifest_path.exists():
        return {}
    out: Dict[str, float] = {}
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            out[str(rec["file"])] = float(rec["z_m"])
    return out


# ── augmentation ─────────────────────────────────────────────────────────────

def _spatial_augment(
    holo: torch.Tensor,   # [1, H, W]
    y:    torch.Tensor,   # [2, H, W]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply identical geometric transforms to hologram and targets."""
    # Horizontal flip
    if torch.rand(()) < 0.5:
        holo = torch.flip(holo, dims=(-1,))
        y    = torch.flip(y,    dims=(-1,))
    # Vertical flip
    if torch.rand(()) < 0.5:
        holo = torch.flip(holo, dims=(-2,))
        y    = torch.flip(y,    dims=(-2,))
    # Small rotation
    angle = float(torch.empty(()).uniform_(-12.0, 12.0).item())
    if abs(angle) > 0.5:
        stacked = torch.cat([holo, y], dim=0)   # [3, H, W]
        stacked = TF.rotate(
            stacked, angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
        holo = stacked[0:1]
        y    = stacked[1:3]
    return holo, y


def _photometric_augment(holo: torch.Tensor) -> torch.Tensor:
    """Apply photometric jitter to the hologram only."""
    # Gaussian sensor noise
    if torch.rand(()) < 0.5:
        std   = float(torch.empty(()).uniform_(0.005, 0.030).item())
        holo  = torch.clamp(holo + torch.randn_like(holo) * std, 0.0, 1.0)
    # Illumination intensity jitter
    if torch.rand(()) < 0.5:
        scale = float(torch.empty(()).uniform_(0.85, 1.15).item())
        holo  = torch.clamp(holo * scale, 0.0, 1.0)
    return holo


# ── dataset ───────────────────────────────────────────────────────────────────

class HologramDataset(Dataset):
    """
    Returns (x, y, z):
        x : [1, H, W]  hologram intensity,  float32 ∈ [0, 1]
        y : [2, H, W]  targets
                         ch0 = amplitude ∈ [0, 1]
                         ch1 = phase     ∈ [−1, 1]   (= original phase / π)
        z : [1]        propagation distance in metres (float32)

    Phase encoding:
      On disk:  φ_01 = (φ + π) / (2π) ∈ [0, 1]
      In tensor: phase = φ_01 * 2 − 1  ∈ [−1, 1]   (= φ / π)
    """

    def __init__(
        self,
        samples:    Sequence[Tuple[Path, Path, Path]],
        image_size: int = 256,
        augment:    bool = False,
        z_lookup:   Dict[str, float] | None = None,
    ):
        self.samples    = list(samples)
        self.image_size = image_size
        self.augment    = augment
        self.z_lookup   = z_lookup or {}
        self.default_z  = float(CFG.propagation_m)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        holo_p, amp_p, phase_p = self.samples[idx]

        holo     = _read_grayscale(holo_p,  self.image_size)
        amp      = _read_grayscale(amp_p,   self.image_size)
        phase_01 = _read_grayscale(phase_p, self.image_size)
        phase    = phase_01 * 2.0 - 1.0          # [0,1] → [−1,1]

        x = torch.from_numpy(holo).unsqueeze(0)           # [1,H,W]
        y = torch.cat(
            [
                torch.from_numpy(amp).unsqueeze(0),        # ch0
                torch.from_numpy(phase).unsqueeze(0),      # ch1
            ],
            dim=0,
        )                                                   # [2,H,W]

        if self.augment:
            x, y = _spatial_augment(x, y)
            x    = _photometric_augment(x)

        z_m = self.z_lookup.get(holo_p.name, self.default_z)
        z   = torch.tensor([z_m], dtype=torch.float32)    # [1]
        return x, y, z


# ── dataloader factory ────────────────────────────────────────────────────────

def make_dataloaders(
    samples: Sequence[Tuple[Path, Path, Path]],
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Deterministically split samples 80/10/10 and return
    (train_loader, val_loader, test_loader).
    Augmentation is enabled for the training loader only.
    """
    n_total = len(samples)
    n_train = int(n_total * CFG.train_split)
    n_val   = int(n_total * CFG.val_split)

    g    = torch.Generator().manual_seed(CFG.seed)
    perm = torch.randperm(n_total, generator=g).tolist()

    train_s = [samples[i] for i in perm[:n_train]]
    val_s   = [samples[i] for i in perm[n_train: n_train + n_val]]
    test_s  = [samples[i] for i in perm[n_train + n_val:]]

    z_lookup = load_z_manifest(PATHS.sim_manifest_path)

    def _ds(split, aug):
        return HologramDataset(split, image_size=CFG.image_size,
                                augment=aug, z_lookup=z_lookup)

    def _loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=CFG.batch_size,
            shuffle=shuffle,
            num_workers=CFG.num_workers,
            pin_memory=True,
            persistent_workers=(CFG.num_workers > 0),
        )

    return (
        _loader(_ds(train_s, True),  True),
        _loader(_ds(val_s,   False), False),
        _loader(_ds(test_s,  False), False),
    )
