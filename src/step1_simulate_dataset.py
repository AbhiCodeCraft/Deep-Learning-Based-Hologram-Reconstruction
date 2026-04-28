"""
step1_simulate_dataset.py
─────────────────────────
Generate a synthetic hologram / amplitude / phase dataset using the
Angular Spectrum Method (ASM).

Physics
───────
Object field at z = 0:
    U₀(x,y) = A(x,y) · exp(j·φ(x,y))

  A(x,y) — sum of random Gaussian blobs (simulates cells / particles)
  φ(x,y) — smooth random phase from superimposed low-frequency modes

Forward propagation to detector at distance z:
    U_z = IFFT[ FFT(U₀) · H_ASM(fx,fy,z) ]

In-line hologram intensity:
    I(x,y) = |U_z|²  + Gaussian noise,  normalised to [0,1]

Ground-truth targets
────────────────────
  amplitude : A(x,y)                  saved as 8-bit PNG ∈ [0,255]
  phase     : (φ + π)/(2π) ∈ [0,1]   saved as 8-bit PNG ∈ [0,255]
              → on load: phase_01 * 2 − 1 maps back to [−1, 1] = [−π, π]/π

manifest.jsonl
──────────────
One JSON line per sample: {"file": "sample_000000.png", "z_m": <float>}
Required by step2_dataset.py (DataLoader z-lookup) and step6_test.py
(classical baseline reconstruction uses the exact per-sample distance).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from src.config import CFG, PATHS, ensure_dirs
from src.propagation import propagate_angular_spectrum
from src.utils import seed_everything


# ── helpers ───────────────────────────────────────────────────────────────────

def to_uint8(x: np.ndarray) -> np.ndarray:
    return np.clip(x * 255.0, 0, 255).astype(np.uint8)


def sample_z(rng: np.random.Generator) -> float:
    """Sample propagation distance (Gaussian, hard floor at min)."""
    z = float(rng.normal(CFG.propagation_m, CFG.propagation_m_std))
    return max(CFG.propagation_m_min, z)


# ── object field synthesis ─────────────────────────────────────────────────────

def make_object_field(
    image_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (amplitude, phase) for a synthetic object.

    Amplitude
    ─────────
    Sum of 4–12 random Gaussian blobs at random positions, widths, and
    peak intensities.  This mimics bio-cells, particles, or fibers —
    sparse and localised rather than a global pattern.

    Phase
    ─────
    Superposition of 4–10 low-frequency sinusoidal modes at random
    orientations and spatial frequencies.  Produces a smooth, wrapping
    phase map that is representative of optical path-length variations
    through a thin sample.

    Returns
    ───────
    amplitude : float32 (H, W) ∈ [0, 1]
    phase     : float32 (H, W) ∈ [−π, π]
    """
    # ── amplitude: Gaussian blobs ────────────────────────────────────────────
    amp = np.zeros((image_size, image_size), dtype=np.float64)
    n_blobs = int(rng.integers(4, 13))
    for _ in range(n_blobs):
        cx    = rng.uniform(0.05, 0.95) * image_size
        cy    = rng.uniform(0.05, 0.95) * image_size
        sigma = rng.uniform(0.02, 0.12) * image_size
        peak  = rng.uniform(0.3, 1.0)
        yy, xx = np.ogrid[:image_size, :image_size]
        amp += peak * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    amp = np.clip(amp, 0.0, 1.0).astype(np.float32)

    # ── phase: smooth low-frequency modes ────────────────────────────────────
    yy, xx = np.meshgrid(
        np.linspace(-1.0, 1.0, image_size, dtype=np.float32),
        np.linspace(-1.0, 1.0, image_size, dtype=np.float32),
        indexing="ij",
    )
    phase = np.zeros((image_size, image_size), dtype=np.float64)
    n_modes = int(rng.integers(4, 11))
    for _ in range(n_modes):
        fx_m   = rng.uniform(0.5, 4.5)
        fy_m   = rng.uniform(0.5, 4.5)
        angle  = rng.uniform(0.0, np.pi)
        weight = rng.uniform(-1.0, 1.0)
        ca, sa = np.cos(angle), np.sin(angle)
        xr     = ca * xx - sa * yy
        yr     = sa * xx + ca * yy
        phase += weight * np.sin(np.pi * fx_m * xr + np.pi * fy_m * yr)

    # Normalise smoothly to [−π, π]
    max_abs = np.abs(phase).max() + 1e-8
    phase   = np.clip(phase / max_abs * np.pi, -np.pi, np.pi).astype(np.float32)
    return amp, phase


# ── main ──────────────────────────────────────────────────────────────────────

def main(num_samples: int = 15000, image_size: int = 256, seed: int = 42) -> None:
    ensure_dirs()
    seed_everything(seed)
    rng = np.random.default_rng(seed)

    lam   = CFG.wavelength_m
    pitch = CFG.pixel_pitch_m

    # Re-create manifest from scratch
    PATHS.sim_manifest_path.unlink(missing_ok=True)

    print(f"Generating {num_samples} ASM hologram samples …")
    print(f"  λ = {lam*1e9:.0f} nm,  pixel pitch = {pitch*1e6:.2f} µm")

    with open(PATHS.sim_manifest_path, "a", encoding="utf-8") as mf:
        for i in range(num_samples):
            # ── object field ──────────────────────────────────────────────
            amp, phase = make_object_field(image_size, rng)

            # ── forward ASM propagation ───────────────────────────────────
            z_m = sample_z(rng)
            U0  = amp * np.exp(1j * phase.astype(np.float64))
            Uz  = propagate_angular_spectrum(
                U0.astype(np.complex128), z_m, lam, pitch
            )

            # ── in-line hologram intensity ─────────────────────────────────
            intensity = np.abs(Uz) ** 2
            intensity = (intensity - intensity.min()) / (
                intensity.max() - intensity.min() + 1e-8
            )
            intensity += rng.normal(0.0, 0.02, intensity.shape)
            intensity = np.clip(intensity, 0.0, 1.0).astype(np.float32)

            # ── save images ────────────────────────────────────────────────
            name = f"sample_{i:06d}.png"
            Image.fromarray(to_uint8(intensity), mode="L").save(
                PATHS.sim_holo_dir / name
            )
            Image.fromarray(to_uint8(amp), mode="L").save(
                PATHS.sim_amp_dir / name
            )
            # Phase stored as φ_01 = (φ + π) / (2π) ∈ [0,1]
            phase_01 = (phase + np.pi) / (2.0 * np.pi)
            Image.fromarray(to_uint8(phase_01), mode="L").save(
                PATHS.sim_phase_dir / name
            )

            # ── manifest entry ─────────────────────────────────────────────
            mf.write(json.dumps({"file": name, "z_m": float(z_m)}) + "\n")

            if (i + 1) % 1000 == 0:
                print(f"  [{i+1:>6}/{num_samples}]  z = {z_m*1e3:6.2f} mm")

    print(f"\nDone. {num_samples} samples → {PATHS.data_dir}")
    print(f"Manifest → {PATHS.sim_manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simulate hologram dataset via Angular Spectrum Method"
    )
    parser.add_argument("--num-samples", type=int, default=15000)
    parser.add_argument("--image-size",  type=int, default=256)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()
    main(
        num_samples=args.num_samples,
        image_size=args.image_size,
        seed=args.seed,
    )
