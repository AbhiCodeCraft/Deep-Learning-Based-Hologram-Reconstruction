"""
propagation.py
──────────────
Classical wave-optics propagation module used in two places:

  1. step1_simulate_dataset.py  — forward ASM to synthesise holograms.
  2. step6_test.py              — inverse Fresnel and inverse ASM as the
                                  classical baselines that the ResUNet is
                                  benchmarked against.

Two transfer functions
──────────────────────
  Angular Spectrum Method (ASM):
      H_ASM(fx,fy) = exp(j·kz·z)   where  kz = (2π/λ)·√(1−(λfx)²−(λfy)²)
      Evanescent components (sin²>1) are zeroed by the propagating mask.
      This is the exact Helmholtz solution — no paraxial approximation.

  Fresnel (paraxial):
      H_Fres(fx,fy) = exp(j·2πz/λ) · exp(−jπλz·(fx²+fy²))
      Valid when the propagation angle is small.  Used as the simpler
      classical baseline in step6.

Reconstruction from intensity-only hologram
────────────────────────────────────────────
  In-line holography records I = |U|².  A common starting estimate for
  back-propagation is the weak-object approximation:
      U_h ≈ √I · exp(j·0)
  This introduces the well-known twin-image artefact — precisely the
  limitation that the ResUNet is trained to overcome.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np


# ── internal helpers ─────────────────────────────────────────────────────────

def _freq_grid(ny: int, nx: int, dy: float, dx: float) -> Tuple[np.ndarray, np.ndarray]:
    """Spatial-frequency grids (cycles/m) in (ny, nx) FFT axis order."""
    fy = np.fft.fftfreq(ny, d=dy).astype(np.float64)
    fx = np.fft.fftfreq(nx, d=dx).astype(np.float64)
    FY, FX = np.meshgrid(fy, fx, indexing="ij")
    return FX, FY


# ── transfer functions ────────────────────────────────────────────────────────

def asm_transfer_function(
    ny: int, nx: int,
    dy: float, dx: float,
    wavelength_m: float,
    z_m: float,
) -> np.ndarray:
    """
    Angular Spectrum Method transfer function for propagation distance z_m.

    Returns complex128 array of shape (ny, nx).
    Negative z_m = back-propagation.
    """
    k  = 2.0 * np.pi / wavelength_m
    FX, FY = _freq_grid(ny, nx, dy, dx)
    lam = wavelength_m
    inside = np.maximum(1.0 - (lam * FX) ** 2 - (lam * FY) ** 2, 0.0)
    kz = k * np.sqrt(inside)
    return np.exp(1j * kz * z_m).astype(np.complex128)


def fresnel_transfer_function(
    ny: int, nx: int,
    dy: float, dx: float,
    wavelength_m: float,
    z_m: float,
) -> np.ndarray:
    """
    Paraxial Fresnel transfer function for propagation distance z_m.

    Returns complex128 array of shape (ny, nx).
    Negative z_m = back-propagation.
    """
    FX, FY = _freq_grid(ny, nx, dy, dx)
    lam = wavelength_m
    H = (
        np.exp(1j * 2.0 * np.pi * z_m / lam)
        * np.exp(-1j * np.pi * lam * z_m * (FX ** 2 + FY ** 2))
    )
    return H.astype(np.complex128)


# ── forward propagation ───────────────────────────────────────────────────────

def propagate_angular_spectrum(
    field: np.ndarray,
    z_m: float,
    wavelength_m: float,
    pixel_pitch_m: float,
) -> np.ndarray:
    """Forward-propagate complex scalar field (ny, nx) by z_m using ASM."""
    ny, nx = field.shape
    H = asm_transfer_function(ny, nx, pixel_pitch_m, pixel_pitch_m,
                               wavelength_m, z_m)
    return np.fft.ifft2(np.fft.fft2(field.astype(np.complex128)) * H)


# ── back-propagation (inverse) ────────────────────────────────────────────────

def backprop_angular_spectrum(
    field: np.ndarray,
    z_m: float,
    wavelength_m: float,
    pixel_pitch_m: float,
) -> np.ndarray:
    """
    Inverse ASM back-propagation.
    Multiplies the spectrum by conj(H_ASM(z)) which equals H_ASM(−z).
    """
    ny, nx = field.shape
    H = asm_transfer_function(ny, nx, pixel_pitch_m, pixel_pitch_m,
                               wavelength_m, z_m)
    return np.fft.ifft2(np.fft.fft2(field.astype(np.complex128)) * np.conj(H))


def backprop_fresnel(
    field: np.ndarray,
    z_m: float,
    wavelength_m: float,
    pixel_pitch_m: float,
) -> np.ndarray:
    """
    Inverse Fresnel (paraxial) back-propagation.
    Multiplies the spectrum by conj(H_Fresnel(z)).
    """
    ny, nx = field.shape
    H = fresnel_transfer_function(ny, nx, pixel_pitch_m, pixel_pitch_m,
                                   wavelength_m, z_m)
    return np.fft.ifft2(np.fft.fft2(field.astype(np.complex128)) * np.conj(H))


# ── hologram-plane initialisation ─────────────────────────────────────────────

def intensity_to_complex_field(intensity_01: np.ndarray) -> np.ndarray:
    """
    Weak-object / sqrt-intensity estimate of the hologram-plane field:
        U_h = √I · exp(j·0)
    This is the standard starting point for intensity-only reconstruction.
    """
    amp = np.sqrt(np.clip(intensity_01.astype(np.float64), 0.0, 1.0))
    return amp.astype(np.complex128)


def _normalise_amp_phase(field: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract amplitude ∈ [0,1] (min-max normalised) and phase ∈ [−1,1]
    (wrapped angle divided by π) from a complex field.
    """
    amp   = np.abs(field).astype(np.float64)
    phase = np.angle(field).astype(np.float64)          # ∈ [−π, π]
    phase_n = np.clip(phase / np.pi, -1.0, 1.0)         # ∈ [−1, 1]
    amp   = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)
    return amp.astype(np.float32), phase_n.astype(np.float32)


# ── public reconstruction functions (used in step6_test.py) ──────────────────

def fresnel_reconstruct(
    hologram_intensity_01: np.ndarray,
    z_m: float,
    wavelength_m: float,
    pixel_pitch_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classical Fresnel (paraxial) back-propagation baseline.

    Steps:
      1. U_h = √I · exp(j·0)
      2. Fresnel back-propagate by –z  →  Û_obj
      3. Return (amplitude, phase) in normalised form comparable to targets.
    """
    U_h   = intensity_to_complex_field(hologram_intensity_01)
    U_obj = backprop_fresnel(U_h, z_m, wavelength_m, pixel_pitch_m)
    return _normalise_amp_phase(U_obj)


def asm_reconstruct(
    hologram_intensity_01: np.ndarray,
    z_m: float,
    wavelength_m: float,
    pixel_pitch_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classical Angular Spectrum back-propagation baseline.

    Same √I initialisation as Fresnel but uses the exact ASM inverse.
    """
    U_h   = intensity_to_complex_field(hologram_intensity_01)
    U_obj = backprop_angular_spectrum(U_h, z_m, wavelength_m, pixel_pitch_m)
    return _normalise_amp_phase(U_obj)
