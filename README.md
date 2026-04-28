# Deep Learning-Based Single Hologram Reconstruction
### Residual U-Net with Attention Gates (Res-AttU-Net)
**B.Tech Final Year Project — Electrical Engineering, NIT Delhi**

---

## Project Summary

Classical digital holography reconstructs an object field by
back-propagating the recorded hologram intensity using the Fresnel or
Angular Spectrum transfer function.  This approach is parameter-sensitive
and inherently limited by the twin-image artefact in in-line geometry.

This project replaces the classical pipeline with a Res-AttU-Net trained
end-to-end to map a single hologram directly to the object amplitude
and phase.  The trained model is benchmarked against **two** classical
baselines (Fresnel and ASM back-propagation) on PSNR, SSIM, and Phase RMSE.

---

## Architecture: Res-AttU-Net

```
Input [1×256×256]
      │
 ┌────┴────────────────────────────┐
 │  Encoder  (ResBlock + MaxPool)  │  e1…e4
 │  64 → 128 → 256 → 512 channels  │
 └────┬────────────────────────────┘
      │
  Bottleneck  ResBlock  1024 ch
      │
 ┌────┴────────────────────────────┐
 │  Decoder  (ConvT → AttGate ← e  │
 │            → concat → ResBlock) │
 │  512 → 256 → 128 → 64 channels  │
 └────┬────────────────────────────┘
      │
   Head: 1×1 Conv  [2 ch]
   ch0 = amplitude   (clamp → [0,1])
   ch1 = phase logit (tanh  → [−1,1])
```

**ResidualBlock** — additive skip connections prevent vanishing gradients
(He et al., Deep Residual Learning, CVPR 2016).

**AttentionGate** — per-pixel soft weighting of encoder skip features;
suppresses irrelevant background and focuses reconstruction on the object
(Oktay et al., Attention U-Net, MIDL 2018).

~31 M trainable parameters with base_channels = 64.

---

## Composite Loss

```
L = α·(MSE_amp + MSE_phase)    α = 1.0   pixel-level fidelity
  + β·(1 − SSIM_amp)            β = 0.3   structural quality
  + γ·mean(1−cos(Δφ))           γ = 0.5   rotation-invariant phase
  + δ·L1_gradient_amp           δ = 0.1   edge sharpness
```

The phase cosine term handles the circular / wrapping nature of phase
without requiring phase unwrapping during training.

---

## Dataset (Angular Spectrum Method)

15,000 synthetic (hologram, amplitude, phase) triplets.

| Parameter          | Value                     |
|--------------------|---------------------------|
| Wavelength         | 532 nm (green laser)      |
| Pixel pitch        | 3.45 µm                   |
| Propagation z      | Gaussian(25 mm, 10 mm std)|
| Image size         | 256 × 256 px              |
| Train / Val / Test | 80 / 10 / 10 %            |

Object fields use random Gaussian blob amplitudes (4–12 blobs)
and smooth random phase maps (4–10 low-frequency modes).
Each sample has its exact z stored in `manifest.jsonl` — used by
both the DataLoader and step6 for per-sample classical reconstruction.

---

## Run Order (Kaggle)

```bash
# Cell 1 — install dependencies
!pip install -r Hologram/requirements.txt

# Cell 2 — generate dataset
!cd Hologram && python -m src.step1_simulate_dataset --num-samples 15000

# Cell 3 — train (disable W&B on Kaggle free tier)
!cd Hologram && WANDB_DISABLED=1 python -m src.step5_train

# Cell 4 — evaluate (ResUNet vs Fresnel vs ASM)
!cd Hologram && python -m src.step6_test

# Cell 5 — (optional) Gradio demo
!cd Hologram && python -m src.step7_deploy --mode gradio
```

---

## Expected Results

| Metric              | Res-AttU-Net | Fresnel | ASM   |
|---------------------|:------------:|:-------:|:-----:|
| Amp PSNR (dB)  ↑   |  ~28–32      | ~18–22  | ~19–23|
| Amp SSIM       ↑   |  ~0.85–0.92  | ~0.55–0.70 | ~0.58–0.72 |
| Phase RMSE (rad) ↓ |  ~0.15–0.25  | ~0.45–0.80 | ~0.40–0.75 |

---

## File Structure

```
Hologram/
├── requirements.txt
├── README.md
└── src/
    ├── config.py                  — all hyperparameters, paths, physics
    ├── propagation.py             — ASM + Fresnel forward/inverse physics
    ├── step1_simulate_dataset.py  — dataset generation (ASM + Gaussian blobs)
    ├── step2_dataset.py           — Dataset, augmentation, DataLoaders, z-lookup
    ├── step3_unet_model.py        — Res-AttU-Net (ResidualBlock + AttentionGate)
    ├── step4_loss_metrics.py      — composite loss + numpy eval metrics
    ├── step5_train.py             — training loop + W&B logging
    ├── step6_test.py              — evaluation: ResUNet vs Fresnel vs ASM
    ├── step7_deploy.py            — Gradio web app + ONNX export
    └── utils.py                   — seed, AverageMeter, JSON helpers
```

---

## Key Differentiators

| Feature                    | This project | Typical reference repos  |
|----------------------------|:------------:|:------------------------:|
| Unified 2-channel output   | ✅           | Two separate models       |
| Attention Gates            | ✅           | Plain skip connections    |
| Residual blocks            | ✅           | Standard double-conv      |
| Two classical baselines    | ✅           | None or one               |
| Per-sample z in manifest   | ✅           | Fixed z or CSV lookup     |
| Train-time augmentation    | ✅           | None                      |
| Gradient sharpness loss    | ✅           | MSE only                  |
| W&B experiment tracking    | ✅           | None                      |
| ONNX export                | ✅           | None                      |
