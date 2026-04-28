"""
step3_unet_model.py
────────────────────
Residual U-Net with Attention Gates  (Res-AttU-Net).

Architecture
────────────
                     Input [1 × 256 × 256]
                           │
              ┌────────────┼────────────┐
              │       Encoder           │
              │  ResBlock → MaxPool ×4  │ ← skip e1…e4
              │  (64→128→256→512 ch)    │
              └────────────┬────────────┘
                           │
                     Bottleneck
                    ResBlock 1024 ch
                           │
              ┌────────────┼────────────┐
              │       Decoder           │
              │  ConvT → AttGate ←skip  │ ×4
              │    → concat → ResBlock  │
              │  (512→256→128→64 ch)    │
              └────────────┬────────────┘
                           │
                  Head: 1×1 Conv [2 ch]
                  ch0 = amplitude logits
                  ch1 = phase logits  (→ tanh at loss/test time)

Key components
──────────────
ResidualBlock
  Two Conv-BN-ReLU layers with an additive skip (He et al., 2016).
  A 1×1 projection is used when in_ch ≠ out_ch.  Prevents vanishing
  gradients and improves feature reuse through deep networks.

AttentionGate  (Oktay et al., 2018 — "Attention U-Net")
  Additive soft-attention between the gating signal (decoder) and the
  skip-connection features (encoder).  Produces per-spatial-location
  weights α ∈ [0,1] that suppress irrelevant background activations
  and focus the decoder on the object of interest.
  α = σ( ψ( ReLU( Wg·g + Wx·x ) ) )
  output = α ⊙ x

Output convention
──────────────────
  pred[:, 0:1]  → raw amplitude logits   (clamped to [0,1] at test time)
  pred[:, 1:2]  → raw phase logits       (tanh → [−1,1] = [−π,π]/π)
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ── building blocks ──────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """
    Pre-activation Residual Block.
    Conv-BN-ReLU → Conv-BN → + skip → ReLU
    1×1 projection on skip when in_ch ≠ out_ch.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.proj = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.branch(x) + self.proj(x))


class AttentionGate(nn.Module):
    """
    Additive attention gate (Oktay et al., 2018).

    g : gating signal from the decoder (coarser, higher-level context)
    x : skip-connection features from the encoder (same spatial size)

    α = σ( ψ( ReLU( Wg·g + Wx·x ) ) )
    return α ⊙ x
    """

    def __init__(self, f_g: int, f_x: int, f_int: int):
        super().__init__()
        self.Wg  = nn.Sequential(
            nn.Conv2d(f_g,  f_int, 1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.Wx  = nn.Sequential(
            nn.Conv2d(f_x,  f_int, 1, bias=False),
            nn.BatchNorm2d(f_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(f_int, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        alpha = self.psi(self.relu(self.Wg(g) + self.Wx(x)))
        return x * alpha


# ── full model ────────────────────────────────────────────────────────────────

class ResUNet(nn.Module):
    """
    Residual U-Net with Attention Gates.

    Parameters
    ──────────
    in_channels   : 1  (grayscale hologram)
    out_channels  : 2  (amplitude ch0, phase ch1)
    base_channels : 64 → channel widths [64,128,256,512,1024]
                        ≈ 31 M trainable parameters
    """

    def __init__(
        self,
        in_channels:   int = 1,
        out_channels:  int = 2,
        base_channels: int = 64,
    ):
        super().__init__()
        c = [base_channels * (2 ** i) for i in range(5)]
        # c = [64, 128, 256, 512, 1024]

        # ── encoder ──────────────────────────────────────────────────────
        self.enc1 = ResidualBlock(in_channels, c[0])
        self.enc2 = ResidualBlock(c[0],        c[1])
        self.enc3 = ResidualBlock(c[1],        c[2])
        self.enc4 = ResidualBlock(c[2],        c[3])
        self.pool = nn.MaxPool2d(2)

        # ── bottleneck ────────────────────────────────────────────────────
        self.bottleneck = ResidualBlock(c[3], c[4])

        # ── upsamplers ────────────────────────────────────────────────────
        self.up4 = nn.ConvTranspose2d(c[4], c[3], kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose2d(c[3], c[2], kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose2d(c[2], c[1], kernel_size=2, stride=2)
        self.up1 = nn.ConvTranspose2d(c[1], c[0], kernel_size=2, stride=2)

        # ── attention gates (g = decoder ch, x = encoder ch) ─────────────
        self.att4 = AttentionGate(f_g=c[3], f_x=c[3], f_int=c[3] // 2)
        self.att3 = AttentionGate(f_g=c[2], f_x=c[2], f_int=c[2] // 2)
        self.att2 = AttentionGate(f_g=c[1], f_x=c[1], f_int=c[1] // 2)
        self.att1 = AttentionGate(f_g=c[0], f_x=c[0], f_int=c[0] // 2)

        # ── decoder blocks: upsampled + attention-gated skip → concat → ResBlock
        self.dec4 = ResidualBlock(c[3] * 2, c[3])
        self.dec3 = ResidualBlock(c[2] * 2, c[2])
        self.dec2 = ResidualBlock(c[1] * 2, c[1])
        self.dec1 = ResidualBlock(c[0] * 2, c[0])

        # ── output head ───────────────────────────────────────────────────
        self.head = nn.Conv2d(c[0], out_channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── encode ────────────────────────────────────────────────────────
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # ── bottleneck ────────────────────────────────────────────────────
        b = self.bottleneck(self.pool(e4))

        # ── decode with attention-gated skip connections ──────────────────
        d4 = self.dec4(torch.cat([self.up4(b),  self.att4(self.up4(b),  e4)], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), self.att3(self.up3(d4), e3)], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), self.att2(self.up2(d3), e2)], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), self.att1(self.up1(d2), e1)], dim=1))

        return self.head(d1)   # [B, 2, H, W]


# Backward-compatible alias
UNet = ResUNet


# ── quick sanity check ────────────────────────────────────────────────────────
if __name__ == "__main__":
    model = ResUNet(in_channels=1, out_channels=2, base_channels=64)
    n     = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ResUNet (Att)  trainable params: {n:,}")
    dummy = torch.randn(2, 1, 256, 256)
    out   = model(dummy)
    print(f"Input  : {dummy.shape}  →  Output : {out.shape}")
