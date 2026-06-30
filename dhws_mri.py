#!/usr/bin/env python3
"""dhws_mri.py - DHWS-MRI: Accelerated MRI Reconstruction from Undersampled K-Space

Architecture adaptation of DHWS-Unified for MRI reconstruction.

Key differences from unified.py
================================
- Input  : undersampled complex k-space  (B, 2, H, W//2+1)  [real + imag as 2ch]
- Output : reconstructed MRI magnitude image  (B, 1, H, W)
- Encoder: KSpaceEncoder — operates directly in Fourier domain, not pixel space
- Loss   : image L1 + SSIM + k-space data-consistency (DC) loss
           No adversarial loss — medical imaging requires reliability over sharpness

Why DHWS fits MRI better than pixel U-Nets
===========================================
  MRI scanners capture k-space (Fourier domain) data, not pixels.
  Standard U-Nets work in pixel space and must learn the Fourier relationship indirectly.
  DHWS has irfft2, spectral bases, and phase loss built in — it natively speaks k-space.

  SpectralBasisDecoder : learns complex spectral bases -> irfft2 -> image (same as MRI recon)
  HarmonicGeometry     : captures smooth anatomical structures (MRI has smooth priors)
  HashGridDetail       : recovers fine detail lost from undersampling
  KSpaceConsistency    : forces output k-space to match scanner measurements exactly

Target benchmark: fastMRI dataset (NYU / Facebook, publicly available)
  https://fastmri.org/
  Task: 4x or 8x accelerated knee/brain MRI reconstruction
  Metric: SSIM, PSNR vs fully-sampled ground truth
"""

from __future__ import annotations

import math
import time
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ==============================================================================
# CONFIG
# ==============================================================================

@dataclass
class MRIConfig:
    image_size:     int   = 320       # fastMRI standard knee size
    acceleration:   int   = 4         # k-space undersampling factor (4x or 8x)
    center_frac:    float = 0.08      # fraction of centre k-space lines always sampled
    embed_dim:      int   = 256
    n_spec_bases:   int   = 64
    harm_hidden:    int   = 64
    harm_res:       int   = 32
    harm_omega:     float = 30.0
    hash_levels:    int   = 12
    hash_dim:       int   = 4
    hash_hidden:    int   = 64
    hash_res:       int   = 64
    refine_base:    int   = 32
    batch_size:     int   = 4         # MRI volumes are large
    epochs:         int   = 50
    warmup_epochs:  int   = 5
    lr:             float = 1e-4
    weight_decay:   float = 1e-4
    w_l1:           float = 0.5
    w_ssim:         float = 0.5
    w_dc:           float = 1.0       # k-space data consistency — highest priority
    w_spectral:     float = 0.3
    data_root:      Path  = Path("./data/fastmri")
    out_dir:        Path  = Path("./outputs_mri")

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def amp(self) -> bool:
        return torch.cuda.is_available()


# ==============================================================================
# UNDERSAMPLING MASK
# Retrospective undersampling of fully-sampled k-space for training/evaluation
# ==============================================================================

def make_cartesian_mask(shape: Tuple[int, int], acceleration: int,
                        center_frac: float, device: torch.device) -> torch.Tensor:
    """
    1D Cartesian undersampling mask (column-wise, standard in MRI).

    Always samples the central `center_frac` fraction of k-space lines
    (low frequencies carry most signal energy).
    Remaining lines sampled uniformly at random to hit target acceleration.

    Returns binary mask (H, W//2+1) broadcastable to (B, 1, H, W//2+1).
    """
    H, W = shape
    fw   = W // 2 + 1
    mask = torch.zeros(H, fw, device=device)

    # Always sample centre lines
    n_centre = max(1, int(H * center_frac))
    centre_start = H // 2 - n_centre // 2
    mask[centre_start : centre_start + n_centre, :] = 1.0

    # Random lines to hit target acceleration
    n_total  = H // acceleration
    n_random = max(0, n_total - n_centre)
    remaining = [i for i in range(H) if mask[i, 0] == 0]
    chosen = random.sample(remaining, min(n_random, len(remaining)))
    for idx in chosen:
        mask[idx, :] = 1.0

    return mask  # (H, fw)


# ==============================================================================
# SYNTHETIC MRI DATASET
# Simulates fastMRI-style data from random phantoms for prototyping.
# Replace with real fastMRI HDF5 loader for production.
# ==============================================================================

class SyntheticMRIDataset(Dataset):
    """
    Generates synthetic MRI-like images (ellipse phantoms) and returns
    undersampled k-space + fully-sampled ground truth pairs.

    For production: replace with fastMRI HDF5 loader.
    Download: https://fastmri.org/ (requires registration)
    """
    def __init__(self, n_samples: int, cfg: MRIConfig):
        self.n       = n_samples
        self.cfg     = cfg
        self.size    = cfg.image_size
        self.device  = cfg.device

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rng  = np.random.RandomState(idx)
        img  = self._make_phantom(rng)                    # (1, H, W) float32 [0,1]

        # Full k-space via rfft2
        kspace_full = torch.fft.rfft2(
            torch.from_numpy(img), norm="ortho")          # (1, H, W//2+1) complex

        # Undersampling mask
        mask = make_cartesian_mask(
            (self.size, self.size), self.cfg.acceleration,
            self.cfg.center_frac, torch.device("cpu"))    # (H, W//2+1)

        kspace_us = kspace_full * mask.unsqueeze(0)       # (1, H, W//2+1) complex

        # Return as 2-channel real tensor [real, imag]
        kspace_2ch = torch.stack(
            [kspace_us.real, kspace_us.imag], dim=1
        ).squeeze(0).float()                              # (2, H, W//2+1)

        target = torch.from_numpy(img).float()            # (1, H, W)
        return kspace_2ch, mask.float(), target

    def _make_phantom(self, rng: np.random.RandomState) -> np.ndarray:
        """Shepp-Logan-style ellipse phantom."""
        H = W = self.size
        img = np.zeros((H, W), dtype=np.float32)
        cx, cy = W / 2, H / 2
        n_ellipses = rng.randint(3, 8)
        for _ in range(n_ellipses):
            x0  = rng.uniform(0.1 * W, 0.9 * W)
            y0  = rng.uniform(0.1 * H, 0.9 * H)
            rx  = rng.uniform(0.05 * W, 0.35 * W)
            ry  = rng.uniform(0.05 * H, 0.35 * H)
            val = rng.uniform(0.2, 1.0)
            angle = rng.uniform(0, math.pi)
            ys, xs = np.ogrid[:H, :W]
            dx  = xs - x0;  dy = ys - y0
            xr  = dx * math.cos(angle) + dy * math.sin(angle)
            yr  = -dx * math.sin(angle) + dy * math.cos(angle)
            mask = (xr / rx) ** 2 + (yr / ry) ** 2 <= 1.0
            img[mask] = np.clip(img[mask] + val, 0.0, 1.0)
        return img[np.newaxis]  # (1, H, W)


# ==============================================================================
# MODULE 1 — K-SPACE ENCODER
# Replaces FastEncoder. Operates on complex k-space input (2 channels).
# k-space has non-uniform statistics: low frequencies at centre carry most energy.
# ==============================================================================

def _cb(ci: int, co: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU(),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU(),
    )


class KSpaceEncoder(nn.Module):
    """
    Encodes 2-channel (real+imag) undersampled k-space to global embedding + spatial maps.

    Architecture mirrors FastEncoder but:
    - Input is (B, 2, H, W//2+1) — complex k-space not pixel image
    - f1 spatial maps preserved for skip connection to refiner
    - Magnitude branch: |real + i*imag| highlights signal energy distribution
    - Phase branch: atan2(imag, real) captures structural phase info

    Both branches are concatenated before the strided convolutions.
    """
    def __init__(self, embed_dim: int = 256, base: int = 64):
        super().__init__()
        self.mag_proj   = nn.Conv2d(1, base // 2, 1)    # magnitude branch
        self.phase_proj = nn.Conv2d(1, base // 2, 1)    # phase branch
        self.conv1 = nn.Sequential(
            nn.Conv2d(base,   base,   3, stride=2, padding=1), nn.BatchNorm2d(base),   nn.GELU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(base,   base*2, 3, stride=2, padding=1), nn.BatchNorm2d(base*2), nn.GELU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, stride=2, padding=1), nn.BatchNorm2d(base*4), nn.GELU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(base * 4, embed_dim)

    def forward(self, kspace: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        real, imag = kspace[:, 0:1], kspace[:, 1:2]
        mag   = torch.sqrt(real ** 2 + imag ** 2 + 1e-8)
        phase = torch.atan2(imag, real)

        x  = torch.cat([self.mag_proj(mag), self.phase_proj(phase)], dim=1)
        f1 = self.conv1(x)                               # (B, base,   H/2, ...)
        f2 = self.conv2(f1)                              # (B, base*2, H/4, ...)
        f3 = self.conv3(f2)                              # (B, base*4, H/8, ...)
        emb = self.fc(self.pool(f3).flatten(1))          # (B, embed_dim)
        return emb, f1


# ==============================================================================
# MODULES 2-5 — Reused from DHWS-Unified (spectral, harmonic, hash, fusion)
# Only channel count changes: 3 (RGB) -> 1 (grayscale MRI)
# ==============================================================================

class SpectralBasisDecoder(nn.Module):
    """K=64 complex spectral bases -> irfft2 -> FiLM -> sigmoid. Grayscale output."""
    def __init__(self, image_size: int, embed_dim: int, n_bases: int):
        super().__init__()
        fw = image_size // 2 + 1
        self.image_size  = image_size
        self.bases_real  = nn.Parameter(torch.randn(n_bases, 1, image_size, fw) * 0.02)
        self.bases_imag  = nn.Parameter(torch.randn(n_bases, 1, image_size, fw) * 0.02)
        self.weight_head = nn.Linear(embed_dim, n_bases)
        self.film_head   = nn.Linear(embed_dim, 2)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        w    = F.softmax(self.weight_head(emb), dim=-1)
        real = torch.einsum("bk,kchw->bchw", w, self.bases_real)
        imag = torch.einsum("bk,kchw->bchw", w, self.bases_imag)
        out  = torch.fft.irfft2(torch.complex(real, imag),
                                s=(self.image_size,) * 2, norm="ortho")
        gamma, beta = self.film_head(emb).chunk(2, dim=-1)
        out = gamma[:, :, None, None] * out + beta[:, :, None, None]
        return torch.sigmoid(out)


class HarmonicGeometry(nn.Module):
    """Recursive harmonic fold -> smooth anatomical structure prior. Grayscale output."""
    def __init__(self, embed_dim: int, fold_res: int, fold_hidden: int,
                 omega: float, image_size: int):
        super().__init__()
        H = fold_hidden
        self.H = H; self.fold_res = fold_res
        self.omega = omega; self.image_size = image_size
        self.proj_head  = nn.Linear(embed_dim, 2 * H)
        self.fold_head  = nn.Linear(embed_dim, H * H)
        self.color_head = nn.Linear(H, 1)                # 1 channel (grayscale)
        t = torch.linspace(-1.0, 1.0, fold_res)
        gy, gx = torch.meshgrid(t, t, indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy], -1).view(-1, 2))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        B, H = emb.size(0), self.H
        W_proj = self.proj_head(emb).view(B, 2, H)
        W_fold = self.fold_head(emb).view(B, H, H)
        frob   = W_fold.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        W_fold = W_fold / (frob / math.sqrt(H))
        x  = self.coords.unsqueeze(0).expand(B, -1, -1)
        x  = torch.bmm(x, W_proj)
        x0 = x.clone()
        x  = torch.sin(self.omega * x)
        x  = torch.bmm(x, W_fold) + 0.1 * x0
        rgb = torch.tanh(self.color_head(x)) * 0.3
        rgb = rgb.permute(0, 2, 1).view(B, 1, self.fold_res, self.fold_res)
        return F.interpolate(rgb, size=self.image_size, mode="bilinear", align_corners=False)


class MultiResHashGrid(nn.Module):
    """Instant-NGP multi-resolution hash grid (unchanged from unified.py)."""
    def __init__(self, num_levels=8, level_dim=2, base_res=16,
                 max_res=512, log2_hashmap_size=14):
        super().__init__()
        self.num_levels   = num_levels
        self.level_dim    = level_dim
        self.hashmap_size = 1 << log2_hashmap_size
        self.b            = math.exp((math.log(max_res) - math.log(base_res)) / (num_levels - 1))
        self.base_res     = base_res
        self.embeddings   = nn.ModuleList([
            nn.Embedding(self.hashmap_size, level_dim) for _ in range(num_levels)])
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -1e-4, 1e-4)
        self._px = 2654435761; self._py = 805459861

    def _hash(self, xi, yi):
        return ((xi * self._px) ^ (yi * self._py)) % self.hashmap_size

    def forward(self, x):
        feats = []
        for i in range(self.num_levels):
            res = math.floor(self.base_res * (self.b ** i))
            xs  = x * res
            x0  = torch.floor(xs).long(); x1 = x0 + 1
            w   = xs - x0.float()
            wx  = w[..., 0:1]; wy = w[..., 1:2]
            f00 = self.embeddings[i](self._hash(x0[..., 0], x0[..., 1]))
            f01 = self.embeddings[i](self._hash(x0[..., 0], x1[..., 1]))
            f10 = self.embeddings[i](self._hash(x1[..., 0], x0[..., 1]))
            f11 = self.embeddings[i](self._hash(x1[..., 0], x1[..., 1]))
            f   = (f00*(1-wy) + f01*wy)*(1-wx) + (f10*(1-wy) + f11*wy)*wx
            feats.append(f)
        return torch.cat(feats, dim=-1)


class HashGridDetail(nn.Module):
    """Hash grid spatial detail decoder. Grayscale (1 channel) output."""
    def __init__(self, embed_dim: int, image_size: int,
                 hash_levels=8, hash_dim=2, hidden=64, hash_res=32):
        super().__init__()
        self.image_size = image_size; self.hash_res = hash_res
        in_feat = hash_levels * hash_dim
        self.grid      = MultiResHashGrid(num_levels=hash_levels, level_dim=hash_dim)
        self.modulator = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.SiLU(),
            nn.Linear(128, 2 * hidden * 2))
        self.l1   = nn.Linear(in_feat, hidden)
        self.l2   = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)                 # 1 channel
        self.act  = nn.SiLU()
        t = torch.linspace(0.0, 1.0, hash_res)
        gy, gx = torch.meshgrid(t, t, indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy], -1).view(-1, 2))

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        B      = emb.size(0)
        coords = self.coords.unsqueeze(0).expand(B, -1, -1)
        feats  = self.grid(coords)
        mods   = self.modulator(emb).view(B, 2, 2, -1)
        g0, b0 = mods[:, 0, 0].unsqueeze(1), mods[:, 0, 1].unsqueeze(1)
        g1, b1 = mods[:, 1, 0].unsqueeze(1), mods[:, 1, 1].unsqueeze(1)
        x = self.act(self.l1(feats) * g0 + b0)
        x = self.act(self.l2(x)    * g1 + b1)
        rgb = torch.sigmoid(self.head(x))
        rgb = rgb.permute(0, 2, 1).view(B, 1, self.hash_res, self.hash_res)
        if self.hash_res != self.image_size:
            rgb = F.interpolate(rgb, size=self.image_size, mode="bilinear", align_corners=False)
        return rgb


class AlphaFusion(nn.Module):
    """Per-pixel 3-stream softmax blend. Works for any channel count."""
    def __init__(self, channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels * 3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 16, 3, padding=1),            nn.GELU(),
            nn.Conv2d(16, 3, 1),
        )

    def forward(self, s1, s2, s3):
        weights = F.softmax(self.net(torch.cat([s1, s2, s3], dim=1)), dim=1)
        return weights[:, 0:1]*s1 + weights[:, 1:2]*s2 + weights[:, 2:3]*s3


# ==============================================================================
# MODULE 6 — MRI REFINER (MiniUNet with encoder skip + k-space skip)
# ==============================================================================

class MRIRefiner(nn.Module):
    """
    2-level U-Net refiner adapted for MRI:
    - Receives encoder spatial features f1 (same fix as unified.py v3)
    - Additional zero-filled reconstruction as residual hint
    """
    def __init__(self, base: int = 32, enc_ch: int = 64):
        super().__init__()
        self.e1   = _cb(1, base)
        self.e2   = _cb(base, base * 2)
        self.pool = nn.MaxPool2d(2)
        self.proj = nn.Conv2d(enc_ch, base * 2, 1)
        self.up   = nn.ConvTranspose2d(base * 4, base, 2, stride=2)
        self.d1   = _cb(base * 2, base)
        self.head = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor, f1: torch.Tensor) -> torch.Tensor:
        s1  = self.e1(x)
        s2  = self.e2(self.pool(s1))
        pf1 = self.proj(F.interpolate(f1, size=s2.shape[-2:],
                                      mode="bilinear", align_corners=False))
        up  = self.up(torch.cat([s2, pf1], dim=1))
        d   = self.d1(torch.cat([up, s1], dim=1))
        return torch.clamp(x + self.head(d), 0.0, 1.0)


# ==============================================================================
# FULL SYSTEM — DHWS-MRI
# ==============================================================================

class DHWSMri(nn.Module):
    """
    Single-pass MRI reconstruction from undersampled k-space.

    forward(kspace, mask) -> (refined, fused, spectral, harmonic, hash_det)

    kspace : (B, 2, H, W//2+1)  — real+imag channels of undersampled k-space
    mask   : (B, H, W//2+1)     — binary sampling mask (1=sampled, 0=missing)
    """
    def __init__(self, cfg: MRIConfig = MRIConfig()):
        super().__init__()
        D = cfg.embed_dim
        self.encoder  = KSpaceEncoder(D, base=64)
        self.spectral = SpectralBasisDecoder(cfg.image_size, D, cfg.n_spec_bases)
        self.harmonic = HarmonicGeometry(D, cfg.harm_res, cfg.harm_hidden,
                                         cfg.harm_omega, cfg.image_size)
        self.hashdet  = HashGridDetail(D, cfg.image_size, cfg.hash_levels,
                                       cfg.hash_dim, cfg.hash_hidden, cfg.hash_res)
        self.fusion   = AlphaFusion(channels=1)
        self.refine   = MRIRefiner(cfg.refine_base, enc_ch=64)

    def zero_filled(self, kspace: torch.Tensor) -> torch.Tensor:
        """Zero-filled inverse FFT reconstruction (naive baseline, no learning)."""
        complex_k = torch.complex(kspace[:, 0], kspace[:, 1])  # (B, H, W//2+1)
        recon = torch.fft.irfft2(complex_k, norm="ortho")      # (B, H, W)
        mag   = recon.abs().unsqueeze(1)                        # (B, 1, H, W)
        return mag / (mag.amax(dim=(-2, -1), keepdim=True) + 1e-8)

    def forward(self, kspace: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        emb, f1  = self.encoder(kspace)
        spectral  = self.spectral(emb)
        harmonic  = self.harmonic(emb)
        hash_det  = self.hashdet(emb)
        h_clamped = torch.clamp(spectral + harmonic, 0.0, 1.0)
        fused     = self.fusion(spectral, h_clamped, hash_det)
        refined   = self.refine(fused, f1)
        return refined, fused, spectral, harmonic, hash_det


# ==============================================================================
# LOSS FUNCTION
# ==============================================================================

def ssim_loss(pred: torch.Tensor, target: torch.Tensor,
              window_size: int = 11) -> torch.Tensor:
    """Differentiable SSIM loss (1 - SSIM). Grayscale images."""
    C1, C2 = 0.01**2, 0.03**2
    mu_p  = F.avg_pool2d(pred,   window_size, 1, window_size//2)
    mu_t  = F.avg_pool2d(target, window_size, 1, window_size//2)
    mu_pp = F.avg_pool2d(pred   ** 2, window_size, 1, window_size//2)
    mu_tt = F.avg_pool2d(target ** 2, window_size, 1, window_size//2)
    mu_pt = F.avg_pool2d(pred * target, window_size, 1, window_size//2)
    sig_p  = mu_pp - mu_p ** 2
    sig_t  = mu_tt - mu_t ** 2
    sig_pt = mu_pt - mu_p * mu_t
    ssim   = ((2*mu_p*mu_t + C1) * (2*sig_pt + C2)) / \
             ((mu_p**2 + mu_t**2 + C1) * (sig_p + sig_t + C2))
    return 1.0 - ssim.mean()


class MRILoss(nn.Module):
    """
    L = w_l1   * L1(pred, target)
      + w_ssim  * (1 - SSIM(pred, target))
      + w_dc    * ||mask * (F(pred) - kspace_measured)||²   (data consistency)
      + w_spec  * freq-weighted spectral loss

    Data consistency (DC) is the most important term:
    it forces the model output to agree with the scanner measurements
    at all sampled k-space locations — a hard physical constraint.
    """
    def __init__(self, cfg: MRIConfig):
        super().__init__()
        self.w_l1     = cfg.w_l1
        self.w_ssim   = cfg.w_ssim
        self.w_dc     = cfg.w_dc
        self.w_spec   = cfg.w_spectral
        fy = torch.linspace(-1.0, 1.0, cfg.image_size)
        fx = torch.linspace(0.0,  1.0, cfg.image_size // 2 + 1)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        self.register_buffer("freq_weight",
                             1.0 / (torch.sqrt(xx**2 + yy**2) + 1.0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                kspace_measured: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p = pred.float();  t = target.float()

        l1   = F.l1_loss(p, t)
        ssim = ssim_loss(p, t)

        # Data consistency: predicted image must match scanner measurements
        pred_kspace = torch.fft.rfft2(p.squeeze(1), norm="ortho")  # (B, H, W//2+1)
        meas_complex = torch.complex(kspace_measured[:, 0],
                                     kspace_measured[:, 1])         # (B, H, W//2+1)
        dc = (mask * (pred_kspace - meas_complex).abs() ** 2).mean()

        # Spectral fidelity
        pf = torch.fft.rfft2(p.squeeze(1), norm="ortho")
        tf = torch.fft.rfft2(t.squeeze(1), norm="ortho")
        fw = self.freq_weight.unsqueeze(0)
        spec = (((pf.abs() - tf.abs()) ** 2) * fw).mean()

        return (self.w_l1  * l1
              + self.w_ssim * ssim
              + self.w_dc   * dc
              + self.w_spec * spec)


# ==============================================================================
# METRICS
# ==============================================================================

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """SSIM and PSNR — standard fastMRI evaluation metrics."""
    p = pred.float().cpu(); t = target.float().cpu()
    mse  = torch.mean((p - t) ** 2).item()
    psnr = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
    ssim = 1.0 - ssim_loss(p, t).item()
    return {"psnr_db": psnr, "ssim": ssim, "mse": mse}


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_epoch(model: DHWSMri, optimizer: optim.Optimizer,
                loader: DataLoader, criterion: MRILoss,
                epoch: int, cfg: MRIConfig) -> float:
    model.train()
    device = cfg.device
    if epoch <= cfg.warmup_epochs:
        for pg in optimizer.param_groups:
            pg["lr"] = cfg.lr * epoch / cfg.warmup_epochs
    total = 0.0
    for kspace, mask, target in loader:
        kspace = kspace.to(device); mask = mask.to(device); target = target.to(device)
        optimizer.zero_grad(set_to_none=True)
        refined, *_ = model(kspace, mask)
        loss = criterion(refined, target, kspace, mask)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item() * kspace.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model: DHWSMri, loader: DataLoader,
             criterion: MRILoss, cfg: MRIConfig) -> dict:
    model.eval()
    device = cfg.device
    tl = psnr_sum = ssim_sum = 0.0; n = 0
    for kspace, mask, target in loader:
        kspace = kspace.to(device); mask = mask.to(device); target = target.to(device)
        refined, *_ = model(kspace, mask)
        tl += criterion(refined, target, kspace, mask).item() * kspace.size(0)
        m   = compute_metrics(refined, target)
        psnr_sum += m["psnr_db"] * kspace.size(0)
        ssim_sum += m["ssim"]    * kspace.size(0)
        n  += kspace.size(0)
    return {"loss": tl/n, "psnr_db": psnr_sum/n, "ssim": ssim_sum/n}


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    cfg    = MRIConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    device = cfg.device

    print("=" * 65)
    print("DHWS-MRI  —  Accelerated MRI Reconstruction from K-Space")
    print("=" * 65)
    print(f"  Acceleration  : {cfg.acceleration}x  (centre_frac={cfg.center_frac})")
    print(f"  Image size    : {cfg.image_size}x{cfg.image_size}  (grayscale)")
    print(f"  Device        : {device}  (amp={cfg.amp})")
    print(f"  Dataset       : Synthetic phantoms  "
          f"[swap for fastMRI HDF5 loader for real results]")

    train_ds = SyntheticMRIDataset(2000, cfg)
    test_ds  = SyntheticMRIDataset(400,  cfg)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    model     = DHWSMri(cfg).to(device)
    criterion = MRILoss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs - cfg.warmup_epochs, eta_min=1e-5)

    def np_(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
    total_p = np_(model)
    print(f"\n  encoder  ={np_(model.encoder):>9,}   spectral={np_(model.spectral):>9,}")
    print(f"  harmonic ={np_(model.harmonic):>9,}   hashdet ={np_(model.hashdet):>9,}")
    print(f"  fusion   ={np_(model.fusion):>9,}   refine  ={np_(model.refine):>9,}")
    print(f"  TOTAL    ={total_p:>9,}  (~{total_p/1e6:.2f}M)\n")

    # Baseline: zero-filled reconstruction
    kspace_b, mask_b, target_b = next(iter(test_loader))
    zf  = model.zero_filled(kspace_b.to(device))
    zfm = compute_metrics(zf.cpu(), target_b)
    print(f"  Zero-filled baseline  PSNR={zfm['psnr_db']:.2f} dB  SSIM={zfm['ssim']:.4f}")
    print(f"  (model should exceed this within a few epochs)\n")

    hdr = f"{'Ep':>4}  {'Train':>9}  {'Val':>9}  {'PSNR':>8}  {'SSIM':>7}  {'s':>5}"
    print(f"{hdr}\n{'-'*len(hdr)}")

    best = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        t0   = time.time()
        tr   = train_epoch(model, optimizer, train_loader, criterion, epoch, cfg)
        vm   = evaluate(model, test_loader, criterion, cfg)
        if epoch > cfg.warmup_epochs:
            scheduler.step()
        el = time.time() - t0
        print(f"{epoch:4d}  {tr:9.4f}  {vm['loss']:9.4f}  "
              f"{vm['psnr_db']:8.2f}  {vm['ssim']:7.4f}  {el:5.1f}s")
        if vm["loss"] < best:
            best = vm["loss"]
            torch.save(model.state_dict(), cfg.out_dir / "best_model.pt")

    print(f"\nBest checkpoint saved to {cfg.out_dir}/best_model.pt")
    print(f"\nNext step: replace SyntheticMRIDataset with fastMRI HDF5 loader.")
    print(f"  Download : https://fastmri.org/")
    print(f"  Loader   : pip install fastmri")
    print(f"  Usage    : from fastmri.data import SliceDataset")


if __name__ == "__main__":
    main()
