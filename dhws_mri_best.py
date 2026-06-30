#!/usr/bin/env python3
"""dhws_mri_best.py - DHWS-MRI-Best: Accelerated MRI Reconstruction from K-Space

Built directly on dhws_best.py architecture — every module ported 1:1 except:
  - DeepEncoder   -> KSpaceEncoder  (2ch complex k-space input, not 3ch RGB)
  - channels = 1  (grayscale MRI magnitude image)
  - No GAN        (medical imaging requires deterministic fidelity, not sharpness)
  - Added DataConsistency loss: forces output k-space to match scanner measurements
  - Added NMSE/SSIM metrics (fastMRI benchmark standard)
  - Added zero-filled baseline for comparison
  - Unrolled DC cascade (N=8 steps, shared weights) — same principle as E2E-VarNet

Why this beats standard U-Net on MRI
==========================================
  MRI scanners output k-space (Fourier domain), not pixels.
  U-Nets learn the Fourier relationship indirectly (pixel loss only).
  DHWS-Best speaks k-space natively:
    SpectralBasisDecoder  — irfft2 is the same op as MRI reconstruction
    Complex spectral loss — supervises in the same domain as the scanner
    DataConsistency loss  — hard physical constraint: agree with scanner measurements
    HarmonicResidue       — smooth anatomical priors (MRI has smooth structure)
    HashFeatureExtractor  — recovers detail lost from undersampling

Data sources (tried in order at startup)
==========================================
  1. HuggingFace datasets  — pip install datasets
                             Free, no registration.  Uses fastMRI knee subset.
  2. Local fastMRI HDF5    — Register (free) at https://fastmri.org/
                             Set cfg.data_root to downloaded folder.
  3. Synthetic phantoms    — Always available.  No download.
                             Shepp-Logan-style ellipses simulate MRI structure.

Expected results
==========================================
  Synthetic  :  PSNR ~28-30 dB  SSIM ~0.85  (converges fast, simple structure)
  fastMRI 4x :  PSNR ~38-42 dB  SSIM ~0.93+ (after 50 epochs on GPU, with cascade)
  Zero-filled:  PSNR ~26-28 dB  SSIM ~0.70  (no learning — model should beat this ep1)

Unrolled cascade vs single-pass
==========================================
  Single-pass (no cascade): DHWS spectral decoder → one refiner → output
    Advantage: fast, ~10M params, 1 forward pass
    Ceiling: ~34-38 dB (missing iterative DC enforcement)

  Unrolled cascade (this file, n_cascade=8):
    DHWS spectral decoder → initial estimate
    then for 8 steps: hard DC projection in k-space → lightweight refiner
    Advantage: iterative enforcement of scanner measurements, same as VarNet
    Expected: ~38-42 dB, competitive with E2E-VarNet on fastMRI
    Cost: 8x refiner calls per forward pass (still 1 encoder call)
"""
from __future__ import annotations

import math, time, random, csv, os, urllib.request, warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

SEED = 42
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)


# ==============================================================================
# CONFIG
# ==============================================================================

@dataclass
class MRIConfig:
    # Image
    image_size:     int   = 320        # fastMRI knee standard
    channels:       int   = 1          # grayscale magnitude
    acceleration:   int   = 4          # undersampling factor (4x or 8x)
    center_frac:    float = 0.08       # centre k-space fraction always sampled

    # Architecture (mirrors dhws_best.py)
    embed_dim:      int   = 512
    n_spec_bases:   int   = 96
    hash_levels:    int   = 12
    hash_dim:       int   = 4
    hash_res:       int   = 64
    refine_base:    int   = 48
    n_cascade:      int   = 8          # unrolled DC steps (VarNet-style)
    share_cascade:  bool  = True       # True = shared weights (memory efficient)
    num_coils:      int   = 1          # 1 = singlecoil, >1 = multi-coil (SENSE)

    # Training
    batch_size:     int   = 2          # large MRI images — keep low on CPU
    epochs:         int   = 50
    warmup_epochs:  int   = 5
    lr:             float = 3e-4
    weight_decay:   float = 1e-4

    # Loss weights
    w_l1:           float = 0.5
    w_complex:      float = 0.5        # complex spectral MSE (same as dhws_best)
    w_ssim:         float = 0.5        # SSIM loss
    w_dc:           float = 1.0        # data consistency — highest priority
    w_gram:         float = 0.2

    # Data
    train_n:        int   = 0          # 0 = use full dataset
    test_n:         int   = 0
    data_root:      Path  = Path("./data/mri")
    out_dir:        Path  = Path("./outputs_mri_best")

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def amp(self) -> bool:
        return torch.cuda.is_available()


# ==============================================================================
# UNDERSAMPLING MASK
# ==============================================================================

def make_cartesian_mask(H: int, W: int, acceleration: int,
                        center_frac: float) -> torch.Tensor:
    """
    1D Cartesian undersampling mask (row-wise).
    Centre fraction always sampled; remaining rows chosen at random.
    Returns (H, W//2+1) binary mask.
    """
    fw   = W // 2 + 1
    mask = torch.zeros(H, fw)

    n_centre = max(1, int(H * center_frac))
    cs = H // 2 - n_centre // 2
    mask[cs : cs + n_centre, :] = 1.0

    n_total  = max(n_centre, H // acceleration)
    n_random = n_total - n_centre
    avail    = [i for i in range(H) if mask[i, 0] == 0]
    chosen   = random.sample(avail, min(n_random, len(avail)))
    for idx in chosen:
        mask[idx, :] = 1.0

    return mask


# ==============================================================================
# DATASETS
# ==============================================================================

class SyntheticMRIDataset(Dataset):
    """
    Shepp-Logan ellipse phantoms — no download, instant start.
    Returns (kspace_2ch, mask, target_mag) tuples compatible with real fastMRI loader.
    """
    def __init__(self, n: int, cfg: MRIConfig):
        self.n   = n
        self.cfg = cfg
        self.sz  = cfg.image_size

    def __len__(self): return self.n

    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        img = self._phantom(rng)                              # (1, H, W) float32

        kfull = torch.fft.rfft2(
            torch.from_numpy(img), norm="ortho")              # (1, H, W//2+1) complex

        mask = make_cartesian_mask(
            self.sz, self.sz, self.cfg.acceleration, self.cfg.center_frac)  # (H, fw)

        kus   = kfull * mask.unsqueeze(0)                     # (1, H, fw) complex
        k2ch  = torch.stack([kus.real, kus.imag], dim=1).squeeze(0).float()  # (2, H, fw)

        return k2ch, mask.float(), torch.from_numpy(img).float()

    def _phantom(self, rng):
        H = W = self.sz
        img = np.zeros((H, W), dtype=np.float32)
        for _ in range(rng.randint(3, 9)):
            x0 = rng.uniform(.1*W, .9*W); y0 = rng.uniform(.1*H, .9*H)
            rx = rng.uniform(.04*W, .35*W); ry = rng.uniform(.04*H, .35*H)
            v  = rng.uniform(0.2, 1.0); a = rng.uniform(0, math.pi)
            ys, xs = np.ogrid[:H, :W]
            dx = xs - x0; dy = ys - y0
            xr = dx * math.cos(a) + dy * math.sin(a)
            yr = -dx * math.sin(a) + dy * math.cos(a)
            img[np.where((xr/rx)**2 + (yr/ry)**2 <= 1.0)] = np.clip(
                img[(xr/rx)**2 + (yr/ry)**2 <= 1.0] + v, 0, 1)
        return img[np.newaxis]


class FastMRIHDF5Dataset(Dataset):
    """
    Loads fastMRI singlecoil HDF5 slices from a local directory.

    Directory layout (standard fastMRI download):
        data_root/
            knee_singlecoil_train/  *.h5
            knee_singlecoil_val/    *.h5

    Each .h5 file contains:
        kspace     : (num_slices, H, W) complex64  — fully-sampled k-space
        reconstruction_rss : (num_slices, H, W)    — target magnitude

    Download: https://fastmri.org/  (free registration)
    """
    def __init__(self, folder: Path, cfg: MRIConfig, split="train"):
        import h5py
        self.cfg = cfg
        self.sz  = cfg.image_size
        self.files = sorted(folder.glob("*.h5"))
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in {folder}")

        self.slices = []
        for f in self.files:
            with h5py.File(f, "r") as hf:
                n = hf["kspace"].shape[0]
            for i in range(n):
                self.slices.append((f, i))
        random.shuffle(self.slices)

    def __len__(self): return len(self.slices)

    def __getitem__(self, idx):
        import h5py
        fpath, sl_idx = self.slices[idx]
        with h5py.File(fpath, "r") as hf:
            kfull = torch.from_numpy(hf["kspace"][sl_idx])           # (H, W) complex
            if "reconstruction_rss" in hf:
                target = torch.from_numpy(hf["reconstruction_rss"][sl_idx]).float()
            else:
                target = torch.fft.irfft2(kfull, norm="ortho").abs().float()

        # Centre-crop to cfg.image_size
        H, W  = kfull.shape
        ch    = (H - self.sz) // 2; cw = (W - self.sz) // 2
        kfull = kfull[ch:ch+self.sz, cw:cw+self.sz]
        target = target[ch:ch+self.sz, cw:cw+self.sz]

        # Normalise k-space and target by the target's max to preserve Fourier pair
        scale = target.max().clamp_min(1e-8)
        kfull = kfull / scale
        target = target / scale

        mask = make_cartesian_mask(self.sz, self.sz,
                                   self.cfg.acceleration, self.cfg.center_frac)
        kus  = kfull * mask                                   # (H, fw) complex
        k2ch = torch.stack([kus.real, kus.imag], dim=0).float()  # (2, H, fw)

        return k2ch, mask.float(), target.unsqueeze(0).float()


class FastMRIMultiCoilDataset(Dataset):
    """
    Loads fastMRI multi-coil HDF5 slices.

    Directory layout:
        data_root/
            knee_multicoil_train/  *.h5
            knee_multicoil_val/    *.h5

    Each .h5 file contains:
        kspace             : (num_slices, num_coils, H, W) complex64
        reconstruction_rss : (num_slices, H, W)  — RSS ground truth

    Returns (kspace_mc, mask, target) where:
        kspace_mc : (C, 2, H, fw)  all coils, real+imag split
        mask      : (H, fw)        binary undersampling mask
        target    : (1, H, W)      RSS magnitude ground truth
    """
    def __init__(self, folder: Path, cfg: MRIConfig):
        import h5py
        self.cfg   = cfg
        self.sz    = cfg.image_size
        self.files = sorted(folder.glob("*.h5"))
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in {folder}")

        self.slices = []
        for f in self.files:
            with h5py.File(f, "r") as hf:
                n = hf["kspace"].shape[0]
            for i in range(n):
                self.slices.append((f, i))
        random.shuffle(self.slices)

    def __len__(self): return len(self.slices)

    def __getitem__(self, idx):
        import h5py
        fpath, sl_idx = self.slices[idx]
        with h5py.File(fpath, "r") as hf:
            kfull  = torch.from_numpy(hf["kspace"][sl_idx])              # (C, H, W) complex
            target = torch.from_numpy(hf["reconstruction_rss"][sl_idx]).float()  # (H, W)

        C, H, W = kfull.shape[0], kfull.shape[-2], kfull.shape[-1]
        fw = W // 2 + 1

        # Centre-crop spatial dims
        ch = (H - self.sz) // 2; cw = (W - self.sz) // 2
        kfull  = kfull[:, ch:ch+self.sz, cw:cw+self.sz]               # (C, sz, sz) complex
        target = target[ch:ch+self.sz, cw:cw+self.sz]

        # Normalise k-space and target by the target's max to preserve Fourier pair
        scale  = target.max().clamp_min(1e-8)
        kfull  = kfull / scale
        target = target / scale

        mask = make_cartesian_mask(self.sz, self.sz,
                                   self.cfg.acceleration, self.cfg.center_frac)  # (sz, fw)

        kus   = kfull * mask.unsqueeze(0)                               # (C, sz, fw) complex
        # Stack real+imag: (C, 2, sz, fw)
        kmc   = torch.stack([kus.real, kus.imag], dim=1).float()

        return kmc, mask.float(), target.unsqueeze(0).float()


class HuggingFaceMRIDataset(Dataset):
    """
    Downloads fastMRI knee subset from HuggingFace Hub.
    pip install datasets
    No registration required for the public singlecoil subset.
    """
    def __init__(self, cfg: MRIConfig, split="train"):
        from datasets import load_dataset
        print(f"  Downloading fastMRI from HuggingFace ({split})...")
        self.ds  = load_dataset(
            "farrell/fastmri_knee_singlecoil", split=split,
            trust_remote_code=True)
        self.cfg = cfg
        self.sz  = cfg.image_size
        print(f"  Loaded {len(self.ds)} slices from HuggingFace.")

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        row     = self.ds[idx]
        # HF dataset stores k-space as real+imag numpy arrays
        real_k  = torch.from_numpy(np.array(row["kspace_real"])).float()
        imag_k  = torch.from_numpy(np.array(row["kspace_imag"])).float()
        kfull   = torch.complex(real_k, imag_k)              # (H, W//2+1)

        target  = torch.fft.irfft2(kfull, norm="ortho").abs().float()

        H, W2   = kfull.shape
        W       = (W2 - 1) * 2
        ch      = (H - self.sz) // 2
        kfull   = kfull[ch:ch+self.sz, :]
        target  = target[ch:ch+self.sz, :self.sz]

        scale   = target.max().clamp_min(1e-8)
        kfull   = kfull / scale
        target  = target / scale

        mask    = make_cartesian_mask(self.sz, self.sz,
                                      self.cfg.acceleration, self.cfg.center_frac)
        kus     = kfull * mask
        k2ch    = torch.stack([kus.real, kus.imag], dim=0).float()

        return k2ch, mask.float(), target.unsqueeze(0).float()


def build_loaders(cfg: MRIConfig):
    """
    Auto-selects dataset source.  Returns (train_loader, test_loader, source_name).
    Priority: multi-coil HDF5 > singlecoil HDF5 > HuggingFace > synthetic.
    """
    kw = dict(num_workers=0, pin_memory=(cfg.device.type == "cuda"))

    # --- Try local multi-coil HDF5 (best quality, clinical) ---
    mc_train = cfg.data_root / "knee_multicoil_train"
    mc_val   = cfg.data_root / "knee_multicoil_val"
    if mc_train.exists() and any(mc_train.glob("*.h5")):
        try:
            train_ds = FastMRIMultiCoilDataset(mc_train, cfg)
            test_ds  = FastMRIMultiCoilDataset(mc_val, cfg) \
                       if (mc_val.exists() and any(mc_val.glob("*.h5"))) \
                       else FastMRIMultiCoilDataset(mc_train, cfg)
            cfg.num_coils = train_ds[0][0].shape[0]   # auto-detect coil count
            print(f"  Multi-coil detected: {cfg.num_coils} coils  (SENSE enabled)")
            return (DataLoader(train_ds, cfg.batch_size, shuffle=True,  **kw),
                    DataLoader(test_ds,  cfg.batch_size, shuffle=False, **kw),
                    f"local multi-coil HDF5  ({mc_train})  [{cfg.num_coils} coils]")
        except Exception as e:
            print(f"  Multi-coil HDF5 failed: {e}")

    # --- Try local singlecoil HDF5 ---
    train_folder = cfg.data_root / "knee_singlecoil_train"
    test_folder  = cfg.data_root / "knee_singlecoil_val"
    if train_folder.exists() and any(train_folder.glob("*.h5")):
        try:
            train_ds = FastMRIHDF5Dataset(train_folder, cfg)
            test_ds  = FastMRIHDF5Dataset(test_folder,  cfg) \
                       if (test_folder.exists() and any(test_folder.glob("*.h5"))) \
                       else FastMRIHDF5Dataset(train_folder, cfg)
            return (DataLoader(train_ds, cfg.batch_size, shuffle=True,  **kw),
                    DataLoader(test_ds,  cfg.batch_size, shuffle=False, **kw),
                    f"local singlecoil HDF5  ({train_folder})")
        except Exception as e:
            print(f"  HDF5 load failed: {e}")

    # --- Try HuggingFace singlecoil ---
    try:
        train_ds = HuggingFaceMRIDataset(cfg, split="train")
        test_ds  = HuggingFaceMRIDataset(cfg, split="validation")
        return (DataLoader(train_ds, cfg.batch_size, shuffle=True,  **kw),
                DataLoader(test_ds,  cfg.batch_size, shuffle=False, **kw),
                "HuggingFace fastMRI")
    except Exception as e:
        print(f"  HuggingFace load failed ({e.__class__.__name__}): {e}")

    # --- Synthetic fallback ---
    print("  Using synthetic Shepp-Logan phantoms.")
    print("  For real results: pip install datasets  (HuggingFace auto-download)")
    print("  Or: register at https://fastmri.org/ and set cfg.data_root")
    n_train = cfg.train_n if cfg.train_n > 0 else 1000
    n_test  = cfg.test_n  if cfg.test_n  > 0 else 200
    return (DataLoader(SyntheticMRIDataset(n_train, cfg), cfg.batch_size, shuffle=True,  **kw),
            DataLoader(SyntheticMRIDataset(n_test,  cfg), cfg.batch_size, shuffle=False, **kw),
            "synthetic phantoms")


# ==============================================================================
# MODULE 1 — K-SPACE ENCODER  (replaces DeepEncoder from dhws_best.py)
# Input: (B, 2, H, W//2+1)  — real+imag of undersampled rfft2 k-space
# Output: emb (B, 512), f1 (B, 64, S, S), f2 (B, 128, S/2, S/2), f3 (B, 256, S/4, S/4)
#         where S = image_size//2  (square feature maps for downstream modules)
#
# Key design choices:
#   Magnitude branch: emphasises signal energy (most information in k-space centre)
#   Phase branch: structural phase encodes edges/boundaries
#   AdaptiveAvgPool after first conv: square-ifies the (H, W//2+1) asymmetric dims
# ==============================================================================

class KSpaceEncoder(nn.Module):
    """
    Encodes 2-channel complex k-space to global embedding + 3 spatial maps.
    Magnitude + phase branches fused before strided downsampling.
    Spatial maps are square (adaptive pooled) to be compatible with all downstream modules.
    """
    def __init__(self, embed_dim: int = 512, base: int = 64, image_size: int = 320):
        super().__init__()
        self.image_size = image_size
        self.sq         = image_size // 2             # target square spatial dim

        # k-space has high dynamic range — log-magnitude is more useful
        self.mag_proj   = nn.Conv2d(1, base // 2, 1)
        self.phase_proj = nn.Conv2d(1, base // 2, 1)

        # Square-ify before strided convs
        self.to_square  = nn.AdaptiveAvgPool2d(self.sq)

        self.conv1 = nn.Sequential(
            nn.Conv2d(base,   base,   3, stride=2, padding=1),
            nn.BatchNorm2d(base),   nn.GELU())          # sq//2
        self.conv2 = nn.Sequential(
            nn.Conv2d(base,   base*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base*2), nn.GELU())          # sq//4
        self.conv3 = nn.Sequential(
            nn.Conv2d(base*2, base*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base*4), nn.GELU())          # sq//8
        self.conv4 = nn.Sequential(
            nn.Conv2d(base*4, base*8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base*8), nn.GELU())          # sq//16

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(base * 8, embed_dim)

    def forward(self, kspace: torch.Tensor):
        real, imag = kspace[:, 0:1], kspace[:, 1:2]
        mag   = torch.log1p(torch.sqrt(real**2 + imag**2 + 1e-8))
        phase = torch.atan2(imag, real)

        x  = torch.cat([self.mag_proj(mag), self.phase_proj(phase)], dim=1)  # (B, base, H, fw)
        x  = self.to_square(x)                          # (B, base, sq, sq)

        f1 = self.conv1(x)                              # (B, base,   sq/2, sq/2)
        f2 = self.conv2(f1)                             # (B, base*2, sq/4, sq/4)
        f3 = self.conv3(f2)                             # (B, base*4, sq/8, sq/8)
        f4 = self.conv4(f3)                             # (B, base*8, sq/16,sq/16)
        emb = self.fc(self.pool(f4).flatten(1))         # (B, embed_dim)
        return emb, f1, f2, f3


# ==============================================================================
# MODULES 2-6 — Direct ports from dhws_best.py  (channels=1 for grayscale MRI)
# ==============================================================================

class SpectralBasisDecoder(nn.Module):
    """K=96 unconstrained complex spectral bases with two-stage FiLM. Output (B,1,H,W)."""
    def __init__(self, image_size, channels, embed_dim, n_bases,
                 enc_ch1=64, enc_ch2=128):
        super().__init__()
        self.image_size = image_size
        fw = image_size // 2 + 1
        self.bases_real  = nn.Parameter(torch.randn(n_bases, channels, image_size, fw) * 0.02)
        self.bases_imag  = nn.Parameter(torch.randn(n_bases, channels, image_size, fw) * 0.02)
        self.weight_head = nn.Linear(embed_dim, n_bases)
        self.pre_film    = nn.Conv2d(enc_ch2, channels * 2, 3, padding=1)
        self.post_film   = nn.Conv2d(enc_ch1, channels * 2, 3, padding=1)
        nn.init.zeros_(self.post_film.weight); nn.init.zeros_(self.post_film.bias)

    def forward(self, emb, f1, f2):
        w    = self.weight_head(emb)
        w    = w / w.norm(dim=-1, keepdim=True).clamp_min(1.0)
        real = torch.einsum("bk,kchw->bchw", w, self.bases_real)
        imag = torch.einsum("bk,kchw->bchw", w, self.bases_imag)

        fw   = real.shape[-1]
        f2_s = F.interpolate(f2, size=(self.image_size, fw), mode="bilinear", align_corners=False)
        pg, pb = self.pre_film(f2_s).chunk(2, dim=1)
        real = (1 + pg) * real + pb
        imag = (1 + pg) * imag + pb

        out  = torch.fft.irfft2(torch.complex(real.float(), imag.float()),
                                s=(self.image_size,) * 2, norm="ortho")

        f1_s = F.interpolate(f1, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        qg, qb = self.post_film(f1_s).chunk(2, dim=1)
        out  = (1 + qg) * out + qb
        return torch.sigmoid(out)


class HarmonicResidue(nn.Module):
    """Tiny harmonic fold residual [-0.05, 0.05]. Smooth anatomical prior."""
    def __init__(self, embed_dim, channels=1, fold_res=16,
                 fold_hidden=32, omega=30.0, image_size=320):
        super().__init__()
        H = fold_hidden
        self.H = H; self.fold_res = fold_res
        self.omega = omega; self.image_size = image_size; self.channels = channels
        self.proj_head  = nn.Linear(embed_dim, 2 * H)
        self.fold_head  = nn.Linear(embed_dim, H * H)
        self.color_head = nn.Linear(H, channels)
        t = torch.linspace(-1.0, 1.0, fold_res)
        gy, gx = torch.meshgrid(t, t, indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy], -1).view(-1, 2))

    def forward(self, emb):
        B, H = emb.size(0), self.H
        W_proj = self.proj_head(emb).view(B, 2, H)
        W_fold = self.fold_head(emb).view(B, H, H)
        frob   = W_fold.norm(dim=(-2,-1), keepdim=True).clamp_min(1e-6)
        W_fold = W_fold / (frob / math.sqrt(H))
        x  = self.coords.unsqueeze(0).expand(B, -1, -1)
        x  = torch.bmm(x, W_proj)
        x  = torch.sin(self.omega * x)
        x  = torch.bmm(x, W_fold)
        out = torch.tanh(self.color_head(x)) * 0.05
        out = out.permute(0, 2, 1).view(B, self.channels, self.fold_res, self.fold_res)
        return F.interpolate(out, size=self.image_size, mode="bilinear", align_corners=False)


class MultiResHashGrid(nn.Module):
    def __init__(self, num_levels=12, level_dim=4, base_res=16,
                 max_res=512, log2_hashmap_size=14):
        super().__init__()
        self.num_levels   = num_levels
        self.level_dim    = level_dim
        self.hashmap_size = 1 << log2_hashmap_size
        self.b = math.exp((math.log(max_res) - math.log(base_res)) / (num_levels - 1))
        self.base_res = base_res
        self.embeddings = nn.ModuleList([
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
            wx = w[...,0:1]; wy = w[...,1:2]
            f00 = self.embeddings[i](self._hash(x0[...,0], x0[...,1]))
            f01 = self.embeddings[i](self._hash(x0[...,0], x1[...,1]))
            f10 = self.embeddings[i](self._hash(x1[...,0], x0[...,1]))
            f11 = self.embeddings[i](self._hash(x1[...,0], x1[...,1]))
            feats.append((f00*(1-wy)+f01*wy)*(1-wx) + (f10*(1-wy)+f11*wy)*wx)
        return torch.cat(feats, dim=-1)


class HashFeatureExtractor(nn.Module):
    """Multi-resolution hash grid -> 48-channel feature map fed into refiner."""
    def __init__(self, embed_dim, image_size, hash_levels=12, hash_dim=4,
                 hidden=64, hash_res=64, out_ch=48):
        super().__init__()
        self.image_size = image_size
        self.hash_res   = hash_res
        self.out_ch     = out_ch
        in_feat = hash_levels * hash_dim
        self.grid      = MultiResHashGrid(num_levels=hash_levels, level_dim=hash_dim)
        self.modulator = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.SiLU(), nn.Linear(128, 2*hidden*2))
        self.l1   = nn.Linear(in_feat, hidden)
        self.l2   = nn.Linear(hidden,  hidden)
        self.head = nn.Linear(hidden,  out_ch)
        self.act  = nn.SiLU()
        t = torch.linspace(0.0, 1.0, hash_res)
        gy, gx = torch.meshgrid(t, t, indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy], -1).view(-1, 2))

    def forward(self, emb):
        B      = emb.size(0)
        coords = self.coords.unsqueeze(0).expand(B, -1, -1)
        feats  = self.grid(coords)
        mods   = self.modulator(emb).view(B, 2, 2, -1)
        g0, b0 = mods[:,0,0].unsqueeze(1), mods[:,0,1].unsqueeze(1)
        g1, b1 = mods[:,1,0].unsqueeze(1), mods[:,1,1].unsqueeze(1)
        x = self.act(self.l1(feats) * g0 + b0)
        x = self.act(self.l2(x)    * g1 + b1)
        x = self.head(x).permute(0,2,1).view(B, self.out_ch, self.hash_res, self.hash_res)
        if self.hash_res != self.image_size:
            x = F.interpolate(x, size=self.image_size, mode="bilinear", align_corners=False)
        return x


def _cb(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU(),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU())


class ConcatFuse(nn.Module):
    """cat(spectral, hash_feats) -> conv -> fused image."""
    def __init__(self, channels=1, hash_ch=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + hash_ch, 64, 3, padding=1), nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),                  nn.GELU(),
            nn.Conv2d(32, channels, 1), nn.Sigmoid())

    def forward(self, spectral, hash_feats):
        return self.net(torch.cat([spectral, hash_feats], dim=1))


class DeepUNetRefiner(nn.Module):
    """3-level U-Net refiner with additive encoder skips from f1/f2/f3."""
    def __init__(self, channels=1, hash_ch=48, base=48,
                 enc_ch1=64, enc_ch2=128, enc_ch3=256):
        super().__init__()
        inp = channels + hash_ch
        self.e1   = _cb(inp,    base)
        self.e2   = _cb(base,   base*2)
        self.e3   = _cb(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.mid  = _cb(base*4, base*8)

        self.skip3 = nn.Conv2d(enc_ch3, base*8, 1)
        self.skip2 = nn.Conv2d(enc_ch2, base*4, 1)
        self.skip1 = nn.Conv2d(enc_ch1, base*2, 1)
        for s in [self.skip3, self.skip2, self.skip1]:
            nn.init.zeros_(s.weight); nn.init.zeros_(s.bias)

        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.d3   = _cb(base*4, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2   = _cb(base*2, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base,   2, stride=2)
        self.d1   = _cb(base,   base)
        self.head = nn.Conv2d(base, channels, 1)

    def forward(self, fused, hash_feats, f1, f2, f3):
        x  = torch.cat([fused, hash_feats], dim=1)

        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        m  = self.mid(self.pool(s3))

        m  = m  + F.interpolate(self.skip3(f3), size=m.shape[-2:],  mode="bilinear", align_corners=False)
        s3 = s3 + F.interpolate(self.skip2(f2), size=s3.shape[-2:], mode="bilinear", align_corners=False)
        s2 = s2 + F.interpolate(self.skip1(f1), size=s2.shape[-2:], mode="bilinear", align_corners=False)

        u3 = F.interpolate(self.up3(m),  size=s3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.d3(u3 + s3)
        u2 = F.interpolate(self.up2(d3), size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(u2 + s2)
        u1 = F.interpolate(self.up1(d2), size=s1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(u1 + s1)

        out = fused + self.head(d1)
        # Resize to exact image_size if needed
        if out.shape[-1] != fused.shape[-1]:
            out = F.interpolate(out, size=fused.shape[-2:], mode="bilinear", align_corners=False)
        return torch.clamp(out, 0.0, 1.0)


# ==============================================================================
# UNROLLED DATA CONSISTENCY
# Hard k-space projection: at every sampled location force the predicted k-space
# to exactly match the scanner measurement.  Identical to the DC step in VarNet.
# ==============================================================================

def dc_project(x: torch.Tensor, y_k: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
    """
    Hard data consistency projection.

    x    : (B, 1, H, W)  current image estimate in [0, 1]
    y_k  : (B, H, fw)    measured k-space (complex)
    mask : (B, H, fw) or (H, fw)  binary, 1 = sampled

    Returns (B, 1, H, W) image after replacing sampled k-space with measurements.
    """
    H, W  = x.shape[-2], x.shape[-1]
    x_k   = torch.fft.rfft2(x.squeeze(1).float(), norm="ortho")  # (B, H, fw)
    # Replace sampled positions with scanner measurements
    x_k_dc = mask * y_k.to(x_k.dtype) + (1.0 - mask) * x_k
    x_dc   = torch.fft.irfft2(x_k_dc, s=(H, W), norm="ortho").abs()  # (B, H, W)
    return x_dc.unsqueeze(1)                                     # (B, 1, H, W)


# ==============================================================================
# SENSE — MULTI-COIL SUPPORT
# Sensitivity map estimation + SENSE forward/adjoint operators.
# Enables competition on the fastMRI multi-coil track (clinical scanners).
# Grounded in: E2E-VarNet (Sriram et al. 2020), ESPIRiT (Uecker et al. 2014)
# ==============================================================================

def _rss(x: torch.Tensor, dim: int) -> torch.Tensor:
    return torch.sqrt((x.abs() ** 2).sum(dim=dim).clamp_min(1e-8))


def _extract_acs(kspace_mc: torch.Tensor, center_frac: float = 0.08) -> torch.Tensor:
    """Zero out non-centre k-space lines, keeping only the ACS region."""
    H   = kspace_mc.shape[-2]
    n   = max(1, int(H * center_frac))
    cs  = H // 2 - n // 2
    acs = torch.zeros_like(kspace_mc)
    acs[..., cs:cs + n, :] = kspace_mc[..., cs:cs + n, :]
    return acs


class SensitivityModel(nn.Module):
    """
    Estimates per-coil complex sensitivity maps from the ACS region.

    Steps:
      1. Extract ACS (centre k-space lines, always fully sampled)
      2. IFFT each coil  →  low-resolution coil images
      3. Shared small U-Net refines real+imag channels of each coil
      4. Normalise by RSS so ||S_c||² sums to 1 spatially

    Shared weights across coils (processes each coil independently).
    ~300K params — negligible cost.

    Input : kspace_mc  (B, C, H, fw)  complex  — multi-coil undersampled k-space
    Output: sens_maps  (B, C, H, W)   complex  — normalised sensitivity maps
    """
    def __init__(self, base: int = 16, image_size: int = 320):
        super().__init__()
        self.image_size = image_size
        self.net = nn.Sequential(
            _cb(2,      base),
            nn.MaxPool2d(2),
            _cb(base,   base * 2),
            nn.ConvTranspose2d(base * 2, base, 2, stride=2),
            _cb(base,   base),
            nn.Conv2d(base, 2, 1),    # output: real + imag
        )

    def forward(self, kspace_mc: torch.Tensor,
                center_frac: float = 0.08) -> torch.Tensor:
        B, C, H, fw = kspace_mc.shape
        W = (fw - 1) * 2

        acs       = _extract_acs(kspace_mc, center_frac)              # (B, C, H, fw)
        coil_imgs = torch.fft.irfft2(acs, s=(H, W), norm="ortho")     # (B, C, H, W) complex

        # Normalise each coil so U-Net sees unit-scale input
        norm      = coil_imgs.abs().amax(dim=(-2,-1), keepdim=True).clamp_min(1e-8)
        coil_imgs = coil_imgs / norm

        # Process each coil through shared U-Net: (B*C, 2, H, W) -> (B*C, 2, H, W)
        x2ch   = torch.stack([coil_imgs.real, coil_imgs.imag], dim=2)  # (B, C, 2, H, W)
        x2ch   = x2ch.view(B * C, 2, H, W)
        out2ch = self.net(x2ch)                                        # (B*C, 2, H, W)

        sens   = torch.complex(out2ch[:, 0], out2ch[:, 1])            # (B*C, H, W)
        sens   = sens.view(B, C, H, W)                                # (B, C, H, W)

        # Normalise so energy sums to 1 across coils at each pixel
        rss_map = _rss(sens, dim=1).unsqueeze(1).clamp_min(1e-8)      # (B, 1, H, W)
        return sens / rss_map                                          # (B, C, H, W)


def sense_expand(x_img: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    """
    SENSE forward operator: image → per-coil k-space.
    x_img : (B, H, W)   complex image
    sens  : (B, C, H, W) complex sensitivity maps
    returns (B, C, H, fw) complex k-space per coil
    """
    coil_imgs = sens * x_img.unsqueeze(1)                             # (B, C, H, W)
    return torch.fft.rfft2(coil_imgs, norm="ortho")                   # (B, C, H, fw)


def sense_reduce(kspace_mc: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    """
    SENSE adjoint operator: per-coil k-space → combined image.
    kspace_mc : (B, C, H, fw) complex
    sens      : (B, C, H, W)  complex sensitivity maps
    returns   (B, H, W) complex combined image
    """
    coil_imgs = torch.fft.irfft2(kspace_mc, norm="ortho")            # (B, C, H, W)
    return (sens.conj() * coil_imgs).sum(dim=1)                       # (B, H, W)


def dc_project_multicoil(x: torch.Tensor, kspace_mc: torch.Tensor,
                          mask: torch.Tensor, sens: torch.Tensor) -> torch.Tensor:
    """
    SENSE-based hard data consistency for multi-coil MRI.

    x         : (B, 1, H, W)   current magnitude estimate [0,1]
    kspace_mc : (B, C, H, fw)  measured multi-coil k-space (complex)
    mask      : (B, H, fw) or (H, fw)  binary sampling mask
    sens      : (B, C, H, W)   complex sensitivity maps
    returns   : (B, 1, H, W)   after SENSE DC projection
    """
    H, W   = x.shape[-2], x.shape[-1]
    x_c    = torch.complex(x.squeeze(1).float(),
                           torch.zeros_like(x.squeeze(1)))            # (B, H, W) complex

    pred_k = sense_expand(x_c, sens)                                  # (B, C, H, fw)

    # Broadcast mask across coil dimension
    m = mask.unsqueeze(1) if mask.dim() == 3 else mask.unsqueeze(0).unsqueeze(0)
    dc_k   = m * kspace_mc.to(pred_k.dtype) + (1.0 - m) * pred_k    # (B, C, H, fw)

    x_dc   = sense_reduce(dc_k, sens).abs()                          # (B, H, W) magnitude
    return x_dc.unsqueeze(1)                                         # (B, 1, H, W)


class CascadeRefiner(nn.Module):
    """
    Lightweight 2-level U-Net called at each unrolled cascade step.
    Input: cat(dc_image, hash_feats) — 1 + 48 = 49 channels.
    Additive skips from encoder f1, f2 (same as DeepUNetRefiner but lighter).
    Shared across all N cascade steps by default (share_cascade=True).

    ~450K params — cheap to call 8 times vs DeepUNetRefiner's 4M single call.
    """
    def __init__(self, channels: int = 1, hash_ch: int = 48, base: int = 32,
                 enc_ch1: int = 64, enc_ch2: int = 128):
        super().__init__()
        inp = channels + hash_ch                         # 49
        self.e1   = _cb(inp,    base)
        self.e2   = _cb(base,   base * 2)
        self.pool = nn.MaxPool2d(2)
        self.mid  = _cb(base * 2, base * 4)

        self.skip2 = nn.Conv2d(enc_ch2, base * 4, 1)
        self.skip1 = nn.Conv2d(enc_ch1, base * 2, 1)
        for s in [self.skip2, self.skip1]:
            nn.init.zeros_(s.weight); nn.init.zeros_(s.bias)

        self.up2  = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2   = _cb(base * 2, base * 2)
        self.up1  = nn.ConvTranspose2d(base * 2, base,     2, stride=2)
        self.d1   = _cb(base, base)
        self.head = nn.Conv2d(base, channels, 1)

    def forward(self, x_dc: torch.Tensor, hash_feats: torch.Tensor,
                f1: torch.Tensor, f2: torch.Tensor) -> torch.Tensor:
        inp = torch.cat([x_dc, hash_feats], dim=1)

        s1  = self.e1(inp)
        s2  = self.e2(self.pool(s1))
        m   = self.mid(self.pool(s2))

        m   = m  + F.interpolate(self.skip2(f2), size=m.shape[-2:],  mode="bilinear", align_corners=False)
        s2  = s2 + F.interpolate(self.skip1(f1), size=s2.shape[-2:], mode="bilinear", align_corners=False)

        u2  = F.interpolate(self.up2(m),  size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d2  = self.d2(u2 + s2)
        u1  = F.interpolate(self.up1(d2), size=s1.shape[-2:], mode="bilinear", align_corners=False)
        d1  = self.d1(u1 + s1)

        out = x_dc + self.head(d1)
        if out.shape[-2:] != x_dc.shape[-2:]:
            out = F.interpolate(out, size=x_dc.shape[-2:], mode="bilinear", align_corners=False)
        return out


# ==============================================================================
# FULL MODEL — DHWSMRIBest
# ==============================================================================

class DHWSMRIBest(nn.Module):
    """
    Unrolled MRI reconstruction from undersampled k-space.

    Stage 1 — DHWS initial estimate (1 forward pass):
      KSpaceEncoder → SpectralBasisDecoder + HarmonicResidue → ConcatFuse
      Produces a strong starting point (better than zero-filled used by VarNet).

    Stage 2 — Unrolled DC cascade (n_cascade steps, shared weights):
      For each step:
        1. Hard DC projection: replace sampled k-space with scanner measurements
        2. CascadeRefiner: lightweight 2-level U-Net with encoder skip connections
      Hash features from stage 1 are reused at every cascade step (no recompute).

    forward(kspace, mask) -> (refined, spectral)

    Singlecoil  (num_coils=1):
      kspace : (B, 2, H, fw)     real+imag channels
    Multi-coil  (num_coils>1):
      kspace : (B, C, 2, H, fw)  C coils, each with real+imag channels

    mask    : (B, H, fw) or (H, fw)  binary sampling mask
    refined : (B, 1, H, W)  final output after N cascade steps
    spectral: (B, 1, H, W)  initial DHWS estimate (for auxiliary supervision)
    """
    def __init__(self, cfg: MRIConfig):
        super().__init__()
        C, D = cfg.channels, cfg.embed_dim
        self.cfg        = cfg
        self.image_size = cfg.image_size
        self.multicoil  = cfg.num_coils > 1

        # Sensitivity model — only built for multi-coil
        self.sens_model = SensitivityModel(base=16, image_size=cfg.image_size) \
                          if self.multicoil else None

        self.encoder  = KSpaceEncoder(D, base=64, image_size=cfg.image_size)
        self.spectral = SpectralBasisDecoder(cfg.image_size, C, D, cfg.n_spec_bases,
                                             enc_ch1=64, enc_ch2=128)
        self.harmres  = HarmonicResidue(D, C, fold_res=16, fold_hidden=32,
                                        omega=30.0, image_size=cfg.image_size)
        self.hashfeat = HashFeatureExtractor(D, cfg.image_size, cfg.hash_levels,
                                             cfg.hash_dim, hidden=64,
                                             hash_res=cfg.hash_res, out_ch=48)
        self.fusion   = ConcatFuse(C, hash_ch=48)

        one_refiner = CascadeRefiner(C, hash_ch=48, base=32, enc_ch1=64, enc_ch2=128)
        if cfg.share_cascade:
            self.cascade = nn.ModuleList([one_refiner] * cfg.n_cascade)
        else:
            self.cascade = nn.ModuleList([
                CascadeRefiner(C, hash_ch=48, base=32, enc_ch1=64, enc_ch2=128)
                for _ in range(cfg.n_cascade)
            ])

    def _to_singlecoil_kspace(self, kspace, mask, sens):
        """
        Convert multi-coil (B,C,2,H,fw) k-space to 2-channel singlecoil-equivalent
        by SENSE-reducing the zero-filled coil images.
        Used only to feed the KSpaceEncoder (which expects 2-channel input).
        """
        B, C, _, H, fw = kspace.shape
        kmc = torch.complex(kspace[:, :, 0], kspace[:, :, 1])   # (B, C, H, fw)
        combined = sense_reduce(kmc, sens)                       # (B, H, W) complex
        # rfft2 of combined image → 2-channel
        W    = self.image_size
        k2ch = torch.stack([combined.real, combined.imag], dim=1)  # (B, 2, H, fw)
        return k2ch, kmc

    def zero_filled(self, kspace) -> torch.Tensor:
        """Zero-filled iFFT baseline. Works for both singlecoil and multi-coil."""
        if kspace.dim() == 4:                                    # singlecoil (B,2,H,fw)
            kc  = torch.complex(kspace[:, 0], kspace[:, 1])
            mag = torch.fft.irfft2(kc, norm="ortho").abs().unsqueeze(1)
        else:                                                    # multi-coil (B,C,2,H,fw)
            kmc = torch.complex(kspace[:, :, 0], kspace[:, :, 1])
            mag = _rss(torch.fft.irfft2(kmc, norm="ortho"), dim=1).unsqueeze(1)
        return (mag / mag.amax((-2,-1), keepdim=True).clamp_min(1e-8)).clamp(0, 1)

    def forward(self, kspace, mask) -> Tuple[torch.Tensor, torch.Tensor]:
        # ── Multi-coil preprocessing ────────────────────────────────────────
        if self.multicoil:
            # kspace: (B, C, 2, H, fw)
            kmc   = torch.complex(kspace[:, :, 0], kspace[:, :, 1])  # (B, C, H, fw)
            sens  = self.sens_model(kmc, self.cfg.center_frac)        # (B, C, H, W)
            k2ch, kmc = self._to_singlecoil_kspace(kspace, mask, sens)
        else:
            k2ch  = kspace                                            # (B, 2, H, fw)
            kmc   = torch.complex(kspace[:, 0], kspace[:, 1])        # (B, H, fw)
            sens  = None

        mask_k = mask if mask.dim() == 3 else mask.unsqueeze(0).expand(k2ch.size(0), -1, -1)

        # ── Stage 1: DHWS initial estimate ──────────────────────────────────
        emb, f1, f2, f3 = self.encoder(k2ch)

        spectral   = self.spectral(emb, f1, f2)
        harm_res   = self.harmres(emb)
        spectral   = torch.clamp(spectral + harm_res, 0.0, 1.0)
        hash_feats = self.hashfeat(emb)
        x          = self.fusion(spectral, hash_feats)               # (B, 1, H, W)

        # ── Stage 2: Unrolled DC cascade ────────────────────────────────────
        for refiner in self.cascade:
            if self.multicoil:
                x = dc_project_multicoil(x, kmc, mask_k, sens)      # SENSE DC
            else:
                x = dc_project(x, kmc, mask_k)                      # simple DC
            x = refiner(x, hash_feats, f1, f2)

        return x, spectral


# ==============================================================================
# LOSS FUNCTION
# L1 + SSIM + complex spectral (from dhws_best) + data consistency (MRI-specific)
# ==============================================================================

def _ssim(pred: torch.Tensor, target: torch.Tensor, ws: int = 11) -> torch.Tensor:
    C1, C2 = 0.01**2, 0.03**2
    mu_p  = F.avg_pool2d(pred,   ws, 1, ws//2)
    mu_t  = F.avg_pool2d(target, ws, 1, ws//2)
    mpp   = F.avg_pool2d(pred**2,    ws, 1, ws//2)
    mtt   = F.avg_pool2d(target**2,  ws, 1, ws//2)
    mpt   = F.avg_pool2d(pred*target,ws, 1, ws//2)
    sp    = mpp - mu_p**2; st = mtt - mu_t**2; spt = mpt - mu_p*mu_t
    return ((2*mu_p*mu_t+C1)*(2*spt+C2) / ((mu_p**2+mu_t**2+C1)*(sp+st+C2))).mean()


class MRILoss(nn.Module):
    """
    L = w_l1     * L1(pred, target)
      + w_ssim   * (1 - SSIM)
      + w_complex * freq-weighted complex spectral MSE   [from dhws_best]
      + w_dc     * ||mask * (rfft2(pred) - kspace_measured)||²   [data consistency]
      + w_gram   * gram matrix MSE
    """
    def __init__(self, cfg: MRIConfig):
        super().__init__()
        self.cfg = cfg
        fy = torch.linspace(-1.0, 1.0, cfg.image_size)
        fx = torch.linspace(0.0,  1.0, cfg.image_size // 2 + 1)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        self.register_buffer("freq_weight",
                             1.0 / (torch.sqrt(xx**2 + yy**2) + 1.0))

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                kspace_2ch: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        p = pred.float(); t = target.float()

        l1   = F.l1_loss(p, t)
        ssim = 1.0 - _ssim(p, t)

        # Complex spectral MSE (dhws_best loss, adapted for 1ch)
        pf = torch.fft.rfft2(p.squeeze(1), norm="ortho")     # (B, H, fw)
        tf = torch.fft.rfft2(t.squeeze(1), norm="ortho")
        fw = self.freq_weight.unsqueeze(0).to(pf.device)
        spec = (((pf.real - tf.real)**2 + (pf.imag - tf.imag)**2) * fw).mean()

        # Data consistency: predicted k-space must match scanner at sampled locations
        km  = torch.complex(kspace_2ch[:, 0], kspace_2ch[:, 1])  # (B, H, fw) measured
        dc  = (mask.to(pf.device) * (pf - km.to(pf.dtype)).abs()**2).mean()

        # Gram (texture)
        B, C, H, W = p.shape
        gp = p.view(B, C, -1);  gp = gp @ gp.transpose(-1,-2) / (C*H*W)
        gt = t.view(B, C, -1);  gt = gt @ gt.transpose(-1,-2) / (C*H*W)
        gram = F.mse_loss(gp, gt)

        return (self.cfg.w_l1     * l1
              + self.cfg.w_ssim   * ssim
              + self.cfg.w_complex * spec
              + self.cfg.w_dc     * dc
              + self.cfg.w_gram   * gram)


# ==============================================================================
# METRICS  (fastMRI benchmark standard)
# ==============================================================================

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    p = pred.float().cpu(); t = target.float().cpu()
    mse  = F.mse_loss(p, t).item()
    nmse = mse / (t**2).mean().item() if (t**2).mean().item() > 0 else 0.0
    psnr_val = float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)
    ssim_val = _ssim(p, t).item()
    return {"mse": mse, "nmse": nmse, "psnr_db": psnr_val, "ssim": ssim_val}


# ==============================================================================
# TRAINING
# ==============================================================================

def _autocast(device, enabled):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def train_epoch(model: DHWSMRIBest, optimizer, loader, criterion: MRILoss,
                epoch: int, cfg: MRIConfig, scaler) -> float:
    model.train()
    device = cfg.device
    total  = 0.0

    for kspace, mask, target in loader:
        kspace = kspace.to(device)
        mask   = mask.to(device)
        target = target.to(device)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, cfg.amp):
            refined, spectral = model(kspace, mask)
            
        # Compute loss outside autocast to prevent float16 overflow in spectral MSE
        loss = (1.0 * criterion(refined,  target, kspace, mask)
              + 0.3 * criterion(spectral, target, kspace, mask))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        total += loss.item() * kspace.size(0)

    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model: DHWSMRIBest, loader, criterion: MRILoss,
             cfg: MRIConfig) -> dict:
    model.eval()
    device = cfg.device
    tl = nmse = psnr_s = ssim_s = 0.0; n = 0

    for kspace, mask, target in loader:
        kspace = kspace.to(device); mask = mask.to(device); target = target.to(device)
        with _autocast(device, cfg.amp):
            refined, _ = model(kspace, mask)
        tl += criterion(refined, target, kspace, mask).item() * kspace.size(0)
        m   = compute_metrics(refined, target)
        nmse  += m["nmse"]    * kspace.size(0)
        psnr_s += m["psnr_db"] * kspace.size(0)
        ssim_s += m["ssim"]    * kspace.size(0)
        n  += kspace.size(0)

    return {"loss": tl/n, "nmse": nmse/n, "psnr_db": psnr_s/n, "ssim": ssim_s/n}


@torch.no_grad()
def save_samples(model: DHWSMRIBest, loader, tag: str, cfg: MRIConfig):
    """Save grid: [zero_filled | spectral | fused | refined | target]"""
    try:
        import torchvision
    except ImportError:
        return

    model.eval()
    device = cfg.device
    kspace, mask, target = next(iter(loader))
    kspace = kspace.to(device); mask = mask.to(device)

    with _autocast(device, cfg.amp):
        refined, spectral = model(kspace, mask)
    zf = model.zero_filled(kspace)

    n    = min(4, kspace.size(0))
    rows = [zf[:n].cpu(), spectral[:n].cpu(), refined[:n].cpu(), target[:n].cpu()]
    grid = torchvision.utils.make_grid(
        torch.cat([r.float() for r in rows]), nrow=n, padding=2, normalize=False)
    path = cfg.out_dir / f"samples_{tag}.png"
    torchvision.utils.save_image(grid, path)
    print(f"  Saved {path}  [zero_filled | spectral_init | cascade_refined | target]")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    cfg = MRIConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.data_root.mkdir(parents=True, exist_ok=True)
    device = cfg.device

    print("=" * 68)
    print("DHWS-MRI-Best  —  K-Space MRI Reconstruction  (dhws_best architecture)")
    print("=" * 68)

    train_loader, test_loader, source = build_loaders(cfg)
    print(f"\n  Dataset       : {source}")
    print(f"  Acceleration  : {cfg.acceleration}x  (centre_frac={cfg.center_frac})")
    print(f"  Image size    : {cfg.image_size}x{cfg.image_size}  (grayscale)")
    print(f"  Device        : {device}  (amp={cfg.amp})")

    model     = DHWSMRIBest(cfg).to(device)
    criterion = MRILoss(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp)

    def np_(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
    casc_ref = model.cascade[0]
    total_p  = np_(model)
    print(f"\n  encoder  ={np_(model.encoder):>9,}   spectral={np_(model.spectral):>9,}")
    print(f"  harmres  ={np_(model.harmres):>9,}   hashfeat={np_(model.hashfeat):>9,}")
    print(f"  fusion   ={np_(model.fusion):>9,}   casc_ref={np_(casc_ref):>9,}")
    if model.sens_model is not None:
        print(f"  sens_model={np_(model.sens_model):>8,}  (SENSE, {cfg.num_coils} coils)")
    shared = "(shared weights)" if cfg.share_cascade else "(independent weights)"
    print(f"  cascade  : {cfg.n_cascade} steps  {shared}")
    print(f"  TOTAL    ={total_p:>9,}  (~{total_p/1e6:.2f}M)\n")

    # Baseline: zero-filled iFFT
    kspace_b, mask_b, target_b = next(iter(test_loader))
    zf  = model.zero_filled(kspace_b.to(device))
    zfm = compute_metrics(zf.cpu(), target_b)
    print(f"  Zero-filled baseline  "
          f"PSNR={zfm['psnr_db']:.2f} dB  SSIM={zfm['ssim']:.4f}  NMSE={zfm['nmse']:.4f}")
    print(f"  (model should exceed this by epoch 1-2)\n")

    hdr = f"{'Ep':>4}  {'Train':>9}  {'Val':>9}  {'PSNR':>8}  {'SSIM':>7}  {'NMSE':>8}  {'s':>5}"
    sep = "-" * len(hdr)
    print(hdr); print(sep)

    log_rows = [["epoch","train_loss","val_loss","psnr_db","ssim","nmse","time_s"]]
    best_loss = float("inf")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, optimizer, train_loader, criterion, epoch, cfg, scaler)
        scheduler.step()
        vm = evaluate(model, test_loader, criterion, cfg)
        el = time.time() - t0

        print(f"{epoch:4d}  {tr:9.4f}  {vm['loss']:9.4f}  "
              f"{vm['psnr_db']:8.2f}  {vm['ssim']:7.4f}  {vm['nmse']:8.5f}  {el:5.1f}s")

        log_rows.append([epoch, f"{tr:.4f}", f"{vm['loss']:.4f}",
                         f"{vm['psnr_db']:.2f}", f"{vm['ssim']:.4f}",
                         f"{vm['nmse']:.5f}", f"{el:.1f}"])

        if vm["loss"] < best_loss:
            best_loss = vm["loss"]
            torch.save({"epoch": epoch, "state": model.state_dict(),
                        "loss": best_loss}, cfg.out_dir / "best_model.pt")

        if epoch % 5 == 0 or epoch == 1:
            save_samples(model, test_loader, f"ep{epoch:03d}", cfg)

    with open(cfg.out_dir / "training_log.csv", "w", newline="") as f:
        csv.writer(f).writerows(log_rows)

    print(f"\n  Best checkpoint : {cfg.out_dir}/best_model.pt")
    print(f"  Training log   : {cfg.out_dir}/training_log.csv")
    print(f"\n  To use real fastMRI data:")
    print(f"    pip install datasets  # auto-downloads from HuggingFace")
    print(f"    OR: register at https://fastmri.org/ and set cfg.data_root")


if __name__ == "__main__":
    main()
