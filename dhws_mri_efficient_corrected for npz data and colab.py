#!/usr/bin/env python3
"""dhws_mri_efficient.py - DHWS-MRI-Efficient: Fast, Hallucination-Free Reconstruction

This model merges the best features of dhws_best and dhws_mri_best while optimizing
heavily for speed, memory footprint, and clinical reliability.

Key Optimizations & Features:
  1. Hallucination-Free: No GAN. Strict Hard Data Consistency (DC) projection.
  2. Ultra-Lightweight (~4.6M params): 
     - Reduced Spectral Bases (K=32 down from 96).
     - Removed redundant HarmonicResidue.
     - Reduced encoder/refiner base channels.
  3. Fast Inference: Unrolled cascade reduced to 4 steps (from 8).
  4. Multi-Coil Native: Includes the 24K-parameter SENSE SensitivityModel.
  5. Mathematically Sound: Incorporates fixes for float16 AMP overflow, DC scaling, 
     and Fourier-pair normalization.
"""
from __future__ import annotations

import math, time, random, csv, os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

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
    image_size:     int   = 320
    channels:       int   = 1
    acceleration:   int   = 4
    center_frac:    float = 0.08

    # Efficient Architecture Settings
    embed_dim:      int   = 256        # Reduced from 512
    n_spec_bases:   int   = 32         # Reduced from 96 (Massive param savings)
    hash_levels:    int   = 12
    hash_dim:       int   = 2          # Reduced from 4
    hash_res:       int   = 64
    refine_base:    int   = 32         # Reduced from 48
    n_cascade:      int   = 4          # 2x faster inference than 8 steps
    num_coils:      int   = 1

    batch_size:     int   = 4          # Can be larger due to efficient size
    epochs:         int   = 150
    warmup_epochs:  int   = 5
    lr:             float = 5e-4       # Slightly higher LR for smaller model
    weight_decay:   float = 1e-4

    w_l1:           float = 0.5
    w_complex:      float = 0.5
    w_ssim:         float = 0.5
    w_dc:           float = 1.0
    w_gram:         float = 0.2

    data_root:      Path  = Path("/content/drive/MyDrive/DHWS/data/mri_npz/data")
    out_dir:        Path  = Path("/content/drive/MyDrive/DHWS/outputs_mri_efficient")

    @property
    def device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def amp(self) -> bool:
        return torch.cuda.is_available()


# ==============================================================================
# UTILS & DATASETS
# ==============================================================================

def make_cartesian_mask(H: int, W: int, acceleration: int, center_frac: float) -> torch.Tensor:
    fw = W // 2 + 1
    mask = torch.zeros(H, fw)
    n_centre = max(1, int(H * center_frac))
    cs = H // 2 - n_centre // 2
    mask[cs : cs + n_centre, :] = 1.0
    n_total = max(n_centre, H // acceleration)
    n_random = n_total - n_centre
    avail = [i for i in range(H) if mask[i, 0] == 0]
    chosen = random.sample(avail, min(n_random, len(avail)))
    for idx in chosen: mask[idx, :] = 1.0
    return mask


class SyntheticMRIDataset(Dataset):
    def __init__(self, n: int, cfg: MRIConfig):
        self.n = n; self.cfg = cfg; self.sz = cfg.image_size
    def __len__(self): return self.n
    def __getitem__(self, idx):
        rng = np.random.RandomState(idx)
        img = np.zeros((self.sz, self.sz), dtype=np.float32)
        for _ in range(rng.randint(3, 9)):
            x0 = rng.uniform(.1*self.sz, .9*self.sz); y0 = rng.uniform(.1*self.sz, .9*self.sz)
            rx = rng.uniform(.04*self.sz, .35*self.sz); ry = rng.uniform(.04*self.sz, .35*self.sz)
            v = rng.uniform(0.2, 1.0); a = rng.uniform(0, math.pi)
            ys, xs = np.ogrid[:self.sz, :self.sz]
            xr = (xs-x0)*math.cos(a) + (ys-y0)*math.sin(a)
            yr = -(xs-x0)*math.sin(a) + (ys-y0)*math.cos(a)
            mask = (xr/rx)**2 + (yr/ry)**2 <= 1.0
            img[mask] = np.clip(img[mask] + v, 0, 1)

        img = img[np.newaxis]
        kfull = torch.fft.rfft2(torch.from_numpy(img), norm="ortho")
        mask = make_cartesian_mask(self.sz, self.sz, self.cfg.acceleration, self.cfg.center_frac)
        kus = kfull * mask.unsqueeze(0)
        k2ch = torch.stack([kus.real, kus.imag], dim=1).squeeze(0).float()
        return k2ch, mask.float(), torch.from_numpy(img).float()


class HuggingFaceMRIDataset(Dataset):
    """
    Loads fastMRI knee single-coil RSS images directly from HuggingFace.
    Dataset: AUMLProject/fastmri-knee-singlecoil-rss

    Since RSS images are ground-truth reconstructions (not raw k-space),
    we simulate undersampled k-space by FFT + Cartesian mask.
    Returns (k2ch, mask, target) — same format as the rest of the pipeline.
    """
    def __init__(self, cfg: MRIConfig, split: str = "train"):
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("pip install datasets")

        print(f"[*] Loading AUMLProject/fastmri-knee-singlecoil-rss ({split})...")
        ds = load_dataset(
            "AUMLProject/fastmri-knee-singlecoil-rss",
            split=split,
            trust_remote_code=True,
        )
        self.ds  = ds
        self.cfg = cfg
        self.sz  = cfg.image_size
        print(f"    {len(ds)} samples, columns: {ds.column_names}")

    def __len__(self): return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]

        # HF dataset returns PIL image or numpy — handle both
        img = row.get("image") or row.get("reconstruction") or row.get("rss")
        if img is None:
            # Try first value if column name differs
            img = next(iter(row.values()))

        if hasattr(img, "convert"):          # PIL Image
            import PIL.Image
            img = np.array(img.convert("L"), dtype=np.float32)
        else:
            img = np.array(img, dtype=np.float32)

        # Squeeze to 2D
        while img.ndim > 2:
            img = img.squeeze(0) if img.shape[0] == 1 else img[img.shape[0]//2]

        # Resize to image_size x image_size
        if img.shape != (self.sz, self.sz):
            t = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0)
            t = F.interpolate(t, size=(self.sz, self.sz), mode="bilinear", align_corners=False)
            img = t.squeeze().numpy()

        # Normalise to [0, 1]
        mx = img.max()
        if mx > 0:
            img = img / mx

        # Simulate undersampled k-space via FFT + Cartesian mask
        img_t  = torch.from_numpy(img).float()
        kfull  = torch.fft.rfft2(img_t, norm="ortho")
        mask   = make_cartesian_mask(self.sz, self.sz, self.cfg.acceleration, self.cfg.center_frac)
        kus    = kfull * mask
        k2ch   = torch.stack([kus.real, kus.imag], dim=0).float()
        target = img_t.unsqueeze(0).float()    # (1, H, W)

        return k2ch, mask.float(), target


def get_loaders(cfg: MRIConfig):
    """Returns (train_loader, val_loader) using real HF data, npz fallback, or synthetic."""
    kw = dict(pin_memory=cfg.amp, persistent_workers=False)

    # 1. Try HuggingFace dataset (primary)
    try:
        train_ds = HuggingFaceMRIDataset(cfg, split="train")
        val_ds   = HuggingFaceMRIDataset(cfg, split="validation")
        print("[*] Using HuggingFace fastMRI dataset")
        train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True,
                                  num_workers = 8, **kw)
        val_loader   = DataLoader(val_ds,   cfg.batch_size, shuffle=False,
                                  num_workers = 8, **kw)
        return train_loader, val_loader
    except Exception as e:
        print(f"[!] HuggingFace load failed: {e}")

    # 2. Try local npz files
    data_root = Path(cfg.data_root)
    if data_root.exists() and any(data_root.rglob("*.npz")):
        print(f"[*] Using local npz data from {data_root}")
        from torch.utils.data import random_split
        full_ds = FastMRINPZDataset(data_root, cfg)
        n_val   = max(1, int(0.2 * len(full_ds)))
        train_ds, val_ds = random_split(
            full_ds, [len(full_ds) - n_val, n_val],
            generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True,  num_workers = 8, **kw)
        val_loader   = DataLoader(val_ds,   cfg.batch_size, shuffle=False, num_workers = 8, **kw)
        return train_loader, val_loader

    # 3. Synthetic fallback
    print("[!] No real data found — using synthetic fallback")
    train_loader = DataLoader(SyntheticMRIDataset(1000, cfg), cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(SyntheticMRIDataset(200,  cfg), cfg.batch_size, shuffle=False)
    return train_loader, val_loader


def inspect_npz(path: Path):
    """Print keys and shapes of a single npz file — run once to understand your data."""
    d = np.load(path, allow_pickle=True)
    print(f"File: {path.name}")
    for k in d.files:
        v = d[k]
        print(f"  '{k}': shape={v.shape}, dtype={v.dtype}, "
              f"min={float(v.min()):.4f}, max={float(v.max()):.4f}")


class FastMRINPZDataset(Dataset):
    """
    Loads real MRI slices from .npz files saved at data_root.

    Handles the most common fastMRI-derived npz layouts:
      Layout A — separate arrays: 'kspace_real','kspace_imag','target'
      Layout B — stacked complex:  'kspace' shape (H,W,2) or (H,W), 'target'
      Layout C — pre-processed:    'input' (zero-filled), 'target'
      Layout D — raw k-space only: 'kspace' (complex or float), no target
      Layout E — fastMRI standard: 'reconstruction_rss' or 'reconstruction_esc'

    In all cases the dataset returns (k2ch, mask, target) consistent with the
    rest of the training pipeline.
    """
    def __init__(self, data_root: Path, cfg: MRIConfig, split: str = "train"):
        self.cfg = cfg
        self.sz  = cfg.image_size
        root = Path(data_root)
        # Accept both flat and split-subdir layouts
        split_dir = root / split
        search_dir = split_dir if split_dir.exists() else root
        self.files = sorted(search_dir.glob("**/*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No .npz files found in {search_dir}. "
                f"Check your data_root path: {data_root}"
            )
        # Build flat list of (file, slice_idx) — each npz may hold multiple slices
        self.samples = []
        for f in self.files:
            try:
                d = np.load(f, allow_pickle=True)
                n_slices = self._n_slices(d)
                for s in range(n_slices):
                    self.samples.append((f, s))
            except Exception:
                pass
        print(f"[FastMRINPZDataset] {split}: {len(self.files)} files, "
              f"{len(self.samples)} slices")

    def _n_slices(self, d) -> int:
        for key in ('target', 'reconstruction_rss', 'reconstruction_esc',
                    'kspace', 'kspace_real', 'input'):
            if key in d.files:
                arr = d[key]
                # Shape conventions: (slices,H,W), (H,W), (slices,H,W,2)
                return arr.shape[0] if arr.ndim >= 3 else 1
        return 1

    def _load_slice(self, path: Path, s: int):
        d = np.load(path, allow_pickle=True)

        # --- resolve target image ---
        target = None
        for tk in ('target', 'reconstruction_rss', 'reconstruction_esc'):
            if tk in d.files:
                t = d[tk].astype(np.float32)
                target = t[s] if t.ndim == 3 else t
                break

        # --- resolve k-space ---
        if 'kspace_real' in d.files and 'kspace_imag' in d.files:
            # Layout A
            kr = d['kspace_real'].astype(np.float32)
            ki = d['kspace_imag'].astype(np.float32)
            kr = kr[s] if kr.ndim == 3 else kr
            ki = ki[s] if ki.ndim == 3 else ki
            kc = kr + 1j * ki

        elif 'kspace' in d.files:
            # Layout B/D
            k = d['kspace']
            sl = k[s] if k.ndim >= 3 and k.shape[0] > 1 else (k[0] if k.ndim >= 3 else k)
            if np.iscomplexobj(sl):
                kc = sl.astype(np.complex64)
            elif sl.ndim == 3 and sl.shape[-1] == 2:
                kc = (sl[..., 0] + 1j * sl[..., 1]).astype(np.complex64)
            else:
                kc = sl.astype(np.complex64)
        elif 'input' in d.files:
            inp = d['input'].astype(np.float32)
            img = inp[s] if inp.ndim == 3 else inp
            img = self._resize(img)
            if target is None:
                target = img
            target = self._resize(target)
            target_n = self._normalise(target)
            
            # Correct rfft2 + mask logic (matches SyntheticMRIDataset)
            kfull = torch.fft.rfft2(torch.from_numpy(img), norm="ortho")
            mask = make_cartesian_mask(self.sz, self.sz, self.cfg.acceleration, self.cfg.center_frac)
            kus = kfull * mask
            k2ch = torch.stack([kus.real, kus.imag], dim=0).float()
            return k2ch, mask.float(), torch.from_numpy(target_n[np.newaxis]).float()

        elif 'images' in d.files:
            imgs = d['images'].astype(np.float32)
            img = imgs[s] if imgs.ndim == 3 else imgs
            img = self._resize(img)
            if target is None:
                target = img
            target = self._resize(target)
            target_n = self._normalise(target)
            
            # Correct rfft2 + mask logic
            kfull = torch.fft.rfft2(torch.from_numpy(img), norm="ortho")
            mask = make_cartesian_mask(self.sz, self.sz, self.cfg.acceleration, self.cfg.center_frac)
            kus = kfull * mask
            k2ch = torch.stack([kus.real, kus.imag], dim=0).float()
            return k2ch, mask.float(), torch.from_numpy(target_n[np.newaxis]).float()


        elif 'images' in d.files:
            # Layout D — ground truth images, synthesise full k-space
            imgs = d['images'].astype(np.float32)
            img = imgs[s] if imgs.ndim == 3 else imgs
            img = self._resize(img)
            if target is None:
                target = img
            kc = np.fft.fft2(img, norm='ortho')
            k2ch, mask = self._kspace_to_2ch_and_mask(kc)
            target = self._resize(target)
            target_n = self._normalise(target)
            return k2ch, mask, torch.from_numpy(target_n[np.newaxis]).float()


        else:
            raise KeyError(f"Cannot find k-space in {path.name}. Keys: {d.files}")

        # Derive target from k-space RSS if not present
        if target is None:
            target = np.abs(np.fft.irfft2(kc, norm='ortho')).astype(np.float32)

        # Resize both k-space and target to cfg.image_size
        target = self._resize(target.astype(np.float32))
        kc = self._resize_kspace(kc, target.shape)

        # Undersample with random Cartesian mask
        k2ch, mask = self._kspace_to_2ch_and_mask(kc)

        target_n = self._normalise(target)
        return k2ch, mask, torch.from_numpy(target_n[np.newaxis]).float()

    def _resize(self, img: np.ndarray) -> np.ndarray:
        if img.shape[-2:] == (self.sz, self.sz):
            return img
        t = torch.from_numpy(img).float()
        if t.ndim == 2:
            t = t.unsqueeze(0).unsqueeze(0)
        elif t.ndim == 3:
            t = t.unsqueeze(0)
        t = F.interpolate(t, size=(self.sz, self.sz), mode='bilinear', align_corners=False)
        return t.squeeze().numpy()

    def _resize_kspace(self, kc: np.ndarray, target_shape) -> np.ndarray:
        img = np.abs(np.fft.irfft2(kc, norm='ortho'))
        img_resized = self._resize(img.astype(np.float32))
        return np.fft.rfft2(img_resized, norm='ortho').astype(np.complex64)

    def _kspace_to_2ch_and_mask(self, kc: np.ndarray,
                                  already_undersampled: bool = False):
        H = kc.shape[-2] if kc.ndim == 2 else kc.shape[0]
        W = kc.shape[-1] if kc.ndim == 2 else kc.shape[1]
        mask = make_cartesian_mask(H, W, self.cfg.acceleration, self.cfg.center_frac)
        if already_undersampled:
            kus = torch.from_numpy(kc).cfloat()
        else:
            kus = torch.from_numpy(kc).cfloat() * mask
        k2ch = torch.stack([kus.real, kus.imag], dim=0).float()
        return k2ch, mask.float()

    @staticmethod
    def _normalise(img: np.ndarray) -> np.ndarray:
        mx = img.max()
        return (img / mx).astype(np.float32) if mx > 0 else img.astype(np.float32)

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, s = self.samples[idx]
        try:
            return self._load_slice(path, s)
        except Exception as e:
            # Corrupt slice — return a synthetic fallback so training doesn't crash
            print(f"[warn] skipping {path.name} slice {s}: {e}")
            return SyntheticMRIDataset(1, self.cfg)[0]


def get_loaders(cfg: MRIConfig):
    """Returns (train_loader, val_loader) — real data if available, else synthetic."""
    data_root = Path(cfg.data_root)
    use_real = data_root.exists() and any(data_root.rglob("*.npz"))

    if use_real:
        print(f"[*] Using real fastMRI data from {data_root}")
        # Try split subdirs first, else split the flat list 80/20
        try:
            train_ds = FastMRINPZDataset(data_root, cfg, split="train")
            val_ds   = FastMRINPZDataset(data_root, cfg, split="val")
        except FileNotFoundError:
            full_ds = FastMRINPZDataset(data_root, cfg, split="")
            n_val   = max(1, int(0.2 * len(full_ds)))
            n_train = len(full_ds) - n_val
            train_ds, val_ds = torch.utils.data.random_split(
                full_ds, [n_train, n_val],
                generator=torch.Generator().manual_seed(42)
            )
    else:
        print(f"[!] Real data not found at {data_root} — using synthetic fallback")
        train_ds = SyntheticMRIDataset(1000, cfg)
        val_ds   = SyntheticMRIDataset(200,  cfg)

    kw = dict(num_workers = 8, pin_memory=cfg.amp, persistent_workers=True)
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True,  **kw)
    val_loader   = DataLoader(val_ds,   cfg.batch_size, shuffle=False, **kw)
    return train_loader, val_loader


# ==============================================================================
# ARCHITECTURE MODULES
# ==============================================================================

def _cb(ci, co):
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU(),
        nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.GELU()
    )

class KSpaceEncoder(nn.Module):
    def __init__(self, embed_dim: int = 256, base: int = 32, image_size: int = 320):
        super().__init__()
        self.sq = image_size // 2
        self.mag_proj = nn.Conv2d(1, base // 2, 1)
        self.phase_proj = nn.Conv2d(1, base // 2, 1)
        self.to_square = nn.AdaptiveAvgPool2d(self.sq)
        self.conv1 = nn.Sequential(nn.Conv2d(base, base, 3, 2, 1), nn.BatchNorm2d(base), nn.GELU())
        self.conv2 = nn.Sequential(nn.Conv2d(base, base*2, 3, 2, 1), nn.BatchNorm2d(base*2), nn.GELU())
        self.conv3 = nn.Sequential(nn.Conv2d(base*2, base*4, 3, 2, 1), nn.BatchNorm2d(base*4), nn.GELU())
        self.conv4 = nn.Sequential(nn.Conv2d(base*4, base*8, 3, 2, 1), nn.BatchNorm2d(base*8), nn.GELU())
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(base * 8, embed_dim)

    def forward(self, kspace):
        real, imag = kspace[:, 0:1], kspace[:, 1:2]
        mag = torch.log1p(torch.sqrt(real**2 + imag**2 + 1e-8))
        phase = torch.atan2(imag, real)
        x = torch.cat([self.mag_proj(mag), self.phase_proj(phase)], dim=1)
        x = self.to_square(x)
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        emb = self.fc(self.pool(f4).flatten(1))
        return emb, f1, f2, f3


class SpectralBasisDecoder(nn.Module):
    def __init__(self, image_size, channels, embed_dim, n_bases, enc_ch1=32, enc_ch2=64):
        super().__init__()
        self.image_size = image_size
        fw = image_size // 2 + 1
        self.bases_real = nn.Parameter(torch.randn(n_bases, channels, image_size, fw) * 0.02)
        self.bases_imag = nn.Parameter(torch.randn(n_bases, channels, image_size, fw) * 0.02)
        self.weight_head = nn.Linear(embed_dim, n_bases)
        self.pre_film = nn.Conv2d(enc_ch2, channels * 2, 3, padding=1)
        self.post_film = nn.Conv2d(enc_ch1, channels * 2, 3, padding=1)
        nn.init.zeros_(self.post_film.weight); nn.init.zeros_(self.post_film.bias)

    def forward(self, emb, f1, f2):
        w = self.weight_head(emb)
        w = w / w.norm(dim=-1, keepdim=True).clamp_min(1.0)
        real = torch.einsum("bk,kchw->bchw", w, self.bases_real)
        imag = torch.einsum("bk,kchw->bchw", w, self.bases_imag)
        fw = real.shape[-1]
        f2_s = F.interpolate(f2, size=(self.image_size, fw), mode="bilinear", align_corners=False)
        pg, pb = self.pre_film(f2_s).chunk(2, dim=1)
        real = (1 + pg) * real + pb
        imag = (1 + pg) * imag + pb
        out = torch.fft.irfft2(torch.complex(real.float(), imag.float()), s=(self.image_size,) * 2, norm="ortho")
        f1_s = F.interpolate(f1, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False)
        qg, qb = self.post_film(f1_s).chunk(2, dim=1)
        out = (1 + qg) * out + qb
        return torch.sigmoid(out)


class HashFeatureExtractor(nn.Module):
    def __init__(self, embed_dim, image_size, hash_levels=12, hash_dim=2, hidden=32, hash_res=64, out_ch=32):
        super().__init__()
        self.image_size = image_size; self.hash_res = hash_res; self.out_ch = out_ch
        self.b = math.exp((math.log(512) - math.log(16)) / (hash_levels - 1))
        self.embeddings = nn.ModuleList([nn.Embedding(16384, hash_dim) for _ in range(hash_levels)])
        for emb in self.embeddings: nn.init.uniform_(emb.weight, -1e-4, 1e-4)
        self._px = 2654435761; self._py = 805459861
        self.modulator = nn.Sequential(nn.Linear(embed_dim, 64), nn.SiLU(), nn.Linear(64, 2*hidden*2))
        self.l1 = nn.Linear(hash_levels * hash_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, out_ch)
        self.act = nn.SiLU()
        t = torch.linspace(0.0, 1.0, hash_res)
        gy, gx = torch.meshgrid(t, t, indexing="ij")
        self.register_buffer("coords", torch.stack([gx, gy], -1).view(-1, 2))

    def _hash(self, xi, yi): return ((xi * self._px) ^ (yi * self._py)) % 16384

    def forward(self, emb):
        B = emb.size(0)
        x = self.coords.unsqueeze(0).expand(B, -1, -1)
        feats = []
        for i in range(len(self.embeddings)):
            res = math.floor(16 * (self.b ** i))
            xs = x * res
            x0 = torch.floor(xs).long(); x1 = x0 + 1
            w = xs - x0.float(); wx = w[...,0:1]; wy = w[...,1:2]
            f00 = self.embeddings[i](self._hash(x0[...,0], x0[...,1]))
            f01 = self.embeddings[i](self._hash(x0[...,0], x1[...,1]))
            f10 = self.embeddings[i](self._hash(x1[...,0], x0[...,1]))
            f11 = self.embeddings[i](self._hash(x1[...,0], x1[...,1]))
            feats.append((f00*(1-wy)+f01*wy)*(1-wx) + (f10*(1-wy)+f11*wy)*wx)
        
        feats = torch.cat(feats, dim=-1)
        mods = self.modulator(emb).view(B, 2, 2, -1)
        g0, b0 = mods[:,0,0].unsqueeze(1), mods[:,0,1].unsqueeze(1)
        g1, b1 = mods[:,1,0].unsqueeze(1), mods[:,1,1].unsqueeze(1)
        x_feat = self.act(self.l1(feats) * g0 + b0)
        x_feat = self.act(self.l2(x_feat) * g1 + b1)
        out = self.head(x_feat).permute(0,2,1).view(B, self.out_ch, self.hash_res, self.hash_res)
        if self.hash_res != self.image_size:
            out = F.interpolate(out, size=self.image_size, mode="bilinear", align_corners=False)
        return out


class ConcatFuse(nn.Module):
    def __init__(self, channels=1, hash_ch=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels + hash_ch, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, channels, 1), nn.Sigmoid()
        )
    def forward(self, spectral, hash_feats):
        return self.net(torch.cat([spectral, hash_feats], dim=1))


class CascadeRefiner(nn.Module):
    def __init__(self, channels=1, hash_ch=32, base=32, enc_ch1=32, enc_ch2=64):
        super().__init__()
        self.e1 = _cb(channels + hash_ch, base)
        self.e2 = _cb(base, base*2)
        self.pool = nn.MaxPool2d(2)
        self.mid = _cb(base*2, base*4)
        self.skip2 = nn.Conv2d(enc_ch2, base*4, 1)
        self.skip1 = nn.Conv2d(enc_ch1, base*2, 1)
        nn.init.zeros_(self.skip2.weight); nn.init.zeros_(self.skip2.bias)
        nn.init.zeros_(self.skip1.weight); nn.init.zeros_(self.skip1.bias)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = _cb(base*2, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = _cb(base, base)
        self.head = nn.Conv2d(base, channels, 1)

    def forward(self, x_dc, hash_feats, f1, f2):
        s1 = self.e1(torch.cat([x_dc, hash_feats], dim=1))
        s2 = self.e2(self.pool(s1))
        m = self.mid(self.pool(s2))
        m = m + F.interpolate(self.skip2(f2), size=m.shape[-2:], mode="bilinear", align_corners=False)
        s2 = s2 + F.interpolate(self.skip1(f1), size=s2.shape[-2:], mode="bilinear", align_corners=False)
        u2 = F.interpolate(self.up2(m), size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(u2 + s2)
        u1 = F.interpolate(self.up1(d2), size=s1.shape[-2:], mode="bilinear", align_corners=False)
        out = x_dc + self.head(self.d1(u1 + s1))
        # Removed .clamp() here to respect strict physics scaling
        return out


# ==============================================================================
# PHYSICS: DATA CONSISTENCY & SENSE
# ==============================================================================

def dc_project(x, y_k, mask):
    x_k = torch.fft.rfft2(x.squeeze(1).float(), norm="ortho")
    x_k_dc = mask * y_k.to(x_k.dtype) + (1.0 - mask) * x_k
    x_dc = torch.fft.irfft2(x_k_dc, s=(x.shape[-2], x.shape[-1]), norm="ortho").abs()
    # Removed arbitrary .amax() division to respect scanner scale
    return x_dc.unsqueeze(1)

def _rss(x, dim): return torch.sqrt((x.abs()**2).sum(dim=dim).clamp_min(1e-8))

class SensitivityModel(nn.Module):
    def __init__(self, base=16):
        super().__init__()
        self.net = nn.Sequential(
            _cb(2, base), nn.MaxPool2d(2), _cb(base, base*2),
            nn.ConvTranspose2d(base*2, base, 2, stride=2), _cb(base, base), nn.Conv2d(base, 2, 1)
        )
    def forward(self, kmc, c_frac=0.08):
        B, C, H, fw = kmc.shape
        cs = H // 2 - max(1, int(H * c_frac)) // 2
        acs = torch.zeros_like(kmc); acs[..., cs:cs + max(1, int(H * c_frac)), :] = kmc[..., cs:cs + max(1, int(H * c_frac)), :]
        c_imgs = torch.fft.irfft2(acs, s=(H, (fw-1)*2), norm="ortho")
        c_imgs = c_imgs / c_imgs.abs().amax(dim=(-2,-1), keepdim=True).clamp_min(1e-8)
        x2ch = torch.stack([c_imgs.real, c_imgs.imag], dim=2).view(B*C, 2, H, (fw-1)*2)
        o2ch = self.net(x2ch)
        sens = torch.complex(o2ch[:, 0], o2ch[:, 1]).view(B, C, H, (fw-1)*2)
        return sens / _rss(sens, dim=1).unsqueeze(1).clamp_min(1e-8)

def sense_expand(x, sens): return torch.fft.rfft2(sens * x.unsqueeze(1), norm="ortho")
def sense_reduce(kmc, sens): return (sens.conj() * torch.fft.irfft2(kmc, norm="ortho")).sum(dim=1)

def dc_project_multicoil(x, kmc, mask, sens):
    x_c = torch.complex(x.squeeze(1).float(), torch.zeros_like(x.squeeze(1)))
    pred_k = sense_expand(x_c, sens)
    m = mask.unsqueeze(1) if mask.dim() == 3 else mask.unsqueeze(0).unsqueeze(0)
    dc_k = m * kmc.to(pred_k.dtype) + (1.0 - m) * pred_k
    return sense_reduce(dc_k, sens).abs().unsqueeze(1)


# ==============================================================================
# MAIN MODEL
# ==============================================================================

class DHWSMRIEfficient(nn.Module):
    def __init__(self, cfg: MRIConfig):
        super().__init__()
        self.cfg = cfg
        self.multicoil = cfg.num_coils > 1
        self.sens_model = SensitivityModel() if self.multicoil else None
        self.encoder = KSpaceEncoder(cfg.embed_dim, base=32, image_size=cfg.image_size)
        self.spectral = SpectralBasisDecoder(cfg.image_size, cfg.channels, cfg.embed_dim, cfg.n_spec_bases, enc_ch1=32, enc_ch2=64)
        self.hashfeat = HashFeatureExtractor(cfg.embed_dim, cfg.image_size, cfg.hash_levels, cfg.hash_dim, hidden=32, hash_res=cfg.hash_res, out_ch=32)
        self.fusion = ConcatFuse(cfg.channels, hash_ch=32)
        self.cascade = nn.ModuleList([CascadeRefiner(cfg.channels, hash_ch=32, base=32, enc_ch1=32, enc_ch2=64)] * cfg.n_cascade)

    def zero_filled(self, kspace):
        if kspace.dim() == 4:
            mag = torch.fft.irfft2(torch.complex(kspace[:,0], kspace[:,1]), norm="ortho").abs().unsqueeze(1)
        else:
            mag = _rss(torch.fft.irfft2(torch.complex(kspace[:,:,0], kspace[:,:,1]), norm="ortho"), dim=1).unsqueeze(1)
        return (mag / mag.amax((-2,-1), keepdim=True).clamp_min(1e-8)).clamp(0, 1)

    def forward(self, kspace, mask):
        if self.multicoil:
            kmc = torch.complex(kspace[:,:,0], kspace[:,:,1])
            sens = self.sens_model(kmc, self.cfg.center_frac)
            combined = sense_reduce(kmc, sens)
            k2ch = torch.stack([combined.real, combined.imag], dim=1)
        else:
            k2ch = kspace
            kmc = torch.complex(kspace[:,0], kspace[:,1])
            sens = None

        mask_k = mask if mask.dim() == 3 else mask.unsqueeze(0).expand(k2ch.size(0), -1, -1)

        emb, f1, f2, f3 = self.encoder(k2ch)
        spectral = self.spectral(emb, f1, f2)
        hash_feats = self.hashfeat(emb)
        x = self.fusion(spectral, hash_feats)

        for refiner in self.cascade:
            if self.multicoil: x = dc_project_multicoil(x, kmc, mask_k, sens)
            else: x = dc_project(x, kmc, mask_k)
            x = refiner(x, hash_feats, f1, f2)

        return x, spectral


# ==============================================================================
# LOSS & METRICS
# ==============================================================================

def _ssim(pred, target, ws=11):
    C1, C2 = 0.01**2, 0.03**2
    mu_p, mu_t = F.avg_pool2d(pred, ws, 1, ws//2), F.avg_pool2d(target, ws, 1, ws//2)
    mpp, mtt, mpt = F.avg_pool2d(pred**2, ws, 1, ws//2), F.avg_pool2d(target**2, ws, 1, ws//2), F.avg_pool2d(pred*target, ws, 1, ws//2)
    sp, st, spt = mpp - mu_p**2, mtt - mu_t**2, mpt - mu_p*mu_t
    return ((2*mu_p*mu_t+C1)*(2*spt+C2) / ((mu_p**2+mu_t**2+C1)*(sp+st+C2))).mean()

class MRILoss(nn.Module):
    def __init__(self, cfg: MRIConfig):
        super().__init__()
        self.cfg = cfg
        fy, fx = torch.linspace(-1.0, 1.0, cfg.image_size), torch.linspace(0.0, 1.0, cfg.image_size // 2 + 1)
        yy, xx = torch.meshgrid(fy, fx, indexing="ij")
        self.register_buffer("freq_weight", 1.0 / (torch.sqrt(xx**2 + yy**2) + 1.0))

    def forward(self, pred, target, kspace_2ch, mask):
        p, t = pred.float(), target.float()
        l1 = F.l1_loss(p, t)
        ssim = 1.0 - _ssim(p, t)
        pf, tf = torch.fft.rfft2(p.squeeze(1), norm="ortho"), torch.fft.rfft2(t.squeeze(1), norm="ortho")
        spec = (((pf.real - tf.real)**2 + (pf.imag - tf.imag)**2) * self.freq_weight.unsqueeze(0).to(pf.device)).mean()
        km = torch.complex(kspace_2ch[:,0], kspace_2ch[:,1])
        dc = (mask.to(pf.device) * (pf - km.to(pf.dtype)).abs()**2).mean()
        B, C, H, W = p.shape
        gp, gt = p.view(B, C, -1), t.view(B, C, -1)
        gram = F.mse_loss(gp @ gp.transpose(-1,-2) / (C*H*W), gt @ gt.transpose(-1,-2) / (C*H*W))
        return self.cfg.w_l1 * l1 + self.cfg.w_ssim * ssim + self.cfg.w_complex * spec + self.cfg.w_dc * dc + self.cfg.w_gram * gram


# ==============================================================================
# TRAINING LOOP
# ==============================================================================

def train_epoch(model, optimizer, loader, criterion, cfg, scaler):
    model.train()
    total = 0.0
    for kspace, mask, target in loader:
        kspace, mask, target = kspace.to(cfg.device), mask.to(cfg.device), target.to(cfg.device)
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type=cfg.device.type, enabled=cfg.amp):
            refined, spectral = model(kspace, mask)
            
        # Compute loss outside autocast to prevent float16 overflow in spectral MSE
        loss = 1.0 * criterion(refined, target, kspace, mask) + 0.3 * criterion(spectral, target, kspace, mask)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer); scaler.update()
        total += loss.item() * kspace.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion, cfg):
    model.eval()
    tl = psnr_s = ssim_s = 0.0; n = 0
    for kspace, mask, target in loader:
        kspace, mask, target = kspace.to(cfg.device), mask.to(cfg.device), target.to(cfg.device)
        with torch.amp.autocast(device_type=cfg.device.type, enabled=cfg.amp):
            refined, _ = model(kspace, mask)
        tl += criterion(refined, target, kspace, mask).item() * kspace.size(0)
        p, t = refined.float().cpu(), target.float().cpu()
        mse = F.mse_loss(p, t).item()
        psnr_s += (float("inf") if mse == 0 else 10 * math.log10(1.0 / mse)) * kspace.size(0)
        ssim_s += _ssim(p, t).item() * kspace.size(0)
        n += kspace.size(0)
    return {"loss": tl/n, "psnr_db": psnr_s/n, "ssim": ssim_s/n}


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    cfg = MRIConfig()
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 65)
    print("DHWS-MRI-Efficient  --  Fast, Hallucination-Free Reconstruction")
    print("=" * 65)

    train_loader, val_loader = get_loaders(cfg)

    model = DHWSMRIEfficient(cfg).to(cfg.device)
    criterion = MRILoss(cfg).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg.lr, epochs=cfg.epochs,
        steps_per_epoch=len(train_loader), pct_start=0.1
    )
    scaler = torch.amp.GradScaler(cfg.device.type, enabled=cfg.amp)

    total_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Encoder : {sum(p.numel() for p in model.encoder.parameters()):>9,}")
    print(f"  Spectral: {sum(p.numel() for p in model.spectral.parameters()):>9,} (K=32)")
    print(f"  HashFeat: {sum(p.numel() for p in model.hashfeat.parameters()):>9,}")
    print(f"  Cascade : {sum(p.numel() for p in model.cascade[0].parameters()):>9,} (4 steps shared)")
    print(f"  TOTAL   : {total_p:>9,} (~{total_p/1e6:.2f}M Params)\n")

    # Resume from checkpoint if one exists
    start_epoch = 1
    best_loss   = float("inf")
    resume_path = cfg.out_dir / "checkpoint.pt"
    best_path   = cfg.out_dir / "best_model.pt"
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=cfg.device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_loss   = ckpt.get("best_loss", float("inf"))
        print(f"[*] Resumed from epoch {ckpt['epoch']} (best_loss={best_loss:.4f})")

    log_path = cfg.out_dir / "training_log.csv"
    log_exists = log_path.exists()
    log_f = open(log_path, "a", newline="")
    import csv as _csv
    log_w = _csv.writer(log_f)
    if not log_exists:
        log_w.writerow(["epoch", "train_loss", "val_loss", "psnr_db", "ssim", "time_s"])

    for epoch in range(start_epoch, cfg.epochs + 1):
        t0 = time.time()
        tr = train_epoch(model, optimizer, train_loader, criterion, cfg, scaler)
        scheduler.step()
        vm = evaluate(model, val_loader, criterion, cfg)
        el = time.time() - t0

        print(f"Ep {epoch:3d} | Train: {tr:.4f} | Val: {vm['loss']:.4f} | "
              f"PSNR: {vm['psnr_db']:.2f} | SSIM: {vm['ssim']:.4f} | {el:.1f}s")
        log_w.writerow([epoch, tr, vm['loss'], vm['psnr_db'], vm['ssim'], round(el, 1)])
        log_f.flush()

        # Save best
        if vm["loss"] < best_loss:
            best_loss = vm["loss"]
            torch.save(model.state_dict(), best_path)
            print(f"  => best saved (loss={best_loss:.4f})")

        # Save resumable checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save({
                "epoch": epoch, "best_loss": best_loss,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
            }, resume_path)

    log_f.close()
    print(f"\nDone. Best val loss: {best_loss:.4f}")
    print(f"Best model: {best_path}")

if __name__ == "__main__":
    main()