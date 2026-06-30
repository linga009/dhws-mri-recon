# DHWS-MRI-Recon

### K-Space Native MRI Reconstruction — Spectral Bases + Hash Grids + Unrolled Data Consistency

> **To our knowledge, no prior work combines Instant-NGP-style hash-grid detail recovery with
> spectral basis decoding for accelerated MRI reconstruction.** This project explores that gap.

---

## The Problem with Current Deep Learning MRI

Most deep learning MRI reconstruction methods — including the widely-used U-Net — work in
**pixel space**. They take an aliased image as input and try to learn the dealiasing transform.

This is fundamentally indirect. The MRI scanner doesn't produce pixels — it produces
**k-space** (raw Fourier measurements). Every pixel-space method must implicitly re-learn
the Fourier relationship from data, wasting model capacity and introducing opportunities
for hallucination.

**DHWS-MRI speaks k-space natively.** Its core operations — `irfft2`, spectral basis
decomposition, complex-domain supervision — are the same mathematics the scanner hardware
uses. The network isn't approximating physics; it's implementing it.

---

## What Makes This Different

| | Standard U-Net | E2E-VarNet | **DHWS-MRI** |
|---|---|---|---|
| Input domain | Pixel (aliased) | K-space | **K-space** |
| Reconstruction op | Learned | Sensitivity-weighted | **Spectral basis + irfft2** |
| Detail recovery | Convolutional | Convolutional | **Hash-grid (Instant-NGP style)** |
| Data consistency | Soft (loss term) | Unrolled | **Unrolled hard projection** |
| Adversarial loss | Sometimes | No | **No** |
| Params (full model) | ~10–30M | ~30M | **~8.6M (Best) / 4.6M (Efficient)** |

The hash-grid detail module (`HashGridDetail`) is borrowed from neural rendering (Instant-NGP)
and repurposed here to recover high-frequency anatomical detail lost during undersampling —
a transfer that, to our knowledge, has not been attempted before for MRI.

---

## Results

| Model | PSNR | SSIM | Setting |
|---|---|---|---|
| Zero-filled (no learning) | ~26–28 dB | ~0.70 | fastMRI 4× |
| DHWS-MRI Base | ~34–38 dB | ~0.88 | Single-pass, 1 forward |
| **DHWS-MRI Best** | **~38–42 dB** | **~0.93+** | 8-step cascade, 50 ep GPU |
| E2E-VarNet (reference) | ~42 dB | ~0.94 | Published benchmark |

> **Honest status:** These figures are from synthetic phantom runs and limited real-data
> experiments. Full validation on the fastMRI knee/brain benchmark is in progress.
> Independent benchmarking and collaboration are actively invited.

---

## Quickstart

```bash
pip install torch numpy huggingface_hub

# Runs on synthetic phantoms immediately — no data download, no registration
python dhws_mri_best.py
```

Real fastMRI data auto-downloads on first run via HuggingFace. Or register free at
[fastmri.org](https://fastmri.org/) and point `cfg.data_root` to your local folder.

**Colab:** Open `colab_mri_best.ipynb` (Best model) or `colab_mri_efficient.ipynb`
(A100-optimised, auto-resumes on disconnect).

---

## How It Works

```
Undersampled k-space  (B, 2, H, W//2+1)   ← real + imag as 2 channels
         │
    KSpaceEncoder
    ├── Magnitude branch  |real + i·imag|
    └── Phase branch      atan2(imag, real)
         │
    emb (512-dim)  +  f1 spatial maps
         │
    ┌────┴──────────────────────────┐
    │  SpectralBasisDecoder          │  ← learns Fourier bases, irfft2 → image
    │  HashGridDetail                │  ← Instant-NGP hash grid for fine texture
    │  HarmonicGeometry              │  ← smooth anatomical priors
    └────────────────────────────────┘
         │
    for step in range(n_cascade):       ← 8 steps (Best) or 10 (Efficient)
        HardDCProjection                ← force k-space to match scanner measurements
        Refiner (lightweight U-Net)
         │
    Reconstructed MRI  (B, 1, H, W)
```

**Loss:**
```
L = 0.5·L1  +  0.5·SSIM  +  0.5·ComplexSpectralMSE  +  1.0·DataConsistency
```

Data consistency carries the highest weight — the scanner measurements are ground truth
and the network is not allowed to contradict them.

---

## Multi-Coil (Experimental)

`dhws_mri_efficient.py` includes a SENSE-based `SensitivityModel` for multi-coil acquisition.
Currently defaults to `num_coils=1`. To enable:

```python
num_coils = 8   # match your scanner's coil count
```

Multi-coil mode has not yet been validated on real multi-coil data.
If you have access to multi-coil fastMRI acquisitions, please get in touch.

---

## Files

```
dhws_mri_best.py                              Best model — start here
dhws_mri_efficient.py                         A100-optimised, 10-step cascade
dhws_mri.py                                   Base single-pass architecture
dhws_mri_efficient_corrected for npz...py     Colab + .npz data variant
colab_mri_best.ipynb                          Colab notebook
colab_mri_efficient.ipynb                     Colab notebook (A100)
```

---

## Status & Roadmap

- [x] Single-coil reconstruction — implemented and training
- [x] Unrolled data-consistency cascade (8 and 10 steps)
- [x] Synthetic phantom validation
- [x] Multi-coil SENSE architecture (experimental)
- [ ] Full fastMRI knee benchmark validation
- [ ] fastMRI brain benchmark
- [ ] Multi-coil real-data validation
- [ ] Comparison with VarNet, E2E-VarNet on held-out test set
- [ ] Preprint / paper

---

## Collaborators Welcome

Independent research project. Looking for collaborators in:

- **MRI physics / acquisition** — real k-space data, multi-coil pipelines, scanner access
- **Clinical radiology** — perceptual quality evaluation, diagnostic relevance
- **Medical imaging ML** — benchmarking, ablations, integration into existing pipelines

Get in touch: **lingamraju26@gmail.com**

---

## Citation

```bibtex
@misc{narlagiri2026dhwsmri,
  author    = {Narlagiri, Dr. Linga Murthy},
  title     = {DHWS-MRI-Recon: K-Space Native Accelerated MRI Reconstruction},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/linga009/dhws-mri-recon}
}
```

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for research, academic, and personal use.
Commercial use requires a separate agreement.

---

## Acknowledgements

- [fastMRI](https://fastmri.org/) — NYU / Meta AI benchmark dataset
- [HuggingFace](https://huggingface.co/datasets/AUMLProject/fastmri-knee-singlecoil-rss) — fastMRI knee singlecoil dataset hosting
- [E2E-VarNet](https://arxiv.org/abs/2004.06688) — Sriram et al., 2020 (cascade design inspiration)
- [Instant-NGP](https://arxiv.org/abs/2201.05989) — Müller et al., 2022 (hash-grid concept)

---

**Dr. Linga Murthy Narlagiri** — lingamraju26@gmail.com
