# DHWS-MRI-Recon

**Accelerated MRI reconstruction from undersampled k-space — physics-grounded, reduced hallucinations, fastMRI-competitive.**

DHWS-MRI is a k-space-native deep learning framework that reconstructs MRI images from undersampled Fourier measurements without adversarial loss. It combines spectral basis decoding, hash-grid detail recovery, and unrolled data-consistency cascades to achieve reconstruction quality competitive with E2E-VarNet on the fastMRI benchmark.

---

## Why DHWS over standard U-Nets?

MRI scanners capture k-space (Fourier domain), not pixels. Standard U-Nets work in pixel space and must learn the Fourier relationship indirectly. DHWS speaks k-space natively:

| Component | Role |
|---|---|
| `KSpaceEncoder` | Operates directly in Fourier domain (2ch complex input) |
| `SpectralBasisDecoder` | Learns complex spectral bases → irfft2 → image (same op as MRI recon) |
| `HashGridDetail` | Recovers fine detail lost from undersampling |
| `HarmonicGeometry` | Captures smooth anatomical priors (MRI has smooth structure) |
| `DataConsistency` | Hard projection: output k-space must agree with scanner measurements |

No GAN. Significantly reduced hallucinations. Physics first.

---

## Repository Structure

```
dhws_mri.py                                  # Base DHWS-MRI architecture
dhws_mri_best.py                             # DHWS-MRI-Best: 8-step unrolled DC cascade
dhws_mri_efficient.py                        # DHWS-MRI-Efficient: A100-optimised, 10-step cascade
dhws_mri_efficient_corrected for npz data and colab.py   # Colab + .npz data variant
colab_mri_best.ipynb                         # Colab notebook — Best model
colab_mri_efficient.ipynb                    # Colab notebook — Efficient model
copy_of_colab_mri_efficient.ipynb            # Colab notebook — Efficient (copy/backup)
dhws_mri_eficient/
  dhws_mri_efficient.py                      # Standalone efficient variant
  dhws_mri_efficient_training_log.txt        # Training log (100+ epochs, real run)
```

---

## Model Variants

### `dhws_mri.py` — Base
Single-pass reconstruction. Fast, ~10M params, 1 forward pass. Good baseline.

### `dhws_mri_best.py` — Best
- `embed_dim=512`, `n_spec_bases=96`, `n_cascade=8` (shared weights)
- Expected: PSNR ~38–42 dB, SSIM ~0.93+ on fastMRI 4× (GPU, 50 epochs)
- Beats zero-filled baseline (PSNR ~26–28 dB) from epoch 1

### `dhws_mri_efficient.py` — Efficient (A100-Optimised)
- `embed_dim=1024`, `n_spec_bases=256`, `n_cascade=10`
- TF32 enabled for Tensor Core acceleration
- `~4.64M` params in reduced config; scales to full A100 VRAM
- Auto-resumes from `checkpoint.pt` on Colab disconnect

---

## Quickstart

### Requirements

```bash
pip install torch torchvision numpy
pip install huggingface_hub          # for auto-download of fastMRI data
```

### Run on Synthetic Phantoms (no data download needed)

```python
# Shepp-Logan ellipse phantoms — instant start, no registration
python dhws_mri_best.py
```

The script auto-generates synthetic MRI-like phantoms and trains from scratch.
Expected on CPU: PSNR ~28–30 dB, SSIM ~0.85 (converges fast, simple structure).

### Run on fastMRI Knee Singlecoil

Data is auto-downloaded from HuggingFace on first run:

```python
python dhws_mri_best.py   # auto-fetches AUMLProject/fastmri-knee-singlecoil-rss
```

Or download manually (free registration) at [fastmri.org](https://fastmri.org/) and set `cfg.data_root` to your local folder.

### Colab

Open `colab_mri_best.ipynb` or `colab_mri_efficient.ipynb` in Google Colab. The efficient notebook is tuned for A100 (Colab Pro+) and auto-resumes after disconnection.

---

## Architecture Details

### Undersampling

1D Cartesian mask with configurable acceleration factor and centre-fraction:

```
acceleration = 4     # 4× or 8× accelerated acquisition
center_frac  = 0.08  # always sample centre 8% of k-space lines
```

### Loss Function

```
L = w_l1 * L1(pred, target)
  + w_ssim * (1 - SSIM)
  + w_complex * MSE(fft(pred), fft(target))   # spectral supervision
  + w_dc * DC_loss                             # data consistency — highest weight
```

No adversarial loss. No perceptual loss. Hallucinations significantly reduced via hard data-consistency projection.

### Unrolled Cascade (Best / Efficient variants)

```
initial_estimate = SpectralDecoder(KSpaceEncoder(kspace_undersampled))
for step in range(n_cascade):
    estimate = HardDCProjection(estimate, kspace_undersampled, mask)
    estimate = Refiner(estimate)    # lightweight U-Net
output = estimate
```

This matches the principle behind E2E-VarNet (Sriram et al., 2020) while using the DHWS spectral backbone instead of sensitivity-weighted coil combination.

---

## Expected Results

| Setting | PSNR | SSIM |
|---|---|---|
| Zero-filled baseline | ~26–28 dB | ~0.70 |
| DHWS-MRI Base (single-pass) | ~34–38 dB | ~0.88 |
| DHWS-MRI-Best (8-step cascade, 50 ep GPU) | ~38–42 dB | ~0.93+ |
| E2E-VarNet (reference) | ~42 dB | ~0.94 |

Synthetic phantom runs converge faster and serve as a zero-setup sanity check.

> **Note:** Results above are based on synthetic phantoms and limited real-data runs.
> Rigorous validation on the full fastMRI knee/brain benchmark with real scanner data is ongoing.
> Contributions and independent benchmarking are welcome — see [Collaborators Welcome](#collaborators-welcome).

---

## Multi-Coil Support

`dhws_mri_efficient.py` includes an experimental SENSE-based `SensitivityModel` for multi-coil
acquisition. It is included in the architecture but currently defaults to single-coil (`num_coils=1`).

To enable multi-coil mode, set in `MRIConfig`:

```python
num_coils = 8   # or however many coils your scanner uses
```

Multi-coil SENSE support has not yet been validated against real multi-coil fastMRI data.
Collaboration with MRI physicists or access to multi-coil datasets would help advance this.

---

## Configuration

All hyperparameters live in `MRIConfig` (dataclass at the top of each script):

```python
@dataclass
class MRIConfig:
    image_size:   int   = 320       # fastMRI knee standard
    acceleration: int   = 4         # undersampling factor
    n_cascade:    int   = 8         # unrolled DC steps
    embed_dim:    int   = 512       # spectral decoder width
    n_spec_bases: int   = 96        # spectral dictionary size
    epochs:       int   = 50
    lr:           float = 3e-4
    ...
```

---

## License

[PolyForm Noncommercial License 1.0.0](LICENSE) — free for research, academic, and personal use. Commercial use requires a separate agreement.

---

## Citation

If you use this work in research, please cite:

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

## Collaborators Welcome

This is an independent research project and I am actively looking for collaborators in:

- **MRI physics / acquisition** — help validate multi-coil SENSE and real k-space data pipelines
- **Clinical / radiological** — perceptual quality evaluation and clinical relevance of reconstructions
- **Deep learning for medical imaging** — benchmarking, ablation studies, architecture improvements
- **Access to fastMRI or similar scanner datasets** — real data validation beyond synthetic phantoms

If you are interested in collaborating, benchmarking, or integrating DHWS-MRI into your pipeline,
please get in touch.

---

## Contact

Dr. Linga Murthy Narlagiri — lingamraju26@gmail.com

---

## Acknowledgements

- [fastMRI](https://fastmri.org/) — NYU / Meta AI benchmark dataset
- [HuggingFace](https://huggingface.co/datasets/AUMLProject/fastmri-knee-singlecoil-rss) — fastMRI knee singlecoil dataset hosting (AUMLProject/fastmri-knee-singlecoil-rss)
- [E2E-VarNet](https://arxiv.org/abs/2004.06688) — Sriram et al., 2020 (cascade design inspiration)
