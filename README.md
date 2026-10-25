# mind-diffusion

A research project that uses **diffusion-model AI techniques** to simulate a
digital "mind" by learning the distribution of **neuron / brain-slice images**
and then walking through the learned latent space.

## Pipeline

1. **VAE** — compresses 64×64 brain images into an 8×8×4 latent grid ("mind code")
2. **Conditional latent DDPM/DDIM** — learns to generate neural configurations,
   with classifier-free guidance conditioned on patient/source identity
3. **Simulation** — a "thought" is a trajectory through latent space,
   decoded frame-by-frame into an animated GIF

## What's new in v2 (R&D)

| Feature | v1 | v2 |
|---------|----|----|
| Data | 1 source, 200 images | 3 sources, 3.8k images (real brain MRI) |
| Conditioning | none | classifier-free guidance (54 classes) |
| Sampler | DDPM only | DDPM + DDIM (deterministic, fewer steps) |
| Trajectory modes | random walk | random_walk, ddim_interpolate, **connectome**, **ode** |
| Splits | none | train/val (90/10) |
| Augmentation | none | flips + rotations |
| Analysis | none | latent PCA/t-SNE, reconstructions, trajectory plots |
| GPU | CPU only | DirectML (AMD Radeon iGPU) |

### Trajectory modes

- **random_walk** — Gaussian perturbation + manifold pull-back (v1 behavior)
- **ddim_interpolate** — slerp between two sampled latents (smoothest transition)
- **connectome** — drive the latent walk with a real/synthetic neural
  connectome adjacency matrix, projected to the latent dimension via SVD.
  Models a "thought" as activation propagating through a network.
- **ode** — integrate a score-based latent flow ODE for smoothest path

## Quick start

```powershell
pip install -r requirements.txt
pip install torch-directml   # AMD/Intel GPU (optional)

# prepare data + connectome
python scripts/prepare_data.py --force
python scripts/make_connectome.py --nodes 256

# train (VAE on GPU, diffusion on CPU - see benchmark below)
python scripts/train_vae.py --device directml --epochs 30
python scripts/train_diffusion.py --device cpu --epochs 30

# simulate all 4 modes
python scripts/run_simulation.py --mode ddim_interpolate --steps 60
python scripts/run_simulation.py --mode connectome --steps 60 --connectome data/connectome.npy

# analysis plots
python scripts/analyze.py --device cpu
```

## Results

- **VAE**: PSNR ~21.7 dB on validation reconstructions
- **Diffusion**: loss converges ~0.21 after 30 epochs (B=64, 1000 timesteps)
- **DDIM sampling**: 50 steps, deterministic, matches data latent distribution

### Hardware benchmark (AMD Radeon iGPU via DirectML)

| Metric | CPU | DirectML | Speedup |
|--------|-----|----------|---------|
| VAE step (B=16) | 6234 ms | 1241 ms | **5.02x** |
| Diff step (B=16) | 360 ms | 532 ms | 0.68x |
| Sample (50 steps) | 1847 ms | 2705 ms | 0.68x |

**Finding**: DirectML excels at conv-heavy VAE but is slower for the UNet
(GroupNorm/attention backward falls back to CPU, adding overhead).
Strategy: train VAE on DirectML, diffusion on CPU.

## Data sources

| Source | Images | Type |
|--------|--------|------|
| Brain-MRI-LGG-Segmentation (HF) | 1873 | T1w brain slices, 52 patient labels |
| g4m3r/T1w_MRI_Brain_Slices (HF) | ~1 | T1w axial slices |
| Procedural generator | 2000 | synthetic neuron morphology |

## Project layout

```
src/mind_diffusion/
  config.py        # all hyperparameters
  device.py        # DirectML -> CUDA -> CPU selection
  data.py          # multi-source, splits, augmentation, conditioning
  models.py        # VAE, ConditionalUNet, MindDiffusion (DDPM/DDIM/CFG)
  simulation.py    # 4 trajectory modes + GIF/montage output
scripts/
  prepare_data.py, make_connectome.py, train_vae.py,
  train_diffusion.py, run_simulation.py, analyze.py, run_all.py
tests/
  smoke.py, benchmark_device.py
```

## Disclaimer

Exploratory art/research project. Does not model real neurobiology or
cognition; it learns image distributions and samples smooth trajectories
through them. Treat outputs as visualizations, not scientific predictions.