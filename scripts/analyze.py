"""Analysis & visualization for mind-diffusion.

Produces:
  * latent space scatter (PCA + t-SNE if available) of real images colored by class
  * VAE reconstruction montage (original | reconstruction pairs)
  * diffusion samples grid
  * trajectory latent-path plot (PCA projection of a simulation's latents)

Usage:
    python scripts/analyze.py [--vae vae.pt] [--diff diffusion.pt]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mind_diffusion.config import cfg
from mind_diffusion.device import get_device, set_seed
from mind_diffusion.data import build_dataloader
from mind_diffusion.models import VAE, MindDiffusion
from mind_diffusion.simulation import load_models, simulate_trajectory

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s | %(message)s")
log = logging.getLogger("analyze")


def _pca_2d(latents: np.ndarray) -> np.ndarray:
    """Simple PCA to 2D (no sklearn dependency)."""
    X = latents - latents.mean(0, keepdims=True)
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return (X @ Vt[:2].T), S[:2]


def _tsne_2d(latents: np.ndarray) -> np.ndarray:
    try:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, init="pca", perplexity=min(30, len(latents) - 1),
                    random_state=42).fit_transform(latents)
    except ImportError:
        log.info("sklearn not available, using PCA")
        return _pca_2d(latents)[0]


def plot_latent_space(cfg, vae, device, path):
    """Encode all training images and plot their latent PCA/TSNE by class."""
    loader = build_dataloader(cfg, split="val", batch_size=64)
    all_z, all_c = [], []
    vae.eval()
    with torch.no_grad():
        for batch, labels in loader:
            z = vae.encode(batch.to(device))
            all_z.append(z.cpu().numpy().reshape(len(z), -1))
            all_c.append(labels.numpy())
    Z = np.concatenate(all_z, 0)
    C = np.concatenate(all_c, 0)
    log.info("Encoded %d latents for visualization", len(Z))

    proj, _ = _pca_2d(Z)
    try:
        proj2 = _tsne_2d(Z)
        use_tsne = True
    except Exception:
        use_tsne = False

    fig, axes = plt.subplots(1, 2 if use_tsne else 1, figsize=(12, 5))
    if not use_tsne:
        axes = [axes]
    sc = axes[0].scatter(proj[:, 0], proj[:, 1], c=C, cmap="tab20", s=8, alpha=0.7)
    axes[0].set_title("Latent PCA (by class)")
    axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2")
    plt.colorbar(sc, ax=axes[0], label="class")
    if use_tsne:
        sc2 = axes[1].scatter(proj2[:, 0], proj2[:, 1], c=C, cmap="tab20", s=8, alpha=0.7)
        axes[1].set_title("Latent t-SNE (by class)")
        plt.colorbar(sc2, ax=axes[1], label="class")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    log.info("saved latent space plot -> %s", path)


def plot_reconstructions(cfg, vae, device, path, n=8):
    """Original | reconstruction pairs."""
    loader = build_dataloader(cfg, split="val", batch_size=n)
    batch, _ = next(iter(loader))
    vae.eval()
    with torch.no_grad():
        recon, _z = vae(batch.to(device))
        recon = recon.cpu()
    batch = batch.cpu()
    fig, axes = plt.subplots(2, n, figsize=(n * 1.5, 3))
    for i in range(n):
        axes[0, i].imshow(batch[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[1, i].axis("off")
    axes[0, 0].set_ylabel("Original")
    axes[1, 0].set_ylabel("Reconstruction")
    plt.suptitle("VAE Reconstructions", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    log.info("saved reconstructions -> %s", path)


def plot_samples(cfg, diff, device, path, n=8):
    """Grid of unconditional DDIM samples."""
    diff.eval()
    with torch.no_grad():
        samples = diff.sample_ddim(batch_size=n, device=device).cpu()
    samples = samples.numpy()
    decoded = []
    # decode with VAE loaded separately; for now just show latents as images
    fig, axes = plt.subplots(1, n, figsize=(n * 1.5, 1.5))
    for i in range(n):
        # average over channels for display
        img = samples[i].mean(0)
        axes[i].imshow(img, cmap="viridis")
        axes[i].axis("off")
    plt.suptitle("DDIM Latent Samples", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close(fig)
    log.info("saved samples -> %s", path)


def plot_trajectory(cfg, vae, diff, device, path, mode="ddim_interpolate"):
    """Run a simulation and plot the latent path in PCA space."""
    loader = build_dataloader(cfg, split="val", batch_size=64)
    all_z = []
    vae.eval()
    with torch.no_grad():
        for batch, _ in loader:
            z = vae.encode(batch.to(device))
            all_z.append(z.cpu().numpy().reshape(len(z), -1))
    Z = np.concatenate(all_z, 0)
    _, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    pc = Vt[:2].T  # (D, 2)

    cfg.sim_mode = mode
    # capture latents during trajectory by monkey-patching decode
    latents_path = []
    orig_decode = vae.decode

    def capture_decode(z):
        latents_path.append(z.detach().cpu().numpy().flatten())
        return orig_decode(z)

    vae.decode = capture_decode
    frames = simulate_trajectory(cfg, vae, diff, device, steps=40, rng_seed=0)
    vae.decode = orig_decode

    L = np.array(latents_path)
    traj_proj = (L - Z.mean(0)) @ pc

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    data_proj = (Z - Z.mean(0)) @ pc
    axes[0].scatter(data_proj[:, 0], data_proj[:, 1], c="lightgray", s=5, alpha=0.4, label="data")
    axes[0].plot(traj_proj[:, 0], traj_proj[:, 1], "r-o", ms=3, lw=1.5, label="thought")
    axes[0].scatter(traj_proj[:, 0], traj_proj[:, 1], c=range(len(traj_proj)),
                    cmap="plasma", s=20, zorder=5)
    axes[0].set_title(f"Latent trajectory ({mode})")
    axes[0].legend()

    # montage of frames
    ncols = 8
    nrows = (len(frames) + ncols - 1) // ncols
    axes[1].axis("off")
    for idx, f in enumerate(frames):
        r, c = idx // ncols, idx % ncols
        # small inset axes
        ax = fig.add_axes([0.55 + c * 0.055, 0.85 - r * 0.12, 0.05, 0.1])
        ax.imshow(f, cmap="gray")
        ax.axis("off")
    plt.savefig(path, dpi=140)
    plt.close(fig)
    log.info("saved trajectory plot -> %s", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=cfg.device)
    ap.add_argument("--vae", type=str, default=None)
    ap.add_argument("--diff", type=str, default=None)
    args = ap.parse_args()

    set_seed(cfg.seed)
    device = get_device(args.device)
    vae_path = args.vae or str(Path(cfg.checkpoint_dir) / "vae.pt")
    diff_path = args.diff or str(Path(cfg.checkpoint_dir) / "diffusion.pt")

    vae, diff = load_models(cfg, device, vae_path, diff_path)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    plot_latent_space(cfg, vae, device, str(out / "analysis_latent_space.png"))
    plot_reconstructions(cfg, vae, device, str(out / "analysis_reconstructions.png"))
    plot_samples(cfg, diff, device, str(out / "analysis_samples.png"))
    plot_trajectory(cfg, vae, diff, device, str(out / "analysis_trajectory_ddim.png"), "ddim_interpolate")
    plot_trajectory(cfg, vae, diff, device, str(out / "analysis_trajectory_ode.png"), "ode")
    log.info("Analysis complete.")


if __name__ == "__main__":
    main()