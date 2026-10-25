"""Train the VAE that maps neuron images <-> mind latent space.

Usage:
    python scripts/train_vae.py [--epochs N] [--batch-size N] [--device auto]
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

from mind_diffusion.config import cfg
from mind_diffusion.device import get_device, set_seed
from mind_diffusion.data import build_dataloader
from mind_diffusion.models import VAE, save_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s | %(message)s")
log = logging.getLogger("train_vae")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=cfg.vae_epochs)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--device", type=str, default=cfg.device)
    ap.add_argument("--lr", type=float, default=cfg.vae_lr)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    set_seed(cfg.seed)
    device = get_device(args.device)
    log.info("device=%s", device)

    loader = build_dataloader(cfg, split="train", batch_size=args.batch_size)
    vae = VAE(cfg).to(device)
    opt = torch.optim.AdamW(vae.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(Path(cfg.checkpoint_dir) / "vae.pt")

    for epoch in range(args.epochs):
        vae.train()
        t0 = time.perf_counter()
        total, n = 0.0, 0
        for i, (batch, _labels) in enumerate(loader):
            batch = batch.to(device)
            recon, z = vae(batch)
            recon_loss = torch.nn.functional.mse_loss(recon, batch, reduction="mean")
            kl = -0.5 * torch.mean(1 + torch.log(z.var(dim=0) + 1e-8) - z.mean(dim=0).pow(2) - z.var(dim=0))
            loss = recon_loss + cfg.vae_kl_weight * kl

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            opt.step()

            total += loss.item() * batch.shape[0]
            n += batch.shape[0]
            if (i + 1) % cfg.log_interval == 0:
                log.info("  epoch %d step %d recon=%.4f kl=%.4f (%.1fs)",
                         epoch + 1, i + 1, recon_loss.item(), float(kl),
                         time.perf_counter() - t0)
        sched.step()
        log.info("Epoch %d/%d  avg_loss=%.4f  (%.1fs)",
                 epoch + 1, args.epochs, total / max(n, 1),
                 time.perf_counter() - t0)

        if (epoch + 1) % cfg.save_every == 0:
            save_model(vae, out_path)

    save_model(vae, out_path)
    log.info("VAE training complete -> %s", out_path)


if __name__ == "__main__":
    main()