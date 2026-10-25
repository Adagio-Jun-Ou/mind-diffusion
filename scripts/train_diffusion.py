"""Train the latent diffusion model (DDPM/DDIM) on encoded neuron images
with optional classifier-free guidance conditioning.

Prereq: VAE trained + data prepared.

Usage:
    python scripts/train_diffusion.py [--epochs N] [--batch-size N]
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
from mind_diffusion.models import VAE, MindDiffusion, save_model

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s | %(message)s")
log = logging.getLogger("train_diffusion")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=cfg.diffusion_epochs)
    ap.add_argument("--batch-size", type=int, default=cfg.batch_size)
    ap.add_argument("--device", type=str, default=cfg.device)
    ap.add_argument("--lr", type=float, default=cfg.diffusion_lr)
    ap.add_argument("--vae", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    set_seed(cfg.seed)
    device = get_device(args.device)
    log.info("device=%s", device)

    vae_path = args.vae or str(Path(cfg.checkpoint_dir) / "vae.pt")
    if not Path(vae_path).exists():
        log.error("VAE checkpoint not found at %s. Run train_vae.py first.", vae_path)
        sys.exit(1)

    vae = VAE(cfg).to(device).eval()
    vae.load_state_dict(torch.load(vae_path, map_location=device))
    for p in vae.parameters():
        p.requires_grad = False

    loader = build_dataloader(cfg, split="train", batch_size=args.batch_size)
    diff = MindDiffusion(cfg).to(device)
    opt = torch.optim.AdamW(diff.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    Path(cfg.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    out_path = args.out or str(Path(cfg.checkpoint_dir) / "diffusion.pt")

    for epoch in range(args.epochs):
        diff.train()
        t0 = time.perf_counter()
        total, n = 0.0, 0
        for i, (batch, labels) in enumerate(loader):
            batch = batch.to(device)
            labels = labels.to(device).long()
            with torch.no_grad():
                z0 = vae.encode(batch)
            loss = diff.training_step(z0, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(diff.parameters(), 1.0)
            opt.step()
            total += loss.item() * batch.shape[0]
            n += batch.shape[0]
            if (i + 1) % cfg.log_interval == 0:
                log.info("  epoch %d step %d  loss=%.5f (%.1fs)",
                         epoch + 1, i + 1, loss.item(), time.perf_counter() - t0)
        sched.step()
        log.info("Epoch %d/%d  avg_loss=%.5f  (%.1fs)",
                 epoch + 1, args.epochs, total / max(n, 1),
                 time.perf_counter() - t0)

        if (epoch + 1) % cfg.save_every == 0:
            save_model(diff, out_path)

    save_model(diff, out_path)
    log.info("Diffusion training complete -> %s", out_path)


if __name__ == "__main__":
    main()