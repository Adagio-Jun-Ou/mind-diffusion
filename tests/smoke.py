"""Smoke test - imports and instantiates everything without training.

Run:  python tests/smoke.py
Exits non-zero if anything fails.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from mind_diffusion.config import cfg
from mind_diffusion.device import get_device, set_seed
from mind_diffusion.models import VAE, MindDiffusion
from mind_diffusion import data as data_mod
from mind_diffusion.simulation import simulate_trajectory


def main():
    set_seed(cfg.seed)
    device = get_device("auto")
    print(f"[smoke] device={device}")

    # 1. procedural generator
    img = data_mod._draw_neuron(np.random.default_rng(0), size=cfg.image_size)
    assert img.shape == (cfg.image_size, cfg.image_size)
    print(f"[smoke] procedural neuron shape OK: {img.shape}")

    # 2. VAE forward/backward
    vae = VAE(cfg).to(device)
    x = torch.randn(2, cfg.image_channels, cfg.image_size, cfg.image_size).to(device)
    recon, z = vae(x)
    assert z.shape == (2, cfg.latent_channels, cfg.latent_size, cfg.latent_size), z.shape
    assert recon.shape == x.shape, recon.shape
    loss = (recon - x).pow(2).mean()
    loss.backward()
    print(f"[smoke] VAE OK  z={tuple(z.shape)} recon={tuple(recon.shape)} loss={loss.item():.4f}")

    # 3. conditional diffusion forward / training_step
    diff = MindDiffusion(cfg).to(device)
    z0 = torch.randn(2, cfg.latent_channels, cfg.latent_size, cfg.latent_size).to(device)
    labels = torch.randint(0, cfg.num_classes, (2,), device=device)
    loss = diff.training_step(z0, labels)
    loss.backward()
    print(f"[smoke] diffusion OK  loss={loss.item():.4f}")

    # 4. DDIM sampling (few steps to keep it quick)
    cfg.diffusion_timesteps = 20
    cfg.ddim_steps = 5
    diff = MindDiffusion(cfg).to(device)
    s = diff.sample_ddim(batch_size=1, device=device, labels=torch.tensor([1], device=device))
    assert s.shape == (1, cfg.latent_channels, cfg.latent_size, cfg.latent_size)
    print(f"[smoke] DDIM sample OK  {tuple(s.shape)}")

    # 5. simulate trajectory (all modes, decode only - random weights)
    for mode in ["random_walk", "ddim_interpolate", "ode", "knot"]:
        cfg.sim_mode = mode
        stats = [] if mode == "knot" else None
        frames = simulate_trajectory(cfg, vae, diff, device, steps=3, rng_seed=0,
                                     stats_out=stats)
        assert len(frames) == 3, f"{mode}: {len(frames)} frames"
        assert frames[0].shape == (cfg.image_size, cfg.image_size), f"{mode}: {frames[0].shape}"
        if stats is not None:
            assert len(stats) == 3 and "pred_error" in stats[0]
        print(f"[smoke] simulate[{mode}] OK  {len(frames)} frames of {frames[0].shape}")

    # 6. MindJEPA bridge: terminal adapter + latent-only world model + coupled knot
    # (CPU: DirectML chokes on GRU ops, same class of fallback as UNet attention)
    from mind_diffusion.mind_jepa_bridge import (
        MindJEPA, WorldToMindBridge, MindToWorldAction, TerminalWorldAdapter,
        simulate_mind_world_knot, rankme, action_gap, rollout_error,
    )
    from mind_diffusion.terminal_world import VirtualShell, generate_session, turns_to_pairs
    cpu = torch.device("cpu")
    vae_c, diff_c = vae.to(cpu), diff.to(cpu)
    adapter = TerminalWorldAdapter()
    turns = [("ls -la", "total 12 drwxr-xr-x src tests"),
             ("git status", "On branch main nothing to commit"),
             ("python tests/smoke.py", "ALL CHECKS PASSED")]
    obs, acts = adapter.session_to_frames(turns)
    assert obs.shape[1] == 64 and acts.shape[1] == 32, (obs.shape, acts.shape)
    jepa = MindJEPA(obs_dim=64, act_dim=32).to(cpu)
    loss, pl, sl, rl = jepa.training_step(obs[None], acts[None])
    loss.backward()
    assert rankme(jepa.frame_enc(obs)) > 1.0
    gap, err_t, err_s = action_gap(jepa, obs[None], acts[None])
    errs = rollout_error(jepa, obs[None], acts[None], horizons=(1, 2))
    bridge = WorldToMindBridge(world_dim=64).to(cpu)
    actor = MindToWorldAction().to(cpu)
    stats = []
    cfg.sim_mode = "knot"
    frames = simulate_mind_world_knot(cfg, vae_c, diff_c, jepa, bridge, actor,
                                      obs[None], cpu, steps=3, step_scale=0.35,
                                      stats_out=stats)
    assert len(frames) == 3 and len(stats) == 3 and "pred_error" in stats[0]
    print(f"[smoke] MindJEPA bridge OK  loss={loss.item():.4f} (pred={pl:.4f} self={sl:.4f} reg={rl:.4f}), "
          f"rankme>1, rollout={{{', '.join(f'{k}:{v:.3f}' for k, v in errs.items())}}}, "
          f"coupled knot 3 frames, world obs={tuple(obs.shape)}")

    # 7. terminal world simulator: dynamics + error/recovery signals
    sim = VirtualShell(seed=0)
    rng = np.random.default_rng(0)
    sess = generate_session(sim, rng, n_turns=10)
    pairs = turns_to_pairs(sess)
    codes = [t["exit"] for t in sess]
    assert len(pairs) == 10 and any(c == 0 for c in codes)
    assert any(c != 0 for c in codes), "generator should produce some failures"
    o2, a2 = adapter.session_to_frames(sess)
    assert o2.shape == (10, 64) and a2.shape == (10, 32)
    print(f"[smoke] terminal world OK  {len(sess)} turns, "
          f"{sum(1 for c in codes if c != 0)} non-zero exits, {sum(1 for c in codes if c == 0)} ok")

    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()