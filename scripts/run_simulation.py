"""Run a mind simulation with the upgraded trajectory engine.

Modes: random_walk | ddim_interpolate | connectome | ode | knot | world_knot

Usage:
    python scripts/run_simulation.py --mode ddim_interpolate --steps 60 --tag thought1
    python scripts/run_simulation.py --mode connectome --connectome data/connectome.npy
    python scripts/run_simulation.py --mode knot --bicameral 0.9 --tag voices
    python scripts/run_simulation.py --mode world_knot --bicameral 0.3 --tag embodied
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch

from mind_diffusion.config import cfg
from mind_diffusion.device import get_device, set_seed
from mind_diffusion.simulation import (
    simulate_trajectory, save_gif, save_montage, save_knot_diagnostics,
    load_models, trajectory_metrics,
)
from mind_diffusion.mind_jepa_bridge import (
    MindJEPA, WorldToMindBridge, MindToWorldAction, TerminalWorldAdapter,
    simulate_mind_world_knot, fit_bridge_to_vae,
)
from mind_diffusion.terminal_world import VirtualShell, generate_session, turns_to_pairs, load_jsonl_sessions

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s | %(message)s")
log = logging.getLogger("run_sim")


def _load_world_seed(args, obs_dim, act_dim):
    """Real session from JSONL, else a synthetic virtual-shell session."""
    adapter = TerminalWorldAdapter(obs_dim, act_dim)
    if args.jsonl:
        sessions = load_jsonl_sessions(args.jsonl, max_sessions=1)
        turns = sessions[0] if sessions else []
        log.info("world seed: %d real turns from %s", len(turns), args.jsonl)
    else:
        rng = np.random.default_rng(args.seed)
        sim = VirtualShell(seed=args.seed)
        turns = generate_session(sim, rng, n_turns=max(args.steps // 4, 8))
        log.info("world seed: %d synthetic virtual-shell turns", len(turns))
    obs, act = adapter.session_to_frames(turns)
    return obs[None], act[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default=cfg.sim_mode,
                    choices=["random_walk", "ddim_interpolate", "connectome",
                             "ode", "knot", "world_knot"])
    ap.add_argument("--steps", type=int, default=cfg.sim_steps)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", type=str, default="thought1")
    ap.add_argument("--device", type=str, default=cfg.device)
    ap.add_argument("--vae", type=str, default=None)
    ap.add_argument("--diff", type=str, default=None)
    ap.add_argument("--jepa", type=str, default=None,
                    help="path to trained JEPA checkpoint (world_knot mode)")
    ap.add_argument("--jsonl", type=str, default=None,
                    help="real terminal sessions JSONL to seed the world")
    ap.add_argument("--no-align", action="store_true",
                    help="skip bridge moment-alignment to VAE latent stats")
    ap.add_argument("--step-scale", type=float, default=cfg.sim_step_scale)
    ap.add_argument("--connectome", type=str, default=None,
                    help="path to connectome .npy (for connectome mode)")
    ap.add_argument("--label", type=int, default=None,
                    help="conditioning class label (for ddim_interpolate / knot)")
    ap.add_argument("--bicameral", type=float, default=None,
                    help="knot mode: 0 = perceptual, 1 = internal voice only")
    ap.add_argument("--error-gain", type=float, default=None,
                    help="knot mode: gain on sensory prediction error")
    ap.add_argument("--no-montage", action="store_true")
    args = ap.parse_args()

    set_seed(cfg.seed)
    device = get_device(args.device)
    log.info("device=%s  mode=%s", device, args.mode)

    vae_path = args.vae or str(Path(cfg.checkpoint_dir) / "vae.pt")
    diff_path = args.diff or str(Path(cfg.checkpoint_dir) / "diffusion.pt")
    if not Path(vae_path).exists():
        log.error("VAE not found: %s", vae_path); sys.exit(1)
    if not Path(diff_path).exists():
        log.error("Diffusion not found: %s", diff_path); sys.exit(1)

    cfg.sim_mode = args.mode
    cfg.sim_step_scale = args.step_scale
    if args.bicameral is not None:
        cfg.sim_bicameral = args.bicameral
    if args.error_gain is not None:
        cfg.sim_knot_error_gain = args.error_gain
    if args.connectome:
        cfg.sim_connectome_path = args.connectome

    vae, diff = load_models(cfg, device, vae_path, diff_path)
    log.info("simulating %d-step %s trajectory (seed=%d, scale=%.2f)...",
             args.steps, args.mode, args.seed, args.step_scale)

    labels = torch.tensor([args.label], device=device).long() if args.label is not None else None
    knot_stats = [] if args.mode in ("knot", "world_knot") else None
    latents = [] if args.mode in ("knot", "world_knot") else None

    if args.mode == "world_knot":
        jepa_path = args.jepa or str(Path(cfg.checkpoint_dir) / "jepa.pt")
        if not Path(jepa_path).exists():
            log.error("JEPA checkpoint not found: %s (run scripts/train_jepa.py)", jepa_path)
            sys.exit(1)
        jepa = MindJEPA().to("cpu").eval()
        jepa.load_state_dict(torch.load(jepa_path, map_location="cpu"))
        bridge = WorldToMindBridge(world_dim=jepa.zdim).to("cpu")
        actor = MindToWorldAction().to("cpu")
        if not args.no_align:
            fit_bridge_to_vae(bridge, vae, cfg, device)
        obs_frames, _act = _load_world_seed(args, 64, 32)
        frames = simulate_mind_world_knot(
            cfg, vae, diff, jepa, bridge, actor, obs_frames, "cpu",
            steps=args.steps, step_scale=args.step_scale,
            bicameral=cfg.sim_bicameral, error_gain=cfg.sim_knot_error_gain,
            labels=labels.cpu() if labels is not None else None,
            stats_out=knot_stats, latents_out=latents,
        )
    else:
        frames = simulate_trajectory(
            cfg, vae, diff, device,
            steps=args.steps, step_scale=args.step_scale, rng_seed=args.seed,
            labels=labels, stats_out=knot_stats, latents_out=latents,
        )

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    gif_path = str(Path(cfg.output_dir) / f"sim_{args.mode}_{args.tag}.gif")
    save_gif(frames, gif_path, fps=10)
    log.info("Wrote %d frames to %s", len(frames), gif_path)
    if not args.no_montage:
        save_montage(frames, str(Path(cfg.output_dir) / f"montage_{args.mode}_{args.tag}.png"))
    if knot_stats is not None:
        metrics = trajectory_metrics(latents) if latents else {}
        stats_path = Path(cfg.output_dir) / f"knot_stats_{args.tag}.json"
        stats_path.write_text(json.dumps({
            "mode": args.mode, "bicameral": cfg.sim_bicameral,
            "error_gain": cfg.sim_knot_error_gain, "steps": args.steps,
            "metrics": metrics, "records": knot_stats,
        }, indent=2))
        log.info("saved knot stats -> %s  metrics=%s", stats_path,
                 {k: round(v, 3) for k, v in metrics.items()})
        save_knot_diagnostics(
            knot_stats, str(Path(cfg.output_dir) / f"knot_diag_{args.tag}.png"),
            bicameral=cfg.sim_bicameral, error_gain=cfg.sim_knot_error_gain)
    log.info("Done.")


if __name__ == "__main__":
    main()