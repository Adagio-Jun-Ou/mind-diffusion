"""Bicameral phase diagram: sweep the dial from perception to internal voice.

Runs the knot (or world_knot) loop across a grid of bicameral values and
error gains, measures the dynamics fingerprints (effective rank, lag-1
autocorrelation, activity, tail-fraction, pixel change, prediction error),
and renders a phase diagram: where does the loop transition from a living
perception-driven flow into self-echo attractor collapse?

This is the experiment behind the Jaynes finding: "hearing voices" as a
dynamical phase transition, not a metaphor.

Usage:
    python scripts/bicameral_sweep.py --mode knot --steps 50
    python scripts/bicameral_sweep.py --mode world_knot --steps 50
    python scripts/bicameral_sweep.py --mode knot --bicameral 0,0.25,0.5,0.75,1.0
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
    simulate_knot, load_models, save_gif, save_montage, trajectory_metrics,
)
from mind_diffusion.mind_jepa_bridge import (
    MindJEPA, WorldToMindBridge, MindToWorldAction, TerminalWorldAdapter,
    simulate_mind_world_knot, fit_bridge_to_vae,
)
from mind_diffusion.terminal_world import VirtualShell, generate_session

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s | %(message)s")
log = logging.getLogger("bicameral_sweep")


def run_point(mode, cfg, vae, diff, device, steps, seed, b, eg,
              jepa=None, bridge=None, actor=None, obs_frames=None):
    stats, latents = [], []
    if mode == "world_knot":
        frames = simulate_mind_world_knot(
            cfg, vae, diff, jepa, bridge, actor, obs_frames, "cpu",
            steps=steps, step_scale=cfg.sim_step_scale,
            bicameral=b, error_gain=eg, stats_out=stats, latents_out=latents)
    else:
        frames = simulate_knot(
            cfg, vae, diff, device, steps, cfg.sim_step_scale, seed,
            bicameral=b, error_gain=eg, stats_out=stats, latents_out=latents)
    m = trajectory_metrics(latents)
    tail = max(steps // 3, 1)
    m["pixel_change"] = float(np.mean([s["pixel_change"] for s in stats[tail:]]))
    m["pred_error"] = float(np.mean([s["pred_error"] for s in stats[tail:]]))
    m["err_cosine"] = float(np.mean([s["err_cosine"] for s in stats[tail:]]))
    return m, frames


def plot_phase(results, b_vals, eg_vals, path, mode):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_names = [("pixel_change", "pixel change (thought mobility)"),
                    ("autocorr", "lag-1 autocorrelation (self-echo)"),
                    ("effective_rank", "effective rank (state diversity)"),
                    ("pred_error", "prediction error")]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.4))
    B, G = np.meshgrid(b_vals, eg_vals, indexing="ij")
    for ax, (key, title) in zip(axes, metric_names):
        Z = np.array([[results[(b, g)][key] for g in eg_vals] for b in b_vals])
        im = ax.imshow(Z, origin="lower", aspect="auto", cmap="viridis",
                       extent=[min(eg_vals), max(eg_vals), min(b_vals), max(b_vals)])
        ax.set_xlabel("error gain")
        ax.set_ylabel("bicameral")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax)
    fig.suptitle(f"bicameral phase diagram ({mode}) - collapse onset", fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    log.info("saved phase diagram %s", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["knot", "world_knot"], default="knot")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--bicameral", type=str, default="0,0.25,0.5,0.75,0.9,1.0")
    ap.add_argument("--error-gain", type=str, default="0.5,1.0,2.0")
    ap.add_argument("--vae", type=str, default=None)
    ap.add_argument("--diff", type=str, default=None)
    ap.add_argument("--jepa", type=str, default=None)
    ap.add_argument("--tag", type=str, default="phase")
    args = ap.parse_args()

    set_seed(cfg.seed)
    device = get_device(args.device) if args.mode == "knot" else torch.device("cpu")
    b_vals = [float(x) for x in args.bicameral.split(",")]
    eg_vals = [float(x) for x in args.error_gain.split(",")]

    vae_path = args.vae or str(Path(cfg.checkpoint_dir) / "vae.pt")
    diff_path = args.diff or str(Path(cfg.checkpoint_dir) / "diffusion.pt")
    vae, diff = load_models(cfg, device, vae_path, diff_path)

    jepa = bridge = actor = obs_frames = None
    if args.mode == "world_knot":
        jepa_path = args.jepa or str(Path(cfg.checkpoint_dir) / "jepa.pt")
        if not Path(jepa_path).exists():
            log.error("JEPA checkpoint not found: %s", jepa_path); sys.exit(1)
        jepa = MindJEPA().to("cpu").eval()
        jepa.load_state_dict(torch.load(jepa_path, map_location="cpu"))
        bridge = WorldToMindBridge(world_dim=jepa.zdim).to("cpu")
        actor = MindToWorldAction().to("cpu")
        fit_bridge_to_vae(bridge, vae, cfg, device)
        rng = np.random.default_rng(args.seed)
        turns = generate_session(VirtualShell(seed=args.seed), rng,
                                 n_turns=max(args.steps // 4, 8))
        obs_frames, _ = TerminalWorldAdapter().session_to_frames(turns)
        obs_frames = obs_frames[None]

    log.info("sweeping %s: %d bicameral x %d gain = %d runs",
             args.mode, len(b_vals), len(eg_vals), len(b_vals) * len(eg_vals))
    results = {}
    for b in b_vals:
        for eg in eg_vals:
            m, _ = run_point(args.mode, cfg, vae, diff, device, args.steps,
                             args.seed, b, eg, jepa, bridge, actor, obs_frames)
            results[(b, eg)] = m
            log.info("b=%.2f gain=%.2f | rank=%.1f autocorr=%.2f activity=%.4f "
                     "tail=%.3f pix=%.4f err=%.2f",
                     b, eg, m["effective_rank"], m["autocorr"], m["activity"],
                     m["tail_fraction"], m["pixel_change"], m["pred_error"])

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    json_path = out / f"bicameral_{args.tag}_{args.mode}.json"
    json_path.write_text(json.dumps({
        "mode": args.mode, "steps": args.steps,
        "bicameral": b_vals, "error_gain": eg_vals,
        "results": {f"{b}|{g}": v for (b, g), v in results.items()},
    }, indent=2))
    log.info("saved sweep -> %s", json_path)
    plot_phase(results, b_vals, eg_vals,
               str(out / f"bicameral_{args.tag}_{args.mode}.png"), args.mode)

    # headline: where does collapse begin?
    for g in eg_vals:
        pix = [results[(b, g)]["pixel_change"] for b in b_vals]
        base = max(pix[0], 1e-6)
        onset = next((b_vals[i] for i in range(len(b_vals))
                      if pix[i] < 0.25 * base), None)
        log.info("gain=%.1f: collapse onset (pixel change < 25%% of b=0) at b=%s",
                 g, onset if onset is not None else ">max")
    log.info("Done.")


if __name__ == "__main__":
    main()
