"""Mind simulation engine (v2 - connectome, DDIM interpolation, ODE, knot).

Trajectory modes (cfg.sim_mode):
  - "random_walk"    : Gaussian perturbation + manifold pull-back (v1)
  - "ddim_interpolate": interpolate between two sampled latents via DDIM
  - "connectome"      : drive the latent walk with a real neural connectome
                        adjacency matrix (C. elegans or arbitrary .npy)
  - "ode"            : integrate a latent flow ODE for smooth trajectories
  - "knot"           : endless-knot loop - generation (action out) is
                        re-perceived (action in) and its prediction error
                        drives the next step, with a bicameral blend between
                        sensory error and internal command, inside a
                        walled-garden latent boundary

All modes decode each latent step through the VAE and collect frames.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from .models import VAE, MindDiffusion

log = logging.getLogger("mind-diffusion.sim")


def _to_uint8(img_tensor):
    arr = img_tensor.detach().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    if arr.shape[0] == 1:
        arr = arr[0]
    else:
        arr = arr.mean(0)
    return (arr * 255).astype(np.uint8)


def load_models(cfg, device, vae_path=None, diff_path=None):
    vae = VAE(cfg).to(device).eval()
    diff = MindDiffusion(cfg).to(device).eval()
    if vae_path:
        vae.load_state_dict(torch.load(vae_path, map_location=device))
        log.info("loaded VAE from %s", vae_path)
    if diff_path:
        diff.load_state_dict(torch.load(diff_path, map_location=device))
        log.info("loaded diffusion from %s", diff_path)
    return vae, diff


# --------------------------------------------------------------------------- #
# Connectome loading
# --------------------------------------------------------------------------- #
def load_connectome(path: str, latent_dim: int, device="cpu") -> torch.Tensor:
    """Load an adjacency matrix and project it to a latent-rotation operator.

    The connectome drives the "thought" as a linear dynamical system on the
    latent grid:  z_{t+1} = W z_t + noise, where W is derived from the
    connectome's spectral properties (scaled to keep the system stable).
    """
    if not path:
        return None
    adj = np.load(path) if path.endswith(".npy") else np.loadtxt(path, delimiter=",")
    adj = np.asarray(adj, dtype=np.float32)
    # symmetrize + normalize
    adj = (adj + adj.T) * 0.5
    deg = adj.sum(axis=1, keepdims=True) + 1e-6
    norm_adj = adj / np.sqrt(deg) / np.sqrt(deg).T
    # project to latent dimension via SVD truncation
    U, S, _Vt = np.linalg.svd(norm_adj)
    d = min(latent_dim, U.shape[1])
    W = (U[:, :d] * (0.9 * S[:d] / (S[0] + 1e-6))) @ U[:, :d].T
    W = torch.tensor(W, dtype=torch.float32, device=device)
    log.info("connectome loaded: %s -> %dx%d operator (spectral radius %.3f)",
             path, W.shape[0], W.shape[1], float(S[0]))
    return W


# --------------------------------------------------------------------------- #
# Trajectory generators
# --------------------------------------------------------------------------- #
def _manifold_pull(diff, z, t_idx, device):
    """One DDIM denoise step to pull z back toward the learned manifold."""
    t = torch.full((z.shape[0],), t_idx, device=device, dtype=torch.long)
    eps = diff.unet(z, t, None)
    ab = diff.alpha_bars[t_idx]
    x0 = (z - (1 - ab).sqrt() * eps) / ab.sqrt().clamp(min=1e-8)
    return x0 * ab.sqrt() + (1 - ab).sqrt() * eps  # re-noise slightly


def simulate_random_walk(cfg, vae, diff, device, steps, step_scale, rng_seed):
    torch.manual_seed(rng_seed)
    z = diff.sample_ddim(batch_size=1, device=device) if cfg.sim_seed_latents else \
        torch.randn(1, cfg.latent_channels, cfg.latent_size, cfg.latent_size, device=device)
    frames = []
    T = cfg.diffusion_timesteps
    t_idx = max(0, min(T - 1, int(T * 0.1)))
    for i in range(steps):
        frames.append(_to_uint8(vae.decode(z)[0]))
        z = z + step_scale * torch.randn_like(z)
        z = _manifold_pull(diff, z, t_idx, device)
    return frames


def simulate_ddim_interpolate(cfg, vae, diff, device, steps, rng_seed, labels=None):
    """Sample two latents z_a, z_b and slerp between them, decoding each step."""
    torch.manual_seed(rng_seed)
    z_a = diff.sample_ddim(batch_size=1, device=device, labels=labels)
    z_b = diff.sample_ddim(batch_size=1, device=device,
                            labels=labels,
                            steps=cfg.ddim_steps)
    z_a = z_a.flatten()
    z_b = z_b.flatten()
    # slerp
    dot = (z_a * z_b).sum()
    na = z_a.norm() + 1e-8
    nb = z_b.norm() + 1e-8
    omega = torch.acos(torch.clamp(dot / (na * nb), -1, 1))
    frames = []
    for i in range(steps):
        t_frac = i / max(steps - 1, 1)
        if omega.abs() < 1e-4:
            z = z_a * (1 - t_frac) + z_b * t_frac
        else:
            z = (torch.sin((1 - t_frac) * omega) / torch.sin(omega)) * z_a \
              + (torch.sin(t_frac * omega) / torch.sin(omega)) * z_b
        z_grid = z.reshape(1, cfg.latent_channels, cfg.latent_size, cfg.latent_size)
        frames.append(_to_uint8(vae.decode(z_grid)[0]))
    return frames


def simulate_connectome(cfg, vae, diff, device, steps, step_scale, rng_seed):
    """Drive the latent walk with a connectome-derived linear operator.

    z_{t+1} = W @ z_t + step_scale * noise, with periodic manifold pull-back.
    """
    torch.manual_seed(rng_seed)
    z = diff.sample_ddim(batch_size=1, device=device) if cfg.sim_seed_latents else \
        torch.randn(1, cfg.latent_channels, cfg.latent_size, cfg.latent_size, device=device)
    flat_dim = cfg.latent_channels * cfg.latent_size * cfg.latent_size
    W = load_connectome(cfg.sim_connectome_path, flat_dim, device)
    if W is None:
        log.warning("No connectome path set; falling back to random_walk")
        return simulate_random_walk(cfg, vae, diff, device, steps, step_scale, rng_seed)

    frames = []
    T = cfg.diffusion_timesteps
    t_idx = max(0, min(T - 1, int(T * 0.1)))
    for i in range(steps):
        frames.append(_to_uint8(vae.decode(z)[0]))
        z_flat = z.flatten(1)  # (1, flat_dim)
        # ensure W matches flat_dim
        d = min(W.shape[0], z_flat.shape[1])
        z_new = torch.zeros_like(z_flat)
        z_new[:, :d] = z_flat[:, :d] @ W[:d, :d].T + step_scale * torch.randn(1, d, device=device)
        z_new[:, d:] = z_flat[:, d:]  # pass-through for any extra dims
        z = z_new.reshape(z.shape)
        if i % 5 == 4:
            z = _manifold_pull(diff, z, t_idx, device)
    return frames


def simulate_ode(cfg, vae, diff, device, steps, rng_seed):
    """Smooth ODE trajectory: integrate dz/dt = f(z) where f is approximated by
    the score function (noise prediction) scaled by a drift term.

    This produces the smoothest "thought flow" - a continuous latent path.
    """
    torch.manual_seed(rng_seed)
    z = diff.sample_ddim(batch_size=1, device=device) if cfg.sim_seed_latents else \
        torch.randn(1, cfg.latent_channels, cfg.latent_size, cfg.latent_size, device=device)

    frames = []
    T = cfg.diffusion_timesteps
    t_idx = max(0, min(T - 1, int(T * 0.5)))
    dt = 0.1  # integration step
    for i in range(steps):
        frames.append(_to_uint8(vae.decode(z)[0]))
        # "velocity" = -score function (points toward higher data density)
        for _ in range(3):  # sub-steps for smoothness
            t = torch.full((1,), t_idx, device=device, dtype=torch.long)
            eps = diff.unet(z, t, None)
            ab = diff.alpha_bars[t_idx]
            score = -eps / (1 - ab).sqrt()
            # mild attraction toward manifold + a curl term for rotation
            velocity = -0.05 * score + 0.02 * torch.randn_like(z)
            z = z + dt * velocity
    return frames


def simulate_knot(cfg, vae, diff, device, steps, step_scale, rng_seed,
                  bicameral=None, error_gain=None, labels=None, stats_out=None,
                  latents_out=None):
    """Endless-knot loop: the mind's output becomes its own input.

    Each step closes one turn of the knot:

      1. front face - ACTION OUT (efferent / generative):
           decode z -> percept x_hat. The model commands its world.
      2. back face - ACTION IN (afferent / perceptual):
           the percept echoes back through a noisy sensory channel,
           z_hat = encode(x_hat + noise); the prediction error
           e = z_hat - z measures how much the world disagrees.
      3. bicameral blend (improvement on Jaynes' one-way channel):
           drive = (1 - b) * error_gain * e   (listen to perception)
                 +  b * internal_command       (obey "the voice")
           where internal_command = -score(z) from the diffusion prior.
           b=0.5 is the balanced loop; b->1 is Jaynes' bicameral mind,
           where the model hears only itself and the knot tightens.
      4. walled garden: a radial latent boundary (margin * seed norm)
           plus periodic manifold pull-back keeps the loop in-distribution.

    Returns frames; per-step telemetry is appended to ``stats_out`` if given,
    per-step latents to ``latents_out`` (flat numpy) if given.
    """
    torch.manual_seed(rng_seed)
    b = cfg.sim_bicameral if bicameral is None else bicameral
    eg = cfg.sim_knot_error_gain if error_gain is None else error_gain

    z = diff.sample_ddim(batch_size=1, device=device) if cfg.sim_seed_latents else \
        torch.randn(1, cfg.latent_channels, cfg.latent_size, cfg.latent_size, device=device)

    # walled garden: radius anchored to the seed latent's norm
    garden_r = cfg.sim_garden_margin * float(z.flatten().norm())

    T = cfg.diffusion_timesteps
    t_pull = max(0, min(T - 1, int(T * 0.1)))    # gentle manifold floor
    t_cmd = max(0, min(T - 1, int(T * 0.35)))    # internal command noise level

    frames = []
    prev_img = None
    prev_e = None
    for i in range(steps):
        # ---- front face: action out ----
        img = vae.decode(z)
        frames.append(_to_uint8(img[0]))
        if latents_out is not None:
            latents_out.append(z.flatten().detach().cpu().numpy().copy())

        # ---- back face: action in ----
        x_in = img + step_scale * torch.randn_like(img)
        z_hat = vae.encode(x_in)
        e = z_hat - z
        e_norm = float(e.flatten().norm())
        err_cos = 0.0 if prev_e is None else float(
            torch.nn.functional.cosine_similarity(
                e.flatten(), prev_e.flatten(), dim=0))
        prev_e = e

        # ---- internal command ("the voice"): prior's score direction ----
        t = torch.full((1,), t_cmd, device=device, dtype=torch.long)
        eps = diff.unet(z, t, labels)
        cmd = -eps

        # ---- bicameral blend ----
        drive = (1.0 - b) * eg * e + b * cmd
        z = z + drive + 0.01 * torch.randn_like(z)

        # ---- walled garden ----
        zn = float(z.flatten().norm())
        if zn > garden_r:
            z = z * (garden_r / zn)
        z = _manifold_pull(diff, z, t_pull, device)

        pixel_change = 0.0 if prev_img is None else float(
            (img - prev_img).abs().mean())
        prev_img = img

        if stats_out is not None:
            stats_out.append({
                "step": i,
                "pred_error": e_norm,
                "err_cosine": err_cos,
                "cmd_norm": float(cmd.flatten().norm()),
                "z_norm": float(z.flatten().norm()),
                "pixel_change": pixel_change,
            })
    return frames


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
def simulate_trajectory(cfg, vae, diff, device, seed_latents=None,
                        steps=None, step_scale=None, rng_seed=0, labels=None,
                        stats_out=None, latents_out=None):
    steps = steps or cfg.sim_steps
    step_scale = step_scale or cfg.sim_step_scale
    mode = cfg.sim_mode
    log.info("trajectory mode=%s steps=%d scale=%.3f", mode, steps, step_scale)

    if mode == "random_walk":
        return simulate_random_walk(cfg, vae, diff, device, steps, step_scale, rng_seed)
    elif mode == "ddim_interpolate":
        return simulate_ddim_interpolate(cfg, vae, diff, device, steps, rng_seed, labels)
    elif mode == "connectome":
        return simulate_connectome(cfg, vae, diff, device, steps, step_scale, rng_seed)
    elif mode == "ode":
        return simulate_ode(cfg, vae, diff, device, steps, rng_seed)
    elif mode == "knot":
        return simulate_knot(cfg, vae, diff, device, steps, step_scale,
                             rng_seed, labels=labels, stats_out=stats_out,
                             latents_out=latents_out)
    else:
        raise ValueError(f"unknown sim_mode: {mode}")


# --------------------------------------------------------------------------- #
# Trajectory metrics (consciousness-loop telemetry)
# --------------------------------------------------------------------------- #
def trajectory_metrics(latents):
    """Dynamics fingerprints of a latent trajectory (T, D).

    - effective_rank: exp(entropy of singular values) - diversity of visited
      states; collapse shows up as rank -> 1
    - autocorr: lag-1 autocorrelation (mean over dims) - self-similarity;
      a fixed point / tight knot sits near 1
    - activity: mean per-step displacement - how alive the loop is
    - tail_fraction: variance of the last 25% / variance of the first 25%
      (Shumailov-style tail-diversity ratio; < 1 means the loop narrows)
    """
    L = np.asarray(latents, dtype=np.float64)
    if L.ndim != 2 or len(L) < 4:
        return {"effective_rank": 0.0, "autocorr": 0.0,
                "activity": 0.0, "tail_fraction": 1.0}
    s = np.linalg.svd(L - L.mean(0), compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    eff_rank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    c = L - L.mean(0)
    denom = (c * c).sum(0) + 1e-12
    autocorr = float(((c[:-1] * c[1:]).sum(0) / denom).mean())
    activity = float(np.abs(np.diff(L, axis=0)).mean())
    q = max(len(L) // 4, 1)
    v_head = float(L[:q].var(0).sum())
    v_tail = float(L[-q:].var(0).sum())
    tail_fraction = float(v_tail / max(v_head, 1e-12))
    return {"effective_rank": eff_rank, "autocorr": autocorr,
            "activity": activity, "tail_fraction": tail_fraction}


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def save_gif(frames, path, fps=10):
    import imageio
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, duration=1.0 / fps)
    log.info("saved gif %s", path)


def save_montage(frames, path, ncols=6):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(frames)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.6, nrows * 1.6))
    axes = np.array(axes).reshape(-1)
    for ax, f in zip(axes, frames):
        ax.imshow(f, cmap="gray", vmin=0, vmax=255)
        ax.axis("off")
    for ax in axes[len(frames):]:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)
    log.info("saved montage %s", path)


def save_knot_diagnostics(stats, path, bicameral=None, error_gain=None):
    """Plot knot-loop telemetry: prediction error, internal command strength,
    error self-similarity (knot tightening) and pixel change per step."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = np.array([[d["pred_error"], d["cmd_norm"], d["err_cosine"],
                   d["pixel_change"], d["z_norm"]] for d in stats])
    steps = np.arange(len(stats))
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    title = "endless-knot loop"
    if bicameral is not None:
        title += f"  (bicameral={bicameral:.2f}, error_gain={error_gain:.2f})"
    fig.suptitle(title, fontsize=12)

    axes[0, 0].plot(steps, s[:, 0], label="sensory pred error |e|")
    axes[0, 0].plot(steps, s[:, 1], label="internal command |cmd|")
    axes[0, 0].set_title("action-in error vs internal voice")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_xlabel("knot turn")

    axes[0, 1].plot(steps, s[:, 2], color="darkred")
    axes[0, 1].set_ylim(-0.2, 1.05)
    axes[0, 1].axhline(0.0, color="gray", lw=0.8, ls="--")
    axes[0, 1].set_title("error self-similarity cos(e_t, e_t-1)\n-> 1 = knot tightening (self-echo)")
    axes[0, 1].set_xlabel("knot turn")

    axes[1, 0].plot(steps, s[:, 3], color="teal")
    axes[1, 0].set_title("pixel change per turn (thought stability)")
    axes[1, 0].set_xlabel("knot turn")

    axes[1, 1].plot(steps, s[:, 4], color="purple")
    axes[1, 1].set_title("latent norm (walled garden radius check)")
    axes[1, 1].set_xlabel("knot turn")

    plt.tight_layout()
    plt.savefig(path, dpi=130)
    plt.close(fig)
    log.info("saved knot diagnostics %s", path)