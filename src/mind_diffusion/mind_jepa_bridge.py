"""MindJEPA bridge - terminal / physical world models paired with the knot loop.

Extends mind_moljepa.py (molecular world adapter) with two more "worlds":

  TerminalWorldAdapter : shell sessions -> frame sequences
                         turn = (command, output); text is embedded with a
                         deterministic hash bag-of-words (no vocab training)
  PhysIOAdapter        : sensor / motor streams -> normalized frame sequences

MindJEPA learns world dynamics entirely in latent space (no reconstruction,
the defining JEPA property):

    z_t = frame_encoder(obs_t)                  (EMA target copy -> sg target)
    h_t = context_gru(z_{<=t})
    z_hat_{t+1} = predictor(h_t, action_t)      action OUT: what the mind does
    loss = MSE(z_hat, sg(z_tgt_{t+1})) + VICReg-lite(anti-collapse)

WorldToMindBridge maps world latents -> mind latent grid, so a trained JEPA
can drive the knot loop's action-in channel (see simulation.simulate_knot):

    z_mind --propose_action--> action      (action OUT, efferent)
    action -> world -> JEPA -> z_world'    (world answers)
    z_world' --bridge--> z_in              (action IN, afferent)
    e = z_in - z_mind                      (sensory prediction error)
    z_mind' = z_mind + (1-b)*gain*e + b*internal_command   (bicameral dial)

Two coupled knots: the mind turns on its own latent, the world turns on
its own latent, and the bridge is the thread between front and back faces.

Training data comes from your own history:
    turns = [("ls -la", "total 4 ..."), ("git status", "On branch main ...")]
    obs, acts = TerminalWorldAdapter().session_to_frames(turns)
    loss = MindJEPA(...).training_step(obs[None], acts[None])

Physical I/O: rows of a CSV / sensor log, same interface via PhysIOAdapter.
"""

from __future__ import annotations

import hashlib
import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("mind_jepa_bridge")


# --------------------------------------------------------------------------- #
# World adapters: raw logs -> frame tensors
# --------------------------------------------------------------------------- #
def _hash_embed(text: str, dim: int, char_ngrams: bool = True) -> np.ndarray:
    """Deterministic signed char-n-gram + word hashing (no vocab, no deps).

    char_ngrams=False for short structured text (commands) where word-level
    hashing is cleaner and collides less.
    """
    v = np.zeros(dim, dtype=np.float32)
    text = text.lower()
    for tok in text.split():
        h = int(hashlib.md5(("w:" + tok).encode("utf-8")).hexdigest(), 16)
        v[h % dim] += 1.0
    if char_ngrams:
        padded = f"  {text}  "
        for n in (3, 4):
            for i in range(len(padded) - n + 1):
                g = padded[i:i + n]
                h = int(hashlib.md5(g.encode("utf-8")).hexdigest(), 16)
                sign = 1.0 if (h >> 8) & 1 else -1.0
                v[h % dim] += 0.5 * sign
    nrm = float(np.linalg.norm(v))
    return v / nrm if nrm > 0 else v


class TerminalWorldAdapter:
    """Shell session -> (obs_frames, action_frames).

    Accepts turns as (command, output) tuples or dicts with "cmd"/"output"
    keys (the format produced by terminal_world.generate_session).

    obs_t    = hashed(output_t) + 0.2 * hashed(command_t)   (context leaks in)
    action_t = hashed(command_t)                            (what was done)

    dynamics_norm=True (default) standardizes each obs dimension across the
    session, so static content cancels and the *changes* an action causes
    dominate - this is what makes the world model action-relevant.
    """

    def __init__(self, obs_dim: int = 64, action_dim: int = 32,
                 dynamics_norm: bool = True):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.dynamics_norm = dynamics_norm

    def session_to_frames(self, turns):
        """turns: list of (command, output) tuples or {"cmd","output"} dicts.

        Returns (obs, act): (T, obs_dim), (T, action_dim) float32 tensors.
        """
        obs, act = [], []
        for t in turns:
            if isinstance(t, dict):
                cmd, out = str(t.get("cmd", "")), str(t.get("output", ""))
            else:
                cmd, out = str(t[0]), str(t[1])
            o = _hash_embed(out, self.obs_dim) + 0.2 * _hash_embed(cmd, self.obs_dim)
            n = float(np.linalg.norm(o))
            if n > 0:
                o = o / n
            obs.append(o)
            act.append(_hash_embed(cmd, self.action_dim, char_ngrams=False))
        obs = np.stack(obs)
        if self.dynamics_norm and obs.shape[0] > 1:
            mu = obs.mean(0, keepdims=True)
            sd = obs.std(0, keepdims=True).clip(min=1e-3)
            obs = (obs - mu) / sd
        return torch.from_numpy(obs.astype(np.float32)), \
            torch.from_numpy(np.stack(act).astype(np.float32))


class PhysIOAdapter:
    """Sensor/motor log -> (obs_frames, action_frames).

    sensors: (T, D) array (imu, joint angles, cameras downsampled, ...)
    motors:  (T, A) array (commands actually sent to the body)
    Frames are standardized per-dimension (the body's "walled garden").
    """

    def __init__(self, clip: float = 4.0):
        self.clip = clip
        self.mu = None
        self.sd = None

    def fit(self, sensors: np.ndarray):
        self.mu = sensors.mean(0, keepdims=True)
        self.sd = sensors.std(0, keepdims=True) + 1e-6
        return self

    def frames(self, sensors: np.ndarray, motors: np.ndarray):
        assert self.mu is not None, "call fit(sensors) first"
        s = np.clip((sensors - self.mu) / self.sd, -self.clip, self.clip)
        m = np.clip(motors, -self.clip, self.clip)
        return (torch.from_numpy(s.astype(np.float32)),
                torch.from_numpy(m.astype(np.float32)))


# --------------------------------------------------------------------------- #
# MindJEPA: latent-only world model
# --------------------------------------------------------------------------- #
class MindJEPA(nn.Module):
    """Small CPU-friendly JEPA: frame encoder + GRU context + action predictor.

    No decoder, no reconstruction. The only grounding is the EMA target
    encoder plus VICReg-lite variance/covariance terms against collapse.
    """

    def __init__(self, obs_dim: int = 64, act_dim: int = 32,
                 zdim: int = 64, hidden: int = 128, ema: float = 0.99,
                 self_w: float = 0.1, roll_w: float = 0.5):
        super().__init__()
        self.zdim = zdim
        self.ema = ema
        self.self_w = self_w
        self.roll_w = roll_w
        self.frame_enc = self._make_enc(obs_dim, hidden, zdim)
        self.frame_tgt = self._make_enc(obs_dim, hidden, zdim)
        for p in self.frame_tgt.parameters():
            p.requires_grad_(False)
        self.ctx = nn.GRU(zdim, hidden, batch_first=True)
        # FiLM action conditioning: scale + shift (DINO-WM/DreamerV3-style;
        # stronger than additive concat at small scale)
        self.act_emb = nn.Linear(act_dim, 2 * hidden)
        self.pred = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, zdim))
        self.self_pred = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(),
                                       nn.Linear(hidden, zdim))

    def _modulate(self, h, act):
        """FiLM: h * (1 + scale(act)) + shift(act)."""
        g, b = self.act_emb(act).chunk(2, dim=-1)
        return h * (1.0 + g) + b

    @staticmethod
    def _make_enc(obs_dim, hidden, zdim):
        # LayerNorm on embeddings (V-JEPA 2.1 finding: normalize embeddings,
        # not targets)
        return nn.Sequential(nn.Linear(obs_dim, hidden), nn.SiLU(),
                             nn.Linear(hidden, zdim), nn.LayerNorm(zdim))

    @torch.no_grad()
    def _ema_update(self):
        m = self.ema
        for pt, p in zip(self.frame_tgt.parameters(), self.frame_enc.parameters()):
            pt.mul_(m).add_(p.detach(), alpha=1.0 - m)

    def _vicreg(self, z, var_w=1.0, cov_w=0.04):
        """VICReg-lite over a flat (N, D) batch of embeddings."""
        std = torch.sqrt(z.var(0) + 1e-4)
        var_loss = torch.relu(1.0 - std).mean()
        zc = z - z.mean(0)
        B = z.shape[0]
        cov = (zc.T @ zc) / max(B - 1, 1)
        cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / z.shape[1]
        return var_w * var_loss + cov_w * cov_loss

    def _vicreg_time(self, z, var_w=1.0, cov_w=0.04):
        """PLDM-style anti-collapse: for each feature, variance across TIME is
        hinged below 1 (features must encode change, not static info), plus a
        covariance penalty between feature pairs."""
        B, T, D = z.shape
        zt = z.transpose(0, 1)                          # (T, B, D)
        std = torch.sqrt(zt.var(0) + 1e-4)              # (B, D) across time
        var_loss = torch.relu(1.0 - std).mean()
        zc = zt - zt.mean(0, keepdim=True)              # (T, B, D)
        cov = torch.einsum("tbd,tbe->bde", zc, zc) / max(T - 1, 1)  # (B, D, D)
        off = cov - torch.diag_embed(torch.diagonal(cov, dim1=1, dim2=2))
        cov_loss = off.pow(2).sum() / (B * D)
        return var_w * var_loss + cov_w * cov_loss

    def training_step(self, obs, act):
        """obs: (B, T, obs_dim), act: (B, T, act_dim).

        Losses (research-backed):
          * L1 next-latent prediction on EMA targets (V-JEPA)
          * context self-prediction from GRU state (V-JEPA 2.1)
          * PLDM time-axis VICReg anti-collapse
          * 2-step autoregressive rollout (differentiable, compounding-error)
        Returns (loss, pred_l1, self_l1, reg) floats.
        """
        B, T, _ = obs.shape
        with torch.no_grad():
            z_tgt = self.frame_tgt(obs)                  # (B, T, zdim)
        z = self.frame_enc(obs)
        h, h_n = self.ctx(z)                             # (B, T, hidden)

        h_t = h[:, :-1]
        # alignment: obs[t+1] is caused by act[t+1] (turn = same command/output)
        z_hat = self.pred(torch.tanh(self._modulate(h_t, act[:, 1:])))
        pred_l1 = F.l1_loss(z_hat, z_tgt[:, 1:])

        # context self-prediction: reconstruct own (visible) frame latents
        z_self = self.self_pred(torch.tanh(h))
        self_l1 = F.l1_loss(z_self, z_tgt.detach())

        # differentiable 2-step autoregressive rollout (compounding error)
        if T >= 3:
            out_pre, h_pre = self.ctx(z[:, :-2])          # context through t-3
            z1 = self.pred(torch.tanh(self._modulate(out_pre[:, -1], act[:, -2])))
            out2, _ = self.ctx(z1.unsqueeze(1), h_pre)    # feed prediction back
            z2 = self.pred(torch.tanh(self._modulate(out2[:, 0], act[:, -1])))
            roll_l1 = F.l1_loss(z1, z_tgt[:, -2]) + F.l1_loss(z2, z_tgt[:, -1])
        else:
            roll_l1 = torch.zeros((), device=obs.device)

        reg = self._vicreg_time(z_hat)
        self._ema_update()
        return (pred_l1 + self.self_w * self_l1 + self.roll_w * roll_l1 + reg,
                float(pred_l1), float(self_l1), float(reg))

    @torch.no_grad()
    def roll(self, z0, actions, h0=None):
        """Roll the world forward in latent space.

        z0: (B, zdim) initial world latent (e.g. frame_enc of last obs)
        actions: (B, S, act_dim) proposed actions
        h0: optional (1, B, hidden) GRU state to continue from
        Returns (B, S, zdim) predicted world latents.
        """
        h = h0
        zs = []
        z = z0
        for s in range(actions.shape[1]):
            out, h = self.ctx(z.unsqueeze(1), h)
            z = self.pred(torch.tanh(self._modulate(out[:, 0], actions[:, s])))
            zs.append(z)
        return torch.stack(zs, dim=1)

    @torch.no_grad()
    def encode_last(self, obs):
        """Encode observation history -> (GRU state h_n, last world latent).

        h_n has shape (1, B, hidden) - directly reusable as GRU state for roll().
        """
        z = self.frame_enc(obs)
        _out, h_n = self.ctx(z)
        return h_n, z[:, -1]


# --------------------------------------------------------------------------- #
# Bridges: world latent <-> mind latent
# --------------------------------------------------------------------------- #
class WorldToMindBridge(nn.Module):
    """World latent -> mind latent grid (mirrors mind_moljepa.BridgeNet)."""

    def __init__(self, world_dim: int = 64, mind_latent_channels: int = 4,
                 mind_latent_size: int = 8, hidden: int = 256):
        super().__init__()
        self.lc, self.ls = mind_latent_channels, mind_latent_size
        self.proj = nn.Sequential(
            nn.Linear(world_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, mind_latent_channels * mind_latent_size * mind_latent_size),
        )
        self.register_buffer("mu", torch.zeros(1, 1, 1, 1))
        self.register_buffer("sd", torch.ones(1, 1, 1, 1))
        self.register_buffer("aligned", torch.zeros(1, dtype=torch.bool))

    @torch.no_grad()
    def fit_stats(self, z_mind):
        """Moment-match to real VAE latents: z_out = raw * sd + mu.

        Cheap distribution alignment when no paired (world, brain) data
        exists - keeps the coupled knot loop inside the VAE's training
        distribution (the walled garden's outer wall).
        """
        z = z_mind.reshape(len(z_mind), -1).to(self.proj[0].weight.device)
        mu = z.mean(0).view(1, self.lc, self.ls, self.ls)
        sd = z.std(0).view(1, self.lc, self.ls, self.ls).clamp(min=1e-4)
        self.mu = mu
        self.sd = sd
        self.aligned.fill_(True)
        log.info("bridge aligned to mind latent stats: mu=%.3f sd=%.3f",
                 float(self.mu.mean()), float(self.sd.mean()))
        return self

    def forward(self, z_world):
        z = self.proj(z_world).view(-1, self.lc, self.ls, self.ls)
        if bool(self.aligned):
            z = z * self.sd + self.mu
        return z


class MindToWorldAction(nn.Module):
    """Mind latent -> proposed world action (tanh-bounded, efferent head)."""

    def __init__(self, mind_flat: int = 256, act_dim: int = 32, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(mind_flat, hidden), nn.SiLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )

    def forward(self, z_mind):
        return self.net(z_mind.flatten(1))


# --------------------------------------------------------------------------- #
# Evaluation: is the world model real or collapsed?
# --------------------------------------------------------------------------- #
@torch.no_grad()
def rankme(z):
    """RankMe effective rank: exp(entropy of normalized singular values).
    Label-free collapse detector; low value = representation collapsed."""
    z = z.reshape(-1, z.shape[-1]).float()
    if z.shape[0] < 2:
        return 0.0
    s = torch.linalg.svdvals(z - z.mean(0))
    p = s / s.sum().clamp(min=1e-12)
    return float(torch.exp(-(p * (p + 1e-12).log()).sum()))


@torch.no_grad()
def action_gap(jepa, obs, act):
    """True-action prediction error vs shuffled-action error.

    gap = err(shuffled) - err(true). A model that ignores actions scores
    ~0; a model that uses them scores clearly positive.
    """
    z_tgt = jepa.frame_tgt(obs)
    z = jepa.frame_enc(obs)
    h, _ = jepa.ctx(z)
    # obs[t+1] is caused by act[t+1]
    true_hat = jepa.pred(torch.tanh(jepa._modulate(h[:, :-1], act[:, 1:])))
    perm = torch.randperm(act.shape[0])
    shuf_hat = jepa.pred(torch.tanh(jepa._modulate(h[:, :-1], act[perm, 1:])))
    err_t = float(F.l1_loss(true_hat, z_tgt[:, 1:]))
    err_s = float(F.l1_loss(shuf_hat, z_tgt[:, 1:]))
    return err_s - err_t, err_t, err_s


@torch.no_grad()
def rollout_error(jepa, obs, act, horizons=(1, 2, 4, 8)):
    """Autoregressive latent rollout error at multiple horizons.

    Uses GRU state from the first frame, then rolls forward feeding its own
    predictions - measures compounding error (the planning-usefulness proxy).
    """
    z_tgt = jepa.frame_tgt(obs)
    z = jepa.frame_enc(obs)
    _out, h = jepa.ctx(z[:, :1])
    z_cur = z[:, 0]
    errs = {}
    max_h = min(max(horizons), obs.shape[1] - 1)
    for step in range(1, max_h + 1):
        out, h = jepa.ctx(z_cur.unsqueeze(1), h)
        z_cur = jepa.pred(torch.tanh(jepa._modulate(out[:, 0], act[:, step])))
        if step in horizons:
            errs[step] = float(F.l1_loss(z_cur, z_tgt[:, step]))
    return errs


@torch.no_grad()
def fit_bridge_to_vae(bridge, vae, cfg, device, n: int = 64):
    """Estimate real mind-latent stats by encoding validation images, then align."""
    from .data import build_dataloader
    loader = build_dataloader(cfg, split="val", batch_size=n)
    batch, _ = next(iter(loader))
    z = vae.encode(batch.to(device))
    bridge.fit_stats(z)
    return bridge


# --------------------------------------------------------------------------- #
# Coupled loop: the knot closed through a world model
# --------------------------------------------------------------------------- #
@torch.no_grad()
def simulate_mind_world_knot(cfg, vae, diff, jepa, bridge, actor,
                             obs_frames, device, steps, step_scale,
                             bicameral=0.3, error_gain=1.0, labels=None,
                             stats_out=None, latents_out=None):
    """Endless-knot loop whose action-in channel is a JEPA world model.

    Mirrors simulation.simulate_knot, but the sensory echo is not the mind
    re-encoding its own decode - it is the world model answering the mind's
    proposed action, then bridged back into mind latent space.

    obs_frames: (1, T0, obs_dim) real observation history to seed the world.
    """
    from .simulation import _to_uint8, _manifold_pull

    jepa.eval(); bridge.eval(); actor.eval()
    torch.manual_seed(0)

    # everything in this loop must share one device (GRU ops are CPU-safe)
    device = torch.device("cpu")
    vae = vae.to(device)
    diff = diff.to(device)
    jepa = jepa.to(device)
    bridge = bridge.to(device)
    actor = actor.to(device)

    # seed the mind + the world from real observations
    z_mind = diff.sample_ddim(batch_size=1, device=device) if cfg.sim_seed_latents else \
        torch.randn(1, cfg.latent_channels, cfg.latent_size, cfg.latent_size, device=device)
    obs = obs_frames.to(device)
    h_world, z_world = jepa.encode_last(obs)

    garden_r = cfg.sim_garden_margin * float(z_mind.flatten().norm())
    T = cfg.diffusion_timesteps
    t_pull = max(0, min(T - 1, int(T * 0.1)))
    t_cmd = max(0, min(T - 1, int(T * 0.35)))

    frames = []
    prev_img = None
    prev_e = None
    for i in range(steps):
        # front face: action out
        img = vae.decode(z_mind)
        frames.append(_to_uint8(img[0]))
        if latents_out is not None:
            latents_out.append(z_mind.flatten().detach().cpu().numpy().copy())
        act = actor(z_mind).unsqueeze(1)                  # (1, 1, act_dim)

        # world answers: one recurrent step in world latent space, carrying
        # the world's own GRU state so its dynamics have memory
        out_w, h_world = jepa.ctx(z_world.unsqueeze(1), h_world)
        z_world = jepa.pred(torch.tanh(jepa._modulate(out_w[:, 0], act[:, 0])))

        # back face: action in (bridged world state + sensory noise).
        # The noise matters: a fully deterministic world answer + manifold
        # pull-back settles into a fixed point (dead loop).
        z_in = bridge(z_world) + step_scale * torch.randn_like(z_mind)
        e = z_in - z_mind
        err_cos = 0.0 if prev_e is None else float(
            torch.nn.functional.cosine_similarity(
                e.flatten(), prev_e.flatten(), dim=0))
        prev_e = e

        # internal command + bicameral blend (same dial as simulate_knot)
        t = torch.full((1,), t_cmd, device=device, dtype=torch.long)
        cmd = -diff.unet(z_mind, t, labels)
        drive = (1.0 - bicameral) * error_gain * e + bicameral * cmd
        z_mind = z_mind + drive + 0.01 * torch.randn_like(z_mind)

        # walled garden
        zn = float(z_mind.flatten().norm())
        if zn > garden_r:
            z_mind = z_mind * (garden_r / zn)
        z_mind = _manifold_pull(diff, z_mind, t_pull, device)

        pixel_change = 0.0 if prev_img is None else float(
            (img - prev_img).abs().mean())
        prev_img = img

        if stats_out is not None:
            stats_out.append({
                "step": i,
                "pred_error": float(e.flatten().norm()),
                "err_cosine": err_cos,
                "cmd_norm": float(cmd.flatten().norm()),
                "action_norm": float(act.flatten().norm()),
                "world_z_norm": float(z_world.flatten().norm()),
                "z_norm": float(z_mind.flatten().norm()),
                "pixel_change": pixel_change,
            })
    return frames


if __name__ == "__main__":
    print("MindJEPA bridge: terminal/physical worlds paired with the knot loop")
    print("  obs/act -> MindJEPA (latent-only dynamics) -> WorldToMindBridge -> mind latent")
    print("  coupled loop: simulate_mind_world_knot(cfg, vae, diff, jepa, bridge, actor, obs)")
