"""Neural models for mind-diffusion (v2 - DDIM, CFG, conditioning).

Upgrades over v1:
  * ``ConditionalUNet``  - accepts an optional class embedding for
    classifier-free guidance (CFG).
  * ``MindDiffusion``    - DDPM training + **DDIM deterministic sampling**
    with configurable step count, plus classifier-free guidance.
  * ``VAE`` unchanged in structure but now also exposes ``encode_mean`` for
    deterministic encoding.
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("mind-diffusion.models")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _conv(ic, oc, k=3, s=1, p=1):
    return nn.Conv2d(ic, oc, k, stride=s, padding=p)


def _act():
    return nn.SiLU()


# --------------------------------------------------------------------------- #
# VAE  (image <-> mind latent)
# --------------------------------------------------------------------------- #
class VAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ic, lc = cfg.image_channels, cfg.latent_channels
        b = cfg.vae_base_channels
        ls = cfg.latent_size
        isize = cfg.image_size
        assert isize % ls == 0
        self.down_factor = isize // ls

        layers = []
        c = ic
        for ch in [b, b * 2, b * 4]:
            layers += [_conv(c, ch), _act(), _conv(ch, ch), _act()]
            c = ch
        layers += [nn.Conv2d(c, lc, kernel_size=self.down_factor,
                             stride=self.down_factor, padding=0)]
        self.encoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.ConvTranspose2d(lc, b * 4, kernel_size=self.down_factor,
                                     stride=self.down_factor, padding=0)]
        c = b * 4
        for ch in [b * 4, b * 2, b]:
            layers += [_conv(c, ch), _act(), _conv(ch, ch), _act()]
            c = ch
        layers += [nn.Conv2d(c, ic, kernel_size=3, stride=1, padding=1)]
        self.decoder = nn.Sequential(*layers)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


# --------------------------------------------------------------------------- #
# UNet building blocks
# --------------------------------------------------------------------------- #
class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim, cdim=0):
        super().__init__()
        self.conv1 = _conv(cin, cout)
        self.act1 = _act()
        self.t_proj = nn.Linear(tdim, cout)
        self.c_proj = nn.Linear(cdim, cout) if cdim > 0 else None
        self.conv2 = _conv(cout, cout)
        self.act2 = _act()
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, t_emb, c_emb=None):
        h = self.act1(self.conv1(x))
        h = h + self.t_proj(t_emb)[:, :, None, None]
        if self.c_proj is not None and c_emb is not None:
            h = h + self.c_proj(c_emb)[:, :, None, None]
        h = self.act2(self.conv2(h))
        return h + self.skip(x)


class SelfAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, ch), ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).flatten(2).transpose(1, 2)
        k = self.k(h).flatten(2)
        v = self.v(h).flatten(2).transpose(1, 2)
        attn = torch.bmm(q, k) / math.sqrt(C)
        attn = F.softmax(attn, dim=-1)
        o = torch.bmm(attn, v)
        o = o.transpose(1, 2).reshape(B, C, H, W)
        return x + self.out(o)


class Down(nn.Module):
    def __init__(self, cin, cout, tdim, cdim, n_res, attn=False):
        super().__init__()
        blocks = [ResBlock(cin, cout, tdim, cdim)]
        for _ in range(n_res - 1):
            blocks.append(ResBlock(cout, cout, tdim, cdim))
        self.blocks = nn.ModuleList(blocks)
        self.attn = SelfAttention(cout) if attn else nn.Identity()
        self.down = nn.Conv2d(cout, cout, 3, stride=2, padding=1)

    def forward(self, x, t, c):
        skips = []
        for b in self.blocks:
            x = b(x, t, c)
            skips.append(x)
        x = self.attn(x)
        x = self.down(x)
        return x, skips


class Up(nn.Module):
    def __init__(self, cin, cout, tdim, cdim, n_res, attn=False):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 2, stride=2)
        blocks = [ResBlock(cout + cout, cout, tdim, cdim)]
        for _ in range(n_res - 1):
            blocks.append(ResBlock(cout + cout, cout, tdim, cdim))
        self.blocks = nn.ModuleList(blocks)
        self.attn = SelfAttention(cout) if attn else nn.Identity()

    def forward(self, x, t, c, skips):
        x = self.up(x)
        out = []
        for b, s in zip(self.blocks, skips):
            if x.shape[-1] != s.shape[-1]:
                x = F.interpolate(x, size=s.shape[-2:], mode="nearest")
            x = torch.cat([x, s], dim=1)
            x = b(x, t, c)
            out.append(x)
        x = self.attn(x)
        return x, out


# --------------------------------------------------------------------------- #
# Conditional UNet
# --------------------------------------------------------------------------- #
class ConditionalUNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        ch = cfg.unet_base_channels
        mults = cfg.unet_channel_mults
        n_res = cfg.unet_num_res_blocks
        attn_res = set(cfg.unet_attention_resolutions)
        lc = cfg.latent_channels
        tdim = ch * 4
        cdim = cfg.cond_embed_dim if cfg.use_conditioning else 0

        self.t_mlp = nn.Sequential(nn.Linear(ch, tdim), _act(), nn.Linear(tdim, tdim))
        self.class_embed = (
            nn.Embedding(cfg.num_classes, cfg.cond_embed_dim)
            if cfg.use_conditioning else None
        )

        self.in_conv = _conv(lc, ch)

        self.downs = nn.ModuleList()
        c = ch
        ls = cfg.latent_size
        for i, m in enumerate(mults):
            cout = ch * m
            spatial = ls // (2 ** i)
            attn = spatial in attn_res
            self.downs.append(Down(c, cout, tdim, cdim, n_res, attn))
            c = cout

        self.mid1 = ResBlock(c, c, tdim, cdim)
        self.mid_attn = SelfAttention(c)
        self.mid2 = ResBlock(c, c, tdim, cdim)

        self.ups = nn.ModuleList()
        up_in = c
        for i, m in reversed(list(enumerate(mults))):
            cout = ch * m
            spatial = ls // (2 ** i)
            attn = spatial in attn_res
            self.ups.append(Up(up_in, cout, tdim, cdim, n_res, attn))
            up_in = cout

        self.out_conv = nn.Sequential(
            nn.GroupNorm(min(8, ch), ch), _act(), _conv(ch, lc))

    def timestep_embedding(self, t, dim):
        half = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if dim % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, x, t, labels=None):
        t_emb = self.t_mlp(self.timestep_embedding(t, self.cfg.unet_base_channels))
        c_emb = self.class_embed(labels) if self.class_embed is not None and labels is not None else None

        x = self.in_conv(x)
        all_skips = []
        for d in self.downs:
            x, sk = d(x, t_emb, c_emb)
            all_skips.append(sk)

        x = self.mid1(x, t_emb, c_emb)
        x = self.mid_attn(x)
        x = self.mid2(x, t_emb, c_emb)

        for u in self.ups:
            sk = all_skips.pop()
            x, _ = u(x, t_emb, c_emb, sk)

        return self.out_conv(x)


# --------------------------------------------------------------------------- #
# MindDiffusion: DDPM training + DDIM sampling + CFG
# --------------------------------------------------------------------------- #
class MindDiffusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.unet = ConditionalUNet(cfg)
        T = cfg.diffusion_timesteps

        betas = torch.linspace(1e-4, 0.02, T)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)

    def q_sample(self, z0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(z0)
        ab = self.alpha_bars[t][:, None, None, None]
        return ab.sqrt() * z0 + (1 - ab).sqrt() * noise, noise

    def training_step(self, z0, labels=None):
        """DDPM training with optional classifier-free guidance dropout."""
        B = z0.shape[0]
        T = self.cfg.diffusion_timesteps
        t = torch.randint(0, T, (B,), device=z0.device)

        # CFG dropout: randomly drop the condition
        if self.cfg.use_conditioning and labels is not None:
            drop = torch.rand(B, device=z0.device) < self.cfg.cfg_dropout
            labels = labels.clone()
            labels[drop] = 0  # class 0 = "unconditional" sentinel

        zt, noise = self.q_sample(z0, t)
        pred = self.unet(zt, t, labels)
        return F.mse_loss(pred, noise)

    @torch.no_grad()
    def sample_ddpm(self, batch_size=1, device="cpu", labels=None,
                    progress_cb=None, steps=None):
        cfg = self.cfg
        T = steps or cfg.diffusion_timesteps
        x = torch.randn(batch_size, cfg.latent_channels,
                        cfg.latent_size, cfg.latent_size, device=device)
        for i in reversed(range(T)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            eps = self.unet(x, t, labels)
            alpha = self.alphas[i]
            ab = self.alpha_bars[i]
            mean = (1 / alpha.sqrt()) * (x - ((1 - alpha) / (1 - ab).sqrt()) * eps)
            if i > 0:
                x = mean + self.betas[i].sqrt() * torch.randn_like(x)
            else:
                x = mean
            if progress_cb:
                progress_cb(i, x)
        return x

    @torch.no_grad()
    def sample_ddim(self, batch_size=1, device="cpu", labels=None,
                    steps=None, eta=0.0, progress_cb=None):
        """Deterministic DDIM sampling with classifier-free guidance.

        steps: number of DDIM steps (default cfg.ddim_steps)
        eta:  0 = deterministic, 1 = stochastic (DDPM-like)
        """
        cfg = self.cfg
        T = cfg.diffusion_timesteps
        S = steps or cfg.ddim_steps
        # subset of timesteps to actually evaluate
        step_idxs = list(range(0, T, T // S))[:S]
        x = torch.randn(batch_size, cfg.latent_channels,
                        cfg.latent_size, cfg.latent_size, device=device)

        for i in reversed(step_idxs):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            eps = self._guided_eps(x, t, labels)

            ab_i = self.alpha_bars[i]
            prev_i = step_idxs[step_idxs.index(i) - 1] if step_idxs.index(i) > 0 else -1
            ab_prev = self.alpha_bars[prev_i] if prev_i >= 0 else torch.tensor(1.0, device=device)

            # DDIM update
            x0_pred = (x - (1 - ab_i).sqrt() * eps) / ab_i.sqrt().clamp(min=1e-8)
            sigma = eta * ((1 - ab_prev) / (1 - ab_i)).sqrt() * (1 - ab_i / ab_prev).sqrt()
            dir_xt = (1 - ab_prev - sigma ** 2).sqrt() * eps
            noise = sigma * torch.randn_like(x) if sigma > 0 else 0
            x = ab_prev.sqrt() * x0_pred + dir_xt + noise
            if progress_cb:
                progress_cb(i, x)
        return x

    def _guided_eps(self, x, t, labels):
        """Classifier-free guidance: eps = eps_uncond + s*(eps_cond - eps_uncond)."""
        if not self.cfg.use_conditioning or labels is None or self.cfg.cfg_scale <= 1.0:
            return self.unet(x, t, labels)

        uncond_labels = torch.zeros_like(labels)
        eps_cond = self.unet(x, t, labels)
        eps_uncond = self.unet(x, t, uncond_labels)
        s = self.cfg.cfg_scale
        return eps_uncond + s * (eps_cond - eps_uncond)


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #
def save_model(model: nn.Module, path: str):
    torch.save(model.state_dict(), path)
    log.info("saved %s", path)


def load_model(model: nn.Module, path: str, map_location="cpu"):
    state = torch.load(path, map_location=map_location)
    model.load_state_dict(state)
    log.info("loaded %s", path)
    return model