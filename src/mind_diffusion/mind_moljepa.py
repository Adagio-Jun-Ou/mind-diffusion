"""MindMolJEPA — bridging molecular dynamics (MolJEPA) with mind states (mind-diffusion).

The core idea: biomolecular dynamics at the molecular level (protein conformational
changes, neurotransmitter release, ion channel gating) DRIVE neural activity
and mind states. This module provides the bridge:

    z_mol (molecular latent, from MolJEPA EGNN+Koopman)
        -> bridge (cross-modal projection + temporal alignment)
        -> z_mind (mind latent, drives VAE decoder -> brain images)

The bridge is a learned mapping from the molecular world model's latent space
to the mind-diffusion latent space, conditioned on the molecular dynamics.

Architecture:
    1. MolJEPA encodes protein/MD trajectory frames -> z_mol (B, latent_dim_mol)
    2. Koopman advances z_mol forward in time (molecular dynamics prediction)
    3. BridgeNet projects z_mol -> z_mind (cross-modal, temporal)
    4. mind-diffusion VAE decodes z_mind -> brain state images
    5. Connectome operator (optional) advances z_mind -> simulates "thinking"

This creates a multi-scale world model: molecules -> neurons -> mind.

Usage:
    from mind_moljepa import MindMolJEPA
    model = MindMolJEPA(mol_ckpt, mind_vae_ckpt, mind_diff_ckpt, device)
    frames = model.simulate_mind_from_molecule(protein_coords, n_steps=100)
"""
import os, sys, math, logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger("mind_moljepa")


class BridgeNet(nn.Module):
    """Cross-modal bridge: molecular latent -> mind latent.

    Projects the MolJEPA Koopman latent (molecular dynamics state) into the
    mind-diffusion latent grid (brain state). Uses a FiLM-conditioned MLP
    that maps z_mol -> z_mind_grid (latent_channels x latent_size x latent_size).

    The bridge is trained jointly: minimize ||z_mind_pred - z_mind_target||
    where z_mind_target is the VAE encoding of a brain image at the same
    (simulated) time as the molecular state.
    """

    def __init__(self, mol_latent_dim=128, mind_latent_channels=4,
                 mind_latent_size=8, hidden_dim=256):
        super().__init__()
        self.mind_latent_channels = mind_latent_channels
        self.mind_latent_size = mind_latent_size
        self.mind_flat_dim = mind_latent_channels * mind_latent_size * mind_latent_size
        self.proj = nn.Sequential(
            nn.Linear(mol_latent_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, self.mind_flat_dim),
        )
        # temporal modulation: the molecular state at different times
        # should map to different mind states (dynamics -> experience)
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, z_mol, t=None):
        """Project molecular latent to mind latent grid.

        Args:
            z_mol: (B, mol_latent_dim) molecular latent from Koopman
            t: (B,) optional time embedding (normalized 0-1)
        Returns:
            z_mind: (B, latent_channels, latent_size, latent_size) mind latent
        """
        h = self.proj(z_mol)  # (B, mind_flat_dim)
        if t is not None:
            half = self.time_embed[0].out_features
            t_emb = torch.zeros(z_mol.shape[0], half, device=z_mol.device)
            # sinusoidal time encoding
            freqs = torch.exp(-math.log(10000) * torch.arange(half // 2, device=z_mol.device) / (half // 2))
            args = t.unsqueeze(-1) * freqs.unsqueeze(0)
            t_emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
            h = h + self.time_embed(t_emb)  # temporal modulation
        z_mind = h.view(-1, self.mind_latent_channels,
                        self.mind_latent_size, self.mind_latent_size)
        return z_mind


class MindMolJEPA(nn.Module):
    """Full multi-scale world model: molecules -> neurons -> mind.

    Combines:
    - MolJEPA (molecular world model: EGNN encoder + Koopman dynamics)
    - BridgeNet (cross-modal molecular -> mind latent projection)
    - mind-diffusion VAE (mind latent -> brain images)
    - Connectome operator (mind latent dynamics -> "thinking")
    """

    def __init__(self, mol_latent_dim=128, mind_latent_channels=4,
                 mind_latent_size=8, mind_image_size=64, hidden_dim=256):
        super().__init__()
        self.bridge = BridgeNet(mol_latent_dim, mind_latent_channels,
                                 mind_latent_size, hidden_dim)
        # Note: the MolJEPA model and mind-diffusion VAE are loaded separately
        # and passed at inference time. This class owns the bridge + connectome.

    def forward(self, z_mol, t=None):
        """Project molecular latent to mind latent."""
        return self.bridge(z_mol, t)

    def bridge_loss(self, z_mol, z_mind_target, t=None):
        """Training loss for the bridge: MSE in mind latent space."""
        z_mind_pred = self.bridge(z_mol, t)
        return F.mse_loss(z_mind_pred, z_mind_target)

    @torch.no_grad()
    def simulate_mind_from_molecule(self, mol_model, atom_types, pos_ctx,
                                    n_steps=100, temperature=0.05,
                                    vae=None, connectome_W=None,
                                    device='cuda'):
        """Simulate mind states driven by molecular dynamics.

        1. Encode molecular frame -> z_mol
        2. Advance z_mol via Koopman (molecular dynamics)
        3. Bridge z_mol -> z_mind (cross-modal projection)
        4. Optionally advance z_mind via connectome (neural dynamics)
        5. Decode z_mind -> brain image (if VAE provided)

        Returns list of (z_mol, z_mind, brain_image) tuples.
        """
        results = []
        z_mol = mol_model.encode(atom_types, pos_ctx)

        for step in range(n_steps):
            # advance molecular dynamics
            z_mol = mol_model.koop.advance(z_mol, steps=1, stochastic=True,
                                           temperature=temperature)
            # bridge to mind latent
            t = torch.tensor([step / n_steps], device=device).float()
            z_mind = self.bridge(z_mol, t)

            # optional connectome dynamics on mind latent
            if connectome_W is not None:
                z_flat = z_mind.flatten(1)
                z_flat = z_flat @ connectome_W.T + 0.01 * torch.randn_like(z_flat)
                z_mind = z_flat.view(z_mind.shape)

            # decode to brain image
            brain_img = None
            if vae is not None:
                brain_img = vae.decode(z_mind)

            results.append({
                'step': step,
                'z_mol': z_mol.cpu().numpy(),
                'z_mind': z_mind.cpu().numpy(),
                'brain_image': brain_img.cpu().numpy() if brain_img is not None else None,
            })

        return results


def train_bridge(mol_model, bridge, vae, mol_data, mind_data,
                 epochs=1000, lr=1e-3, device='cuda'):
    """Train the BridgeNet: align molecular latents with mind latents.

    Args:
        mol_model: trained MolJEPAWorld (frozen encoder + Koopman)
        bridge: BridgeNet (trainable)
        vae: trained mind-diffusion VAE (frozen, for encoding mind targets)
        mol_data: molecular trajectory data (coords, atom_types)
        mind_data: brain image data aligned (temporally) with mol_data
    """
    mol_model.eval()
    vae.eval()
    opt = torch.optim.AdamW(bridge.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    log.info("Training bridge: %d epochs, lr=%g", epochs, lr)
    for epoch in range(epochs):
        # sample molecular frame + aligned mind frame
        idx = np.random.randint(0, len(mol_data['coords']))
        pos = torch.from_numpy(mol_data['coords'][idx]).float().to(device).unsqueeze(0)
        at = torch.from_numpy(mol_data['atom_types']).long().to(device).unsqueeze(0)

        # encode molecular frame -> z_mol
        with torch.no_grad():
            z_mol = mol_model.encode(at, pos)

        # encode mind frame -> z_mind_target
        mind_img = torch.from_numpy(mind_data[idx]).float().to(device).unsqueeze(0)
        with torch.no_grad():
            z_mind_target = vae.encode(mind_img)

        # train bridge
        loss = bridge.bridge_loss(z_mol, z_mind_target,
                                  t=torch.tensor([idx / len(mol_data)], device=device).float())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        if epoch % 100 == 0:
            log.info("[bridge %5d] loss=%.6f", epoch, loss.item())

    log.info("Bridge training done. Final loss=%.6f", loss.item())
    return bridge


if __name__ == "__main__":
    print("MindMolJEPA: bridging molecular dynamics with mind states")
    print("  z_mol (MolJEPA) -> BridgeNet -> z_mind (mind-diffusion VAE) -> brain images")
    print("  Train with: train_bridge(mol_model, bridge, vae, mol_data, mind_data)")