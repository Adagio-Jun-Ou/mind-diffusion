"""Global configuration for the mind-diffusion project.

Keep every tunable in one place so the heavy training / simulation scripts
remain short and readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    # ---- image / latent sizes -------------------------------------------------
    image_size: int = 64          # px (square); kept small so CPU/DirectML trains fast
    latent_size: int = 8          # the "mind latent space" - a compact 8x8 grid
    latent_channels: int = 4      # channels per latent pixel
    image_channels: int = 1       # grayscale neuron / brain images

    # ---- VAE -------------------------------------------------------------------
    vae_base_channels: int = 32
    vae_lr: float = 1e-3
    vae_epochs: int = 30
    vae_kl_weight: float = 1e-5   # very light KL so reconstructions stay crisp
    vae_perceptual_weight: float = 0.0  # 0 = off; >0 adds simple L1 on edges

    # ---- Diffusion (DDPM/DDIM) ------------------------------------------------
    diffusion_timesteps: int = 1000
    diffusion_lr: float = 2e-4
    diffusion_epochs: int = 50
    unet_base_channels: int = 64
    unet_channel_mults: tuple = (1, 2, 4)
    unet_num_res_blocks: int = 2
    unet_attention_resolutions: tuple = (2, 4)  # latent spatial sizes that use attention
    # DDIM sampling: fewer steps than training schedule
    ddim_steps: int = 50
    ddim_eta: float = 0.0         # 0 = deterministic DDIM, 1 = DDPM-equivalent
    # classifier-free guidance
    cfg_dropout: float = 0.1      # prob of dropping condition during training
    cfg_scale: float = 2.0        # guidance scale at sampling time

    # ---- conditioning ---------------------------------------------------------
    num_classes: int = 64         # max patient/region labels for conditioning
    cond_embed_dim: int = 32      # embedding dim for class condition
    use_conditioning: bool = True

    # ---- "Mind" simulation ----------------------------------------------------
    sim_steps: int = 60           # frames in a thought trajectory
    sim_seed_latents: bool = True
    # trajectory mode: "random_walk" | "ddim_interpolate" | "connectome" | "ode" | "knot"
    sim_mode: str = "ddim_interpolate"
    sim_step_scale: float = 0.35  # std of random-walk perturbation
    sim_connectome_path: str = ""  # path to connectome adjacency (.npy)

    # endless-knot loop (closed perception <-> generation cycle)
    # bicameral dial: 0 = fully perceptual (driven by sensory prediction error),
    #                 1 = fully bicameral (driven by internal command only)
    sim_bicameral: float = 0.3
    sim_knot_error_gain: float = 1.0   # gain on the sensory prediction error
    sim_garden_margin: float = 1.5     # walled garden radius = margin * seed latent norm

    # ---- data -----------------------------------------------------------------
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    checkpoint_dir: str = "checkpoints"
    output_dir: str = "output"

    # multi-source datasets: list of (source_id, glob pattern relative to raw_dir)
    # source_id is used as a prefix for conditioning labels
    data_sources: tuple = (
        "lgg",          # Brain-MRI-LGG-Segmentation (15.7k slices, patient labels)
        "t1w",           # g4m3r/T1w_MRI_Brain_Slices (9k slices)
        "procedural",    # synthetic neuron generator
    )
    # how many images to collect per source (0 = all)
    max_per_source: int = 0
    val_fraction: float = 0.1     # fraction of data for validation
    augment: bool = True           # random flips + small rotations

    # ---- training -------------------------------------------------------------
    batch_size: int = 64
    num_workers: int = 0           # Windows: 0 is safest with DirectML
    seed: int = 42

    # ---- device ---------------------------------------------------------------
    # "auto" tries directml -> cuda -> cpu
    # "vae_directml" trains VAE on DirectML, diffusion on CPU (benchmark-optimal)
    device: str = "auto"

    # ---- dataset source (legacy single-source, still used by download_data.py) -
    hf_dataset_id: str = "g4m3r/T1w_MRI_Brain_Slices"
    use_procedural_fallback: bool = True
    procedural_samples: int = 2000   # generated if HF download fails / offline

    # ---- where to log ---------------------------------------------------------
    log_interval: int = 20
    save_every: int = 5           # save checkpoint every N epochs


cfg = Config()