# AGENTS.md - mind-diffusion

## Commands
```powershell
# Environment (PowerShell)
$env:PYTHONPATH = "src"; $env:OMP_NUM_THREADS = "1"

# Prepare data (multi-source: LGG brain MRI + T1w + procedural)
python scripts/prepare_data.py --force

# Generate synthetic connectome
python scripts/make_connectome.py --nodes 256

# Train VAE on DirectML (GPU, 5x faster)
python scripts/train_vae.py --device directml --epochs 30

# Train diffusion on CPU (DirectML slower for UNet attention/GroupNorm)
python scripts/train_diffusion.py --device cpu --epochs 30

# Simulate (6 modes)
python scripts/run_simulation.py --mode ddim_interpolate --steps 60 --tag final
python scripts/run_simulation.py --mode random_walk --steps 60 --tag final
python scripts/run_simulation.py --mode ode --steps 60 --tag final
python scripts/run_simulation.py --mode connectome --steps 60 --tag final --connectome data/connectome.npy
python scripts/run_simulation.py --mode knot --steps 60 --tag balanced --device cpu
python scripts/run_simulation.py --mode knot --steps 60 --tag voices --bicameral 0.9 --device cpu
python scripts/run_simulation.py --mode world_knot --steps 60 --tag embodied --device cpu --bicameral 0.3
python scripts/run_simulation.py --mode world_knot --steps 60 --tag embodied_voices --device cpu --bicameral 0.9

# Train JEPA world model on synthetic terminal sessions (or --jsonl real logs)
python scripts/train_jepa.py --epochs 60 --sessions 512 --batch 16

# Analysis (latent space, reconstructions, trajectory plots)
python scripts/analyze.py --device cpu

# Smoke test
python tests/smoke.py

# Benchmark DirectML vs CPU
python tests/benchmark_device.py
```

## Architecture
- `src/mind_diffusion/config.py` - all hyperparameters (dataclass `cfg`)
- `src/mind_diffusion/models.py` - VAE, ConditionalUNet, MindDiffusion (DDPM/DDIM + CFG)
- `src/mind_diffusion/simulation.py` - 6 trajectory modes: random_walk, ddim_interpolate, connectome, ode, knot, world_knot
- `src/mind_diffusion/data.py` - multi-source collection, train/val splits, augmentation, class labels
- `src/mind_diffusion/terminal_world.py` - synthetic virtual shell (FSM, error/recovery dynamics) + real log ingest
- `src/mind_diffusion/mind_moljepa.py` - molecular world (MolJEPA Koopman) -> BridgeNet -> mind latent
- `src/mind_diffusion/mind_jepa_bridge.py` - terminal/physical world JEPA (latent-only, EMA target, FiLM action conditioning, PLDM time-VICReg) + coupled mind-world knot loop; run JEPA parts on CPU (DML chokes on GRU)

## Key findings
- DirectML 5x faster for VAE (conv-heavy) but slower for UNet (GroupNorm/attention fall back to CPU)
- VAE PSNR ~21.7 dB on validation; DDIM samples match latent distribution
- Diffusion loss converges ~0.21 after 30 epochs
- torch-directml 0.2.5 pins torch==2.4.1 (+cpu wheel); upgrading torch breaks all imports (c10_cuda.dll WinError 127)
- Knot mode: bicameral=0.9 locks the loop into self-echo (pixel change 4x lower than balanced b=0.3) - Jaynes' "hearing voices" as attractor collapse
- World-knot: same collapse with a trained JEPA world on the action-in channel (err_cos 0.68 vs -0.33, pixel change 6x lower); world answer needs sensory noise or the loop freezes
- JEPA action alignment matters: obs[t+1] is caused by act[t+1]; off-by-one training yields gap~0 (action-blind). Dynamics normalization + FiLM conditioning lifted action gap 0 -> 0.25
- Train/val split on synthetic sessions overfits after ~15 epochs; checkpoint saved on (val_loss - action_gap) score, not val loss alone

## Lint/typecheck
No linter configured. Verify with: `python tests/smoke.py`