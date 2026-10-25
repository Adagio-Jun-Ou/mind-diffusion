"""Benchmark DirectML vs CPU for the mind-diffusion models.

Runs each device in a separate subprocess to avoid the DirectML autograd
engine caching issues when switching devices in-process.

Usage:  python tests/benchmark_device.py
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable


_BENCH_SCRIPT = textwrap.dedent("""
    import os, sys, time, json
    sys.path.insert(0, "src")
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from mind_diffusion.config import cfg
    from mind_diffusion.models import VAE, MindDiffusion

    DEVICE = {device!r}

    def bench(fn, iters=3, warmup=1):
        for _ in range(warmup):
            fn()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        return (time.perf_counter() - t0) / iters

    if DEVICE == "directml":
        import torch_directml
        dev = torch_directml.device()
    else:
        dev = "cpu"

    vae = VAE(cfg).to(dev)
    diff = MindDiffusion(cfg).to(dev)
    B = 16
    x = torch.randn(B, cfg.image_channels, cfg.image_size, cfg.image_size, device=dev)
    z = torch.randn(B, cfg.latent_channels, cfg.latent_size, cfg.latent_size, device=dev)

    opt_v = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    def vae_step():
        opt_v.zero_grad()
        recon, _ = vae(x)
        (recon - x).pow(2).mean().backward()
        opt_v.step()
    t_vae = bench(vae_step, iters=3)

    opt_d = torch.optim.AdamW(diff.parameters(), lr=2e-4)
    def diff_step():
        opt_d.zero_grad()
        diff.training_step(z).backward()
        opt_d.step()
    t_diff = bench(diff_step, iters=3)

    cfg.diffusion_timesteps = 50
    diff2 = MindDiffusion(cfg).to(dev)
    t_sample = bench(lambda: diff2.sample(batch_size=1, device=dev), iters=1, warmup=0)

    print("RESULT", json.dumps({{
        "vae": t_vae, "diff": t_diff, "sample": t_sample
    }}))
""")


def run_in_subprocess(device: str) -> dict:
    script = _BENCH_SCRIPT.format(device=device)
    res = subprocess.run(
        [PY, "-c", script],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=600,
        env={**os.environ, "PYTHONPATH": "src", "OMP_NUM_THREADS": "1"},
    )
    for line in res.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line.split(" ", 1)[1])
    print("STDOUT:", res.stdout[-2000:])
    print("STDERR:", res.stderr[-2000:])
    raise RuntimeError(f"bench failed for {device}")


if __name__ == "__main__":
    print("Benchmarking CPU ...")
    cpu = run_in_subprocess("cpu")
    print(f"  VAE={cpu['vae']*1000:.1f}ms  Diff={cpu['diff']*1000:.1f}ms  Sample={cpu['sample']*1000:.0f}ms")

    print("Benchmarking DirectML ...")
    try:
        dml = run_in_subprocess("directml")
        print(f"  VAE={dml['vae']*1000:.1f}ms  Diff={dml['diff']*1000:.1f}ms  Sample={dml['sample']*1000:.0f}ms")
    except Exception as e:
        print(f"  DirectML bench failed: {e}")
        dml = None

    print("\n=== Summary ===")
    print(f"{'Metric':<16} {'CPU':>12} {'DirectML':>12} {'Speedup':>10}")
    for k in ["vae", "diff", "sample"]:
        cv = cpu[k] * 1000
        if dml:
            dv = dml[k] * 1000
            print(f"  {k:<14} {cv:>10.1f}ms {dv:>10.1f}ms {cpu[k]/dml[k]:>8.2f}x")
        else:
            print(f"  {k:<14} {cv:>10.1f}ms {'n/a':>10} {'n/a':>8}")