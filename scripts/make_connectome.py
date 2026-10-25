"""Generate a synthetic connectome adjacency matrix for the connectome-driven
simulation mode, when a real connectome file is not available.

Creates a small-world network (Watts-Strogatz style) with N nodes, saves as
.npy. The simulation engine projects this to the latent dimension via SVD.

Usage:
    python scripts/make_connectome.py [--nodes 256] [--out data/connectome.npy]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("connectome")


def watts_strogatz(n, k=8, p=0.1, seed=42):
    """Generate a small-world adjacency matrix (Watts-Strogatz)."""
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=np.float32)
    # ring lattice
    for i in range(n):
        for j in range(1, k // 2 + 1):
            adj[i, (i + j) % n] = 1.0
            adj[i, (i - j) % n] = 1.0
    # rewire
    for i in range(n):
        for j in range(n):
            if adj[i, j] > 0 and rng.random() < p:
                adj[i, j] = 0
                new = rng.integers(0, n)
                while new == i:
                    new = rng.integers(0, n)
                adj[i, new] = 1.0
    # add weights (log-normal, mimicking synaptic strength heterogeneity)
    mask = adj > 0
    adj[mask] = rng.lognormal(mean=0, sigma=1, size=mask.sum()).astype(np.float32)
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, default=256,
                    help="number of nodes (should match latent flat dim or be larger)")
    ap.add_argument("--k", type=int, default=8, help="ring degree")
    ap.add_argument("--p", type=float, default=0.1, help="rewiring prob")
    ap.add_argument("--out", type=str, default="data/connectome.npy")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    log.info("Generating Watts-Strogatz connectome: n=%d k=%d p=%.2f",
             args.nodes, args.k, args.p)
    adj = watts_strogatz(args.nodes, args.k, args.p, args.seed)
    log.info("Adjacency: %dx%d, %d edges, mean weight %.3f",
             adj.shape[0], adj.shape[1], int((adj > 0).sum()), float(adj[adj > 0].mean()))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.save(args.out, adj)
    log.info("Saved connectome -> %s", args.out)


if __name__ == "__main__":
    main()