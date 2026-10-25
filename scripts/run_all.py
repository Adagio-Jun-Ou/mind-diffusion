"""Run the entire v2 pipeline: data -> connectome -> train VAE -> train diff -> simulate -> analyze.

Usage:
    python scripts/run_all.py [--quick]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
log = logging.getLogger("run_all")


def run(cmd):
    log.info(">>> %s", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        run([PY, str(HERE / "prepare_data.py"), "--force"])
        run([PY, str(HERE / "make_connectome.py")])
        run([PY, str(HERE / "train_vae.py"), "--epochs", "3", "--batch-size", "32", "--device", "cpu"])
        run([PY, str(HERE / "train_diffusion.py"), "--epochs", "3", "--batch-size", "32", "--device", "cpu"])
        run([PY, str(HERE / "run_simulation.py"), "--mode", "ddim_interpolate", "--steps", "20", "--tag", "quick", "--device", "cpu"])
    else:
        run([PY, str(HERE / "prepare_data.py"), "--force"])
        run([PY, str(HERE / "make_connectome.py")])
        run([PY, str(HERE / "train_vae.py"), "--epochs", "30", "--device", "directml"])
        run([PY, str(HERE / "train_diffusion.py"), "--epochs", "30", "--device", "cpu"])
        for mode in ["ddim_interpolate", "random_walk", "ode", "connectome"]:
            run([PY, str(HERE / "run_simulation.py"), "--mode", mode, "--steps", "60", "--tag", "final", "--device", "cpu"])
        run([PY, str(HERE / "analyze.py"), "--device", "cpu"])

    log.info("All steps complete. See output/ for results.")


if __name__ == "__main__":
    main()