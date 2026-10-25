"""Download / prepare data for mind-diffusion.

Usage:
    python scripts/download_data.py [--procedural-only]

Tries the Hugging Face brain-MRI slice dataset first; if that fails
(or --procedural-only) it generates synthetic neuron images so the
project trains end-to-end offline.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mind_diffusion.config import cfg
from mind_diffusion import data as data_mod

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s | %(message)s",
)
log = logging.getLogger("download_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procedural-only", action="store_true",
                    help="skip Hugging Face download")
    ap.add_argument("--n", type=int, default=None,
                    help="override procedural_samples")
    args = ap.parse_args()

    if args.n is not None:
        cfg.procedural_samples = args.n

    Path(cfg.processed_dir).mkdir(parents=True, exist_ok=True)

    if not args.procedural_only:
        src = data_mod.download_hf_dataset(cfg)
        if src:
            n_real = data_mod.collect_images_to_processed(cfg, src)
            log.info("Real images collected: %d", n_real)
        else:
            log.warning("HF download unavailable - using procedural data only.")

    # always ensure we have enough images
    existing = len(list(data_mod._iter_image_files(Path(cfg.processed_dir))))
    need = max(0, cfg.procedural_samples - existing)
    if need > 0 or existing == 0:
        log.info("Ensuring %d procedural samples (have %d, need %d)",
                 cfg.procedural_samples, existing, need)
        data_mod.generate_procedural_neurons(cfg)

    final = len(list(data_mod._iter_image_files(Path(cfg.processed_dir))))
    log.info("Done. %d images ready in %s", final, cfg.processed_dir)


if __name__ == "__main__":
    main()