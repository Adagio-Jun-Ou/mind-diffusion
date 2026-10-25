"""Prepare all data sources for mind-diffusion (v2 multi-source).

Collects from: LGG brain MRI, T1w brain slices, procedural neurons.
Encodes class labels in filenames for conditioning.

Usage:
    python scripts/prepare_data.py [--force]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mind_diffusion.config import cfg
from mind_diffusion.data import prepare_all_data, download_hf_dataset

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s | %(message)s")
log = logging.getLogger("prepare_data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-process even if images exist")
    args = ap.parse_args()

    # ensure t1w source is downloaded (lgg is already extracted manually)
    if "t1w" in cfg.data_sources:
        src = Path(cfg.raw_dir) / cfg.hf_dataset_id.replace("/", "_")
        if not src.exists():
            log.info("Downloading T1w source...")
            download_hf_dataset(cfg)

    Path(cfg.processed_dir).mkdir(parents=True, exist_ok=True)
    label_map = prepare_all_data(cfg, force=args.force)
    total = len(list(Path(cfg.processed_dir).glob("*.png")))
    log.info("Done. %d images, %d classes -> %s", total, len(label_map), cfg.processed_dir)
    log.info("Label map: %s", dict(list(label_map.items())[:20]))


if __name__ == "__main__":
    main()