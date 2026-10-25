"""Data pipeline for mind-diffusion (v2 - multi-source, splits, conditioning).

Sources (in priority order):
  1. Brain-MRI-LGG-Segmentation  - 15.7k T1w brain slices, patient IDs as labels
  2. g4m3r/T1w_MRI_Brain_Slices - 9k T1w axial slices
  3. procedural neuron generator - synthetic offline fallback

Each image gets a integer "class" label (patient/source id) used for
classifier-free guidance in the diffusion model.
"""

from __future__ import annotations

import io
import logging
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger("mind-diffusion.data")

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _iter_image_files(root: Path, exclude_masks: bool = True):
    for p in root.rglob("*"):
        if p.suffix.lower() not in IMG_EXTS:
            continue
        if exclude_masks and "_mask" in p.name.lower():
            continue
        yield p


def _load_gray(path: Path, size: int) -> np.ndarray:
    img = Image.open(path).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Source collectors
# --------------------------------------------------------------------------- #
def collect_lgg(cfg, processed: Path, label_map: dict) -> int:
    """Collect Brain-MRI-LGG-Segmentation slices. Patient folder = class label."""
    src = Path(cfg.raw_dir) / "Brain-MRI-LGG-Segmentation" / "extracted" / "kaggle_3m"
    if not src.exists():
        log.info("LGG source not found at %s, skipping", src)
        return 0
    n = 0
    patients = sorted([d for d in src.iterdir() if d.is_dir()])
    for pi, pdir in enumerate(patients):
        if cfg.num_classes and pi >= cfg.num_classes:
            break
        label = label_map.setdefault(f"lgg_{pdir.name}", len(label_map))
        for img_path in sorted(pdir.glob("*.tif")):
            if "_mask" in img_path.name.lower():
                continue
            try:
                arr = _load_gray(img_path, cfg.image_size)
                out = processed / f"lgg_{n:05d}_c{label}.png"
                Image.fromarray((arr * 255).astype(np.uint8), "L").save(out)
                n += 1
            except Exception as exc:
                log.debug("skip %s: %s", img_path, exc)
        if n % 500 == 0 and n > 0:
            log.info("  lgg: %d images so far", n)
    log.info("LGG: collected %d images, %d classes", n, len(label_map))
    return n


def collect_t1w(cfg, processed: Path, label_map: dict) -> int:
    """Collect g4m3r/T1w_MRI_Brain_Slices."""
    src = Path(cfg.raw_dir) / cfg.hf_dataset_id.replace("/", "_")
    if not src.exists():
        log.info("T1w source not found at %s, skipping", src)
        return 0
    label = label_map.setdefault("t1w_default", len(label_map))
    n = 0
    for img_path in _iter_image_files(src):
        try:
            arr = _load_gray(img_path, cfg.image_size)
            out = processed / f"t1w_{n:05d}_c{label}.png"
            Image.fromarray((arr * 255).astype(np.uint8), "L").save(out)
            n += 1
            if n % 500 == 0:
                log.info("  t1w: %d images", n)
        except Exception as exc:
            log.debug("skip %s: %s", img_path, exc)
    log.info("T1w: collected %d images", n)
    return n


# --------------------------------------------------------------------------- #
# Procedural neuron generator
# --------------------------------------------------------------------------- #
def _draw_neuron(rng, size=64):
    canvas = np.zeros((size, size), dtype=np.float32)
    cx, cy = rng.uniform(0.3, 0.7) * size, rng.uniform(0.3, 0.7) * size
    soma_r = rng.uniform(1.5, 3.5)
    yy, xx = np.ogrid[:size, :size]
    canvas += np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * soma_r ** 2))) * rng.uniform(0.7, 1.0)
    n_branches = rng.integers(4, 9)
    for _ in range(n_branches):
        x, y = cx, cy
        ang = rng.uniform(0, 2 * math.pi)
        thickness = rng.uniform(0.6, 1.8)
        intensity = rng.uniform(0.4, 0.9)
        for _step in range(rng.integers(20, size)):
            ang += rng.normal(0, 0.25)
            x += math.cos(ang) * rng.uniform(0.8, 1.6)
            y += math.sin(ang) * rng.uniform(0.8, 1.6)
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < size and 0 <= yi < size:
                r = int(max(1, thickness))
                for dy in range(-r, r + 1):
                    for dx in range(-r, r + 1):
                        nx, ny = xi + dx, yi + dy
                        if 0 <= nx < size and 0 <= ny < size and dx*dx + dy*dy <= r*r:
                            canvas[ny, nx] = max(canvas[ny, nx], intensity)
            if rng.random() < 0.04:
                ang2 = ang + rng.choice([-1, 1]) * rng.uniform(0.5, 1.2)
                x2, y2 = x, y
                for _s2 in range(rng.integers(8, 25)):
                    ang2 += rng.normal(0, 0.25)
                    x2 += math.cos(ang2)
                    y2 += math.sin(ang2)
                    xi, yi = int(round(x2)), int(round(y2))
                    if 0 <= xi < size and 0 <= yi < size:
                        canvas[yi, xi] = max(canvas[yi, xi], intensity * 0.7)
    canvas += rng.uniform(0.0, 0.05)
    return np.clip(canvas, 0, 1)


def generate_procedural_neurons(cfg, processed: Path, label_map: dict) -> int:
    rng = np.random.default_rng(cfg.seed)
    label = label_map.setdefault("procedural", len(label_map))
    n = 0
    for i in range(cfg.procedural_samples):
        img = _draw_neuron(rng, size=cfg.image_size)
        out = processed / f"proc_{i:05d}_c{label}.png"
        Image.fromarray((img * 255).astype(np.uint8), "L").save(out)
        n += 1
    log.info("Procedural: generated %d images", n)
    return n


# --------------------------------------------------------------------------- #
# Master prepare: collect all sources, build label map
# --------------------------------------------------------------------------- #
def prepare_all_data(cfg, force: bool = False) -> dict:
    """Collect from all configured sources into processed_dir.

    Filenames encode the class label as ``_c<idx>.png``.
    Returns the label_map dict.
    """
    processed = Path(cfg.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)

    existing = list(processed.glob("*.png"))
    if existing and not force:
        log.info("Processed dir has %d images, rebuilding label map...", len(existing))
        label_map = {}
        for p in existing:
            if "_c" in p.stem:
                cstr = p.stem.rsplit("_c", 1)[1]
                try:
                    label_map.setdefault(f"_c{cstr}", int(cstr))
                except ValueError:
                    pass
        log.info("Label map: %d classes", len(label_map))
        return label_map

    label_map: dict = {}

    if "lgg" in cfg.data_sources:
        collect_lgg(cfg, processed, label_map)
    if "t1w" in cfg.data_sources:
        collect_t1w(cfg, processed, label_map)
    if "procedural" in cfg.data_sources:
        existing = len(list(processed.glob("*.png")))
        need = max(0, cfg.procedural_samples - existing)
        if need > 0:
            generate_procedural_neurons(cfg, processed, label_map)

    total = len(list(processed.glob("*.png")))
    log.info("Total processed images: %d, classes: %d", total, len(label_map))
    return label_map


# --------------------------------------------------------------------------- #
# Dataset + augmentations
# --------------------------------------------------------------------------- #
def _augment(arr: np.ndarray, rng) -> np.ndarray:
    if rng.random() < 0.5:
        arr = arr[:, ::-1].copy()
    if rng.random() < 0.5:
        arr = arr[::-1, :].copy()
    if rng.random() < 0.3:
        k = rng.integers(1, 4)
        arr = np.rot90(arr, k=k).copy()
    return arr


class NeuronDataset:
    """Dataset yielding (image_tensor, class_label) pairs.

    Filenames must encode label as ``..._c<idx>.png``.
    """
    SPLIT_TRAIN = "train"
    SPLIT_VAL = "val"

    def __init__(self, cfg, split="train", transform=None):
        import torch
        self.cfg = cfg
        self.transform = transform
        self.rng = np.random.default_rng(cfg.seed + (0 if split == "train" else 1))

        all_files = sorted(Path(cfg.processed_dir).glob("*.png"))
        if not all_files:
            raise RuntimeError(f"No images in {cfg.processed_dir}. Run prepare_data.py")

        # deterministic split
        n_val = int(len(all_files) * cfg.val_fraction)
        rng = np.random.default_rng(cfg.seed)
        perm = rng.permutation(len(all_files))
        if split == "train":
            idxs = perm[n_val:]
        else:
            idxs = perm[:n_val]
        self.files = [all_files[i] for i in idxs]
        log.info("NeuronDataset[%s]: %d images", split, len(self.files))

    def __len__(self):
        return len(self.files)

    def _label_from_name(self, name: str) -> int:
        if "_c" in name:
            try:
                return int(name.rsplit("_c", 1)[1])
            except (ValueError, IndexError):
                pass
        return 0

    def __getitem__(self, idx):
        import torch
        path = self.files[idx]
        img = Image.open(path).convert("L")
        img = img.resize((self.cfg.image_size, self.cfg.image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
        if self.cfg.augment and self.transform is None:
            arr = _augment(arr, self.rng)
        arr = arr[None, ...]  # (1, H, W)
        t = torch.from_numpy(arr)
        if self.transform:
            t = self.transform(t)
        label = self._label_from_name(path.stem)
        return t, label


def build_dataloader(cfg, split="train", batch_size=None):
    import torch
    from torch.utils.data import DataLoader

    ds = NeuronDataset(cfg, split=split)
    return DataLoader(
        ds,
        batch_size=batch_size or cfg.batch_size,
        shuffle=(split == "train"),
        num_workers=cfg.num_workers,
        drop_last=(split == "train"),
    )


# --------------------------------------------------------------------------- #
# Legacy HF download (kept for backward compat with download_data.py)
# --------------------------------------------------------------------------- #
def download_hf_dataset(cfg) -> str:
    from huggingface_hub import snapshot_download
    raw = Path(cfg.raw_dir)
    raw.mkdir(parents=True, exist_ok=True)
    target = raw / cfg.hf_dataset_id.replace("/", "_")
    if target.exists() and any(target.iterdir()):
        return str(target)
    try:
        snapshot_download(repo_id=cfg.hf_dataset_id, repo_type="dataset",
                          local_dir=str(target))
        return str(target)
    except Exception as exc:
        log.warning("HF download failed (%s).", exc)
        return ""


def collect_images_to_processed(cfg, source_dir: str) -> int:
    if not source_dir:
        return 0
    processed = Path(cfg.processed_dir)
    processed.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in _iter_image_files(Path(source_dir)):
        try:
            arr = _load_gray(img_path, cfg.image_size)
            Image.fromarray((arr * 255).astype(np.uint8), "L").save(
                processed / f"hf_{n:05d}_c0.png")
            n += 1
        except Exception:
            pass
    return n