"""Device selection: DirectML (AMD/Intel GPU) -> CUDA -> CPU.

Windows + AMD Radeon iGPU has no CUDA, so DirectML is the fastest path.
If torch-directml isn't installed we gracefully fall back to CPU.
"""

from __future__ import annotations

import logging

log = logging.getLogger("mind-diffusion.device")


def get_device(preference: str = "auto") -> str:
    """Return a torch device string.

    preference: "auto" | "directml" | "cuda" | "cpu"
    """
    import torch

    if preference in ("cpu",):
        log.info("Using CPU (forced)")
        return "cpu"

    if preference in ("cuda", "auto"):
        if torch.cuda.is_available():
            log.info("Using CUDA: %s", torch.cuda.get_device_name(0))
            return "cuda"

    if preference in ("directml", "auto"):
        try:
            import torch_directml  # type: ignore

            if torch_directml.is_available():
                log.info("Using DirectML device: %s",
                         torch_directml.device_name(0))
                return torch_directml.device()  # returns a device object/string
        except ImportError:
            log.info("torch-directml not installed")

    if preference == "directml":
        raise RuntimeError("Requested DirectML but it is unavailable. "
                           "Install with: pip install torch-directml")

    log.warning("No GPU accelerator found - using CPU. "
                "Training will be slow; consider installing torch-directml.")
    return "cpu"


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)