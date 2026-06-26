"""Torch device selection helpers for agents."""

import torch


def _mps_is_available() -> bool:
    mps_backend = getattr(torch.backends, "mps", None)
    return bool(mps_backend and mps_backend.is_available())


def get_torch_device(device_config: str = "auto") -> torch.device:
    """Return a torch device from config, preferring Apple MPS in auto mode."""
    device_config = (device_config or "auto").lower()

    if device_config == "auto":
        if _mps_is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if device_config == "mps":
        if not _mps_is_available():
            raise RuntimeError("MPS requested but not available!")
        return torch.device("mps")

    if device_config == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available!")
        return torch.device("cuda")

    if device_config == "cpu":
        return torch.device("cpu")

    raise ValueError("Unsupported device. Use 'auto', 'mps', 'cuda', or 'cpu'.")
