"""Device utilities for GPU/CPU fallback."""

import torch


def get_device() -> torch.device:
    """Get the best available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device("cpu")
        print("[DEVICE] Using CPU (CUDA not available)")

    return device


def move_to_device(obj, device: torch.device):
    """Move tensor, model, or collection to device."""
    if hasattr(obj, "to"):
        return obj.to(device)
    elif isinstance(obj, (list, tuple)):
        return type(obj)(move_to_device(item, device) for item in obj)
    elif isinstance(obj, dict):
        return {key: move_to_device(value, device) for key, value in obj.items()}
    else:
        return obj


class DeviceMixin:
    """Mixin class for automatic device management."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = get_device()
        if hasattr(self, "to"):
            self.to(self.device)

    def forward(self, *args, **kwargs):
        """Ensure inputs are on the correct device."""
        args = tuple(move_to_device(arg, self.device) for arg in args)
        kwargs = {k: move_to_device(v, self.device) for k, v in kwargs.items()}
        return super().forward(*args, **kwargs)
