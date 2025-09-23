"""Common dataset loading interface for Medical VQA datasets."""

from .slake_loader import load_slake
from .vqa_rad_loader import load_vqa_rad

__all__ = ["load_slake", "load_vqa_rad", "load_all"]


def load_all():
    """Load all available datasets."""
    return load_slake() + load_vqa_rad()
