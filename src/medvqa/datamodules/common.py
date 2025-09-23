from .slake import load_slake
from .vqa_rad import load_vqa_rad

__all__ = ["load_slake", "load_vqa_rad"]


def load_all():
    return load_slake() + load_vqa_rad()
