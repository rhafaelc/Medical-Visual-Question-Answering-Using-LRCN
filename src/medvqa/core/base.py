"""Base classes and utilities for dataset handling."""

import json
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Union

from .config import DatasetConfig


class DatasetSummary:
    """Utility for summarizing dataset contents."""

    @staticmethod
    def print_summary(dataset_dir: Path, dataset_name: str) -> None:
        """Print summary of dataset contents."""
        images_dir = dataset_dir / DatasetConfig.IMAGES_DIR_NAME
        annotations_dir = dataset_dir / DatasetConfig.ANNOTATIONS_DIR_NAME

        n_images = sum(1 for _ in images_dir.glob("*")) if images_dir.exists() else 0
        n_annotations = (
            sum(1 for _ in annotations_dir.glob("*")) if annotations_dir.exists() else 0
        )

        print(f"\n[{dataset_name.upper()} Summary]")
        print(f"Images: {n_images} files in {images_dir}")
        print(f"Annotations: {n_annotations} files in {annotations_dir}")


class BaseDatasetLoader(ABC):
    """Base class for dataset loaders."""

    def __init__(self, root: Union[str, Path] = None):
        """Initialize loader with root directory."""
        if root is None:
            root = DatasetConfig.DEFAULT_ROOT / self.dataset_name
        self.root = Path(root)
        self.images_dir = self.root / DatasetConfig.IMAGES_DIR_NAME
        self.annotations_dir = self.root / DatasetConfig.ANNOTATIONS_DIR_NAME

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Dataset name identifier."""
        pass

    @abstractmethod
    def load(self) -> List[Dict]:
        """Load and return standardized dataset entries."""
        pass

    def _normalize_answer_type(self, answer_type: Optional[str], answer: str) -> str:
        """Normalize answer type classification."""
        if answer_type and answer_type.lower() in {"open", "closed"}:
            return answer_type.lower()

        answer_lower = str(answer or "").strip().lower()

        # Check for closed-ended patterns
        if answer_lower in DatasetConfig.ModelConfig.ANSWER_TYPE_CLOSED_KEYWORDS:
            return "closed"
        if answer_lower.replace(".", "", 1).isdigit():
            return "closed"

        return "open"

    def _create_entry_id(self, split: str, index: int) -> str:
        """Create standardized entry ID."""
        return f"{self.dataset_name}_{split}_{index:05d}"

    def _validate_json(self, path: Path) -> bool:
        """Validate JSON file integrity."""
        try:
            json.loads(path.read_text(encoding="utf-8"))
            return True
        except (json.JSONDecodeError, FileNotFoundError):
            return False


class BaseDatasetDownloader(ABC):
    """Base class for dataset downloaders."""

    def __init__(self):
        """Initialize downloader."""
        self.root = DatasetConfig.DEFAULT_ROOT / self.dataset_name
        self.temp_dir = self.root / DatasetConfig.TEMP_DIR_NAME
        self.images_dir = self.root / DatasetConfig.IMAGES_DIR_NAME
        self.annotations_dir = self.root / DatasetConfig.ANNOTATIONS_DIR_NAME

    @property
    @abstractmethod
    def dataset_name(self) -> str:
        """Dataset name identifier."""
        pass

    @abstractmethod
    def download(self) -> int:
        """Download dataset. Returns 0 on success, 1 on failure."""
        pass

    def _ensure_dir(self, path: Path) -> Path:
        """Ensure directory exists."""
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _safe_move(self, src: Path, dst: Path) -> Path:
        """Safely move file, creating destination directory if needed."""
        dst = Path(dst)
        if not dst.exists():
            self._ensure_dir(dst.parent)
            shutil.move(str(src), str(dst))
        return dst

    def _cleanup_temp(self) -> None:
        """Remove temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _print_summary(self) -> None:
        """Print download summary."""
        DatasetSummary.print_summary(self.root, self.dataset_name)
        print(f"[OK] {self.dataset_name} -> {self.images_dir}, {self.annotations_dir}")


def normalize_answer_type(answer_type: Optional[str], answer: str) -> str:
    """Standalone function for answer type normalization."""
    loader = type(
        "TempLoader",
        (BaseDatasetLoader,),
        {"dataset_name": "temp", "load": lambda self: []},
    )()
    return loader._normalize_answer_type(answer_type, answer)
