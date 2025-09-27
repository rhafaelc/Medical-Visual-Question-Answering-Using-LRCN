"""SLAKE dataset loader with production-grade conventions."""

import json
from pathlib import Path

from ..core.base import BaseDatasetLoader
from ..core.config import DatasetConfig


class SlakeLoader(BaseDatasetLoader):
    """Loader for SLAKE dataset (English subset)."""

    @property
    def dataset_name(self) -> str:
        return "slake_all"

    def load(self):
        """Load SLAKE dataset with standardized format."""
        if not self.annotations_dir.exists():
            return []

        entries = []

        for split in ("train", "validation", "test"):
            annotation_file = self.annotations_dir / f"{split}.json"
            if not annotation_file.exists():
                continue

            data = json.loads(annotation_file.read_text(encoding="utf-8"))

            for idx, example in enumerate(data):
                # Filter for English language only
                if (example.get("q_lang") or "").strip().lower() != "en":
                    continue

                # Extract image reference
                image_name = example.get("img_name") or example.get("img_id")
                if not image_name:
                    continue

                # Extract and normalize fields
                question = (example.get("question") or "").strip()
                answer = (example.get("answer") or "").strip().lower()
                answer_type = self._normalize_answer_type(
                    example.get("answer_type"), answer
                )

                entry = {
                    "id": self._create_entry_id(split, idx),
                    "dataset": self.dataset_name,
                    "split": split,
                    "image": str(
                        self.images_dir / "imgs" / image_name
                    ),  # Fixed: Add imgs subfolder
                    "question": question,
                    "answer": answer,
                    "answer_type": answer_type,
                }
                entries.append(entry)

        return entries


def load_slake(root=None):
    """Load SLAKE dataset with backward compatibility."""
    if root is None:
        root = DatasetConfig.DEFAULT_ROOT / DatasetConfig.SLAKE_DIR

    loader = SlakeLoader(root)
    return loader.load()
