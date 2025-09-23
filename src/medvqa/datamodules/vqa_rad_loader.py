"""VQA-RAD dataset loader with production-grade conventions."""

import json
from pathlib import Path
from typing import Dict, List

from ..core.base import BaseDatasetLoader
from ..core.config import DatasetConfig


class VqaRadLoader(BaseDatasetLoader):
    """Loader for VQA-RAD dataset."""

    @property
    def dataset_name(self) -> str:
        return "vqa-rad"

    def load(self) -> List[Dict]:
        """Load VQA-RAD dataset with standardized format."""
        annotation_file = self.annotations_dir / DatasetConfig.VQA_RAD_PUBLIC_JSON
        if not annotation_file.exists():
            return []

        data = json.loads(annotation_file.read_text(encoding="utf-8"))
        entries = []
        counters = {"train": 0, "test": 0}

        for example in data:
            # Extract image name
            image_name = example.get("image_name")
            if not image_name:
                continue

            # Extract and normalize fields
            question = (example.get("q_lang") or example.get("question") or "").strip()
            answer = str(example.get("answer") or "").strip().lower()
            answer_type = self._normalize_answer_type(
                example.get("answer_type"), answer
            )

            # Determine split based on phrase_type
            phrase_type = (example.get("phrase_type") or "").strip().lower()
            split = "test" if phrase_type.startswith("test_") else "train"

            idx = counters[split]
            entry = {
                "id": self._create_entry_id(split, idx),
                "dataset": self.dataset_name,
                "split": split,
                "image": str(self.images_dir / image_name),
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
            }
            entries.append(entry)
            counters[split] += 1

        return entries


def load_vqa_rad(root: str = None) -> List[Dict]:
    """Load VQA-RAD dataset with backward compatibility."""
    if root is None:
        root = DatasetConfig.DEFAULT_ROOT / DatasetConfig.VQA_RAD_DIR

    loader = VqaRadLoader(root)
    return loader.load()
