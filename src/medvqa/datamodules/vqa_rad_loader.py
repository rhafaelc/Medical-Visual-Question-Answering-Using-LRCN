"""VQA-RAD dataset loader with stratified train/validation/test splits."""

import json
import random
from collections import defaultdict
from pathlib import Path

from ..core.base import BaseDatasetLoader
from ..core.config import DatasetConfig, ModelConfig


class VqaRadLoader(BaseDatasetLoader):
    """Loader for VQA-RAD dataset with stratified splits."""

    @property
    def dataset_name(self) -> str:
        return "vqa-rad"

    def load(self):
        """Load VQA-RAD dataset with stratified train/validation splits from original training data."""
        annotation_file = self.annotations_dir / DatasetConfig.VQA_RAD_PUBLIC_JSON
        if not annotation_file.exists():
            return []

        data = json.loads(annotation_file.read_text(encoding="utf-8"))

        train_pool = []
        test_entries = []
        train_answer_type_groups = defaultdict(list)

        for example in data:
            image_name = example.get("image_name")
            if not image_name:
                continue

            question = (example.get("q_lang") or example.get("question") or "").strip()
            answer = str(example.get("answer") or "").strip().lower()
            answer_type = self._normalize_answer_type(
                example.get("answer_type"), answer
            )
            phrase_type = (example.get("phrase_type") or "").strip().lower()

            entry = {
                "dataset": self.dataset_name,
                "image": str(self.images_dir / image_name),
                "question": question,
                "answer": answer,
                "answer_type": answer_type,
            }

            if phrase_type.startswith("test_"):
                entry["split"] = "test"
                test_entries.append(entry)
            else:
                train_pool.append(entry)
                train_answer_type_groups[answer_type].append(len(train_pool) - 1)

        train_indices = []
        val_indices = []

        random.seed(ModelConfig.RANDOM_SEED)
        val_ratio = ModelConfig.VQA_RAD_VAL_SPLIT_RATIO

        for answer_type, indices in train_answer_type_groups.items():
            shuffled_indices = indices.copy()
            random.shuffle(shuffled_indices)

            n_total = len(shuffled_indices)
            n_val = int(n_total * val_ratio)

            if n_total >= 2:
                n_val = max(1, n_val) if n_val > 0 else 0
                n_train = n_total - n_val
            else:
                n_train = n_total
                n_val = 0

            train_indices.extend(shuffled_indices[:n_train])
            val_indices.extend(shuffled_indices[n_train:])

        final_entries = []
        counters = {"train": 0, "validation": 0, "test": 0}

        for idx in train_indices:
            entry = train_pool[idx].copy()
            entry["split"] = "train"
            entry["id"] = self._create_entry_id("train", counters["train"])
            final_entries.append(entry)
            counters["train"] += 1

        for idx in val_indices:
            entry = train_pool[idx].copy()
            entry["split"] = "validation"
            entry["id"] = self._create_entry_id("validation", counters["validation"])
            final_entries.append(entry)
            counters["validation"] += 1

        for entry in test_entries:
            entry["id"] = self._create_entry_id("test", counters["test"])
            final_entries.append(entry)
            counters["test"] += 1

        return final_entries


def load_vqa_rad(root=None):
    """Load VQA-RAD dataset with backward compatibility."""
    if root is None:
        root = DatasetConfig.DEFAULT_ROOT / DatasetConfig.VQA_RAD_DIR

    loader = VqaRadLoader(root)
    return loader.load()
