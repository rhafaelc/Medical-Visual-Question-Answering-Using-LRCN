"""PyTorch Dataset for Medical VQA with on-the-fly preprocessing.

Uses preprocessing artifacts built offline and applies sample transformations during training.
"""

import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
from ..core.config import ModelConfig

from ..preprocessing.image_preprocessing import ImagePreprocessor
from ..preprocessing.text_preprocessing import QuestionPreprocessor, AnswerPreprocessor
from ..core.config import ModelConfig


class MedVQADataset(Dataset):
    """Medical VQA Dataset with on-the-fly preprocessing.

    Loads preprocessed vocabularies and applies transformations during __getitem__.
    """

    def __init__(
        self,
        data_entries,
        preprocessing_dir,
        split,
        image_transforms=None,
    ):
        """Initialize dataset with preprocessing artifacts.

        Args:
            data_entries: Raw dataset entries with image paths, questions, answers
            preprocessing_dir: Directory containing preprocessing artifacts
            split: Dataset split ("train", "validation", "test")
            image_transforms: Custom image transforms (overrides config)
        """
        self.data_entries = data_entries
        self.preprocessing_dir = Path(preprocessing_dir)
        self.split = split

        # Load preprocessing artifacts
        self._load_vocabularies()
        self._load_preprocessing_config()

        # Initialize processors with loaded vocabularies
        self._setup_processors(image_transforms)

    def _load_vocabularies(self) -> None:
        """Load question and answer vocabularies from preprocessing artifacts."""
        # Load question vocabulary
        question_vocab_file = self.preprocessing_dir / "question_vocab.json"
        with open(question_vocab_file, "r", encoding="utf-8") as f:
            self.question_vocab = json.load(f)

        # Load answer vocabulary
        answer_vocab_file = self.preprocessing_dir / "answer_vocab.json"
        with open(answer_vocab_file, "r", encoding="utf-8") as f:
            self.answer_vocab = json.load(f)

        # Load inverse vocabularies
        question_idx_file = self.preprocessing_dir / "question_idx_to_word.json"
        with open(question_idx_file, "r", encoding="utf-8") as f:
            self.question_idx_to_word = {int(k): v for k, v in json.load(f).items()}

        answer_idx_file = self.preprocessing_dir / "answer_idx_to_answer.json"
        with open(answer_idx_file, "r", encoding="utf-8") as f:
            self.answer_idx_to_answer = {int(k): v for k, v in json.load(f).items()}

    def _load_preprocessing_config(self) -> None:
        """Load preprocessing configuration and statistics."""
        # Load preprocessing statistics
        stats_file = self.preprocessing_dir / "preprocessing_stats.json"
        with open(stats_file, "r", encoding="utf-8") as f:
            self.preprocessing_stats = json.load(f)

        # Load image processor configuration
        image_config_file = self.preprocessing_dir / "image_processor_config.json"
        with open(image_config_file, "r", encoding="utf-8") as f:
            self.image_config = json.load(f)

        # Extract key parameters
        self.question_max_length = self.preprocessing_stats["question_max_length"]
        self.num_classes = len(self.answer_vocab)

    def _setup_processors(self, image_transforms=None):
        """Setup processors with loaded configurations."""
        # Setup image processor
        if image_transforms is None:
            self.image_processor = ImagePreprocessor(
                image_size=self.image_config["image_size"],
                apply_contrast_stretching=self.image_config[
                    "apply_contrast_stretching"
                ],
                normalize_range=tuple(self.image_config["normalize_range"]),
            )
        else:
            self.image_processor = image_transforms

        # Setup question processor with loaded vocabulary
        self.question_processor = QuestionPreprocessor(
            max_length=self.question_max_length
        )
        self.question_processor.word_to_idx = self.question_vocab
        self.question_processor.idx_to_word = self.question_idx_to_word
        self.question_processor.vocab = list(self.question_vocab.keys())

        # Setup answer processor with loaded vocabulary
        self.answer_processor = AnswerPreprocessor()
        self.answer_processor.answer_to_idx = self.answer_vocab
        self.answer_processor.idx_to_answer = self.answer_idx_to_answer
        self.answer_processor.answer_vocab = list(self.answer_vocab.keys())

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.data_entries)

    def __getitem__(self, idx):
        """Get preprocessed sample for training.

        Args:
            idx: Sample index

        Returns:
            Dictionary containing preprocessed tensors:
            - image: [3, H, W] tensor
            - question: [L_max] token indices tensor
            - answer: class index tensor
            - metadata: additional information
        """
        entry = self.data_entries[idx]

        # Process image
        try:
            image = self.image_processor.preprocess(entry["image"])
        except Exception as e:
            print(f"Warning: Failed to load image {entry['image']}: {e}")
            image = torch.randn(
                3, self.image_config["image_size"], self.image_config["image_size"]
            ).clamp(0, 1)

        # Process question
        question_indices = self.question_processor.encode(entry["question"])
        question_tensor = torch.tensor(
            self.question_processor.pad_or_truncate(question_indices), dtype=torch.long
        )

        # Process answer
        answer_idx = self.answer_processor.encode(entry["answer"])
        answer_tensor = torch.tensor(answer_idx, dtype=torch.long)

        return {
            "image": image,
            "question": question_tensor,
            "answer": answer_tensor,
            "answer_type": entry["answer_type"],
            "dataset": entry["dataset"],
            "id": entry["id"],
            "raw_question": entry["question"],
            "raw_answer": entry["answer"],
        }

    def decode_question(self, question_tensor: torch.Tensor) -> str:
        """Decode question tensor back to text.

        Args:
            question_tensor: Tensor of token indices

        Returns:
            Decoded question string
        """
        return self.question_processor.decode(question_tensor.tolist())

    def decode_answer(self, answer_idx: int) -> str:
        """Decode answer index back to text.

        Args:
            answer_idx: Answer class index

        Returns:
            Decoded answer string
        """
        return self.answer_processor.decode(answer_idx)

    def get_class_weights(self) -> torch.Tensor:
        """Load class weights for handling imbalanced data.

        Returns:
            Class weights tensor
        """
        weights_file = self.preprocessing_dir / "answer_class_weights.json"
        if weights_file.exists():
            with open(weights_file, "r", encoding="utf-8") as f:
                weights_dict = json.load(f)
                weights = [weights_dict[str(i)] for i in range(len(weights_dict))]
                return torch.tensor(weights, dtype=torch.float32)
        else:
            # Uniform weights if not available
            return torch.ones(self.num_classes, dtype=torch.float32)

    def get_dataset_stats(self):
        """Get dataset statistics.

        Returns:
            Dictionary with dataset statistics
        """
        return {
            "num_samples": len(self.data_entries),
            "question_vocab_size": len(self.question_vocab),
            "answer_vocab_size": len(self.answer_vocab),
            "question_max_length": self.question_max_length,
            "num_classes": self.num_classes,
            "image_size": self.image_config["image_size"],
        }


def create_dataloaders(
    preprocessing_dir: Path,
    batch_size: int = ModelConfig.DEFAULT_BATCH_SIZE,
    num_workers: int = ModelConfig.DEFAULT_NUM_WORKERS,
    pin_memory: bool = True,
):
    """Create Medical VQA DataLoaders.

    Args:
        preprocessing_dir: Directory with preprocessing artifacts
        batch_size: Batch size for training
        num_workers: Number of DataLoader workers
        pin_memory: Enable memory pinning

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    from ..datamodules.slake_loader import load_slake
    from ..datamodules.vqa_rad_loader import load_vqa_rad
    from torch.utils.data import DataLoader

    # Load raw datasets
    slake_data = load_slake()
    vqa_rad_data = load_vqa_rad()

    # Split data by splits
    train_data = []
    val_data = []
    test_data = []

    for entry in slake_data + vqa_rad_data:
        if entry["split"] == "train":
            train_data.append(entry)
        elif entry["split"] == "validation":
            val_data.append(entry)
        elif entry["split"] == "test":
            test_data.append(entry)

    # Create datasets
    train_dataset = MedVQADataset(train_data, preprocessing_dir, "train")
    val_dataset = MedVQADataset(val_data, preprocessing_dir, "validation")
    test_dataset = MedVQADataset(test_data, preprocessing_dir, "test")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader
