"""Dataset preprocessing for Medical VQA LRCN.

Builds vocabularies and statistics from training data.
Individual sample transformations handled during training.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np
from collections import Counter

from ..datamodules.slake_loader import load_slake
from ..datamodules.vqa_rad_loader import load_vqa_rad
from ..preprocessing.image_preprocessing import ImagePreprocessor
from ..preprocessing.text_preprocessing import QuestionPreprocessor, AnswerPreprocessor
from ..core.config import DatasetConfig, ModelConfig
from ..core.download_utils import DownloadUtils


class DatasetPreprocessor:
    """Medical VQA dataset preprocessor.

    Builds vocabularies and validates data integrity.
    """

    def __init__(
        self,
        output_dir: Path,
        image_size: int = ModelConfig.IMAGE_SIZE,
        coverage_percentile: int = ModelConfig.COVERAGE_PERCENTILE,
    ):
        """Initialize preprocessor.

        Args:
            output_dir: Directory to save preprocessing artifacts
            image_size: Target image resolution
            coverage_percentile: Coverage for L_max and top-K selection
        """
        self.output_dir = output_dir
        self.coverage_percentile = coverage_percentile

        # Initialize processors
        self.question_processor = QuestionPreprocessor(
            coverage_percentile=coverage_percentile
        )
        self.answer_processor = AnswerPreprocessor(
            coverage_percentile=coverage_percentile
        )

        # Image processor configuration
        self.image_config = {
            "image_size": image_size,
            "apply_contrast_stretching": True,
            "normalize_range": ModelConfig.NORMALIZE_RANGE,
        }

        self.preprocessing_stats = {}

    def load_and_split_datasets(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Load datasets and create train/val/test splits.

        Returns:
            Tuple of (train_data, val_data, test_data)
        """
        print("Loading VQA-RAD and SLAKE datasets...")

        # Load individual datasets
        slake_data = load_slake()
        vqa_rad_data = load_vqa_rad()

        # Extract data by splits
        train_data = []
        val_data = []
        test_data = []

        # Process each dataset entry
        for entry in slake_data + vqa_rad_data:
            if entry["split"] == "train":
                train_data.append(entry)
            elif entry["split"] == "validation":
                val_data.append(entry)
            elif entry["split"] == "test":
                test_data.append(entry)

        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")

        # Store split statistics
        self.preprocessing_stats.update(
            {
                "total_train_samples": len(train_data),
                "total_validation_samples": len(val_data),
                "total_test_samples": len(test_data),
                "total_samples": len(train_data) + len(val_data) + len(test_data),
            }
        )

        return train_data, val_data, test_data

    def analyze_question_statistics(self, questions: List[str]) -> Dict[str, Any]:
        """Analyze question length distribution and tokenization patterns.

        Args:
            questions: List of question strings

        Returns:
            Dictionary containing question statistics
        """
        token_lengths = []
        char_lengths = []

        for question in questions:
            tokens = self.question_processor.tokenize(question)
            token_lengths.append(len(tokens))
            char_lengths.append(len(question))

        stats = {
            "token_length_stats": {
                "min": int(np.min(token_lengths)),
                "max": int(np.max(token_lengths)),
                "mean": float(np.mean(token_lengths)),
                "std": float(np.std(token_lengths)),
                "median": float(np.median(token_lengths)),
                "percentile_95": int(np.percentile(token_lengths, 95)),
                "percentile_99": int(np.percentile(token_lengths, 99)),
            },
            "character_length_stats": {
                "min": int(np.min(char_lengths)),
                "max": int(np.max(char_lengths)),
                "mean": float(np.mean(char_lengths)),
                "std": float(np.std(char_lengths)),
            },
            "total_questions": len(questions),
        }

        return stats

    def analyze_answer_statistics(self, answers: List[str]) -> Dict[str, Any]:
        """Analyze answer distribution and vocabulary patterns.

        Args:
            answers: List of answer strings

        Returns:
            Dictionary containing answer statistics
        """
        # Normalize answers for analysis
        normalized_answers = [ans.lower().strip() for ans in answers]
        answer_counts = Counter(normalized_answers)

        # Classify answer types using existing logic
        closed_count = 0
        open_count = 0

        for answer in normalized_answers:
            if (
                answer in ModelConfig.ANSWER_TYPE_CLOSED_KEYWORDS
                or answer.replace(".", "", 1).isdigit()
            ):
                closed_count += 1
            else:
                open_count += 1

        stats = {
            "total_answers": len(answers),
            "unique_answers": len(answer_counts),
            "answer_type_distribution": {
                "closed_ended": closed_count,
                "open_ended": open_count,
                "closed_percentage": float(closed_count / len(answers) * 100),
                "open_percentage": float(open_count / len(answers) * 100),
            },
            "most_frequent_answers": answer_counts.most_common(20),
            "answer_frequency_stats": {
                "singleton_answers": sum(
                    1 for count in answer_counts.values() if count == 1
                ),
                "max_frequency": max(answer_counts.values()),
                "min_frequency": min(answer_counts.values()),
            },
        }

        return stats

    def build_vocabularies(self, train_data: List[Dict]) -> Dict[str, Any]:
        """Build question and answer vocabularies from training data.

        Args:
            train_data: Training dataset entries

        Returns:
            Dictionary containing vocabulary information
        """
        print("Building vocabularies from training data...")

        # Extract training questions and answers
        train_questions = [item["question"] for item in train_data]
        train_answers = [item["answer"] for item in train_data]

        # Build question vocabulary and compute L_max
        print("Processing questions...")
        question_vocab = self.question_processor.build_vocab(train_questions)
        l_max = self.question_processor.compute_max_length(train_questions)

        # Build answer vocabulary with top-K selection
        print("Processing answers...")
        answer_vocab = self.answer_processor.build_vocab(train_answers)

        # Analyze statistics
        question_stats = self.analyze_question_statistics(train_questions)
        answer_stats = self.analyze_answer_statistics(train_answers)

        vocab_info = {
            "question_vocab_size": len(question_vocab),
            "question_max_length": l_max,
            "answer_vocab_size": len(answer_vocab),
            "num_classes": len(answer_vocab),
            "coverage_percentile": self.coverage_percentile,
            "question_statistics": question_stats,
            "answer_statistics": answer_stats,
        }

        # Store in preprocessing stats
        self.preprocessing_stats.update(vocab_info)

        return vocab_info

    def validate_image_paths(
        self, data_splits: Tuple[List[Dict], List[Dict], List[Dict]]
    ) -> Dict[str, Any]:
        """Validate that all image paths exist and are readable.

        Args:
            data_splits: Tuple of (train, val, test) data

        Returns:
            Dictionary containing validation results
        """
        print("Validating image paths...")

        validation_results = {
            "total_images": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "missing_files": [],
        }

        for split_name, split_data in zip(["train", "validation", "test"], data_splits):
            split_valid = 0
            split_invalid = 0

            for entry in split_data:
                image_path = Path(entry["image"])
                validation_results["total_images"] += 1

                if image_path.exists() and image_path.is_file():
                    split_valid += 1
                    validation_results["valid_images"] += 1
                else:
                    split_invalid += 1
                    validation_results["invalid_images"] += 1
                    validation_results["missing_files"].append(str(image_path))

            print(f"{split_name}: {split_valid} valid, {split_invalid} invalid images")
            validation_results[f"{split_name}_valid"] = split_valid
            validation_results[f"{split_name}_invalid"] = split_invalid

        # Store validation results
        self.preprocessing_stats["image_validation"] = validation_results

        return validation_results

    def save_preprocessed_data(self, vocab_info: Dict[str, Any]) -> None:
        """Save all preprocessed vocabularies and statistics to output directory.

        Args:
            vocab_info: Vocabulary information dictionary
        """
        print(f"Saving preprocessed data to {self.output_dir}...")

        # Save question vocabulary
        question_vocab_file = self.output_dir / "question_vocab.json"
        with open(question_vocab_file, "w", encoding="utf-8") as f:
            json.dump(
                self.question_processor.word_to_idx, f, indent=2, ensure_ascii=False
            )

        # Save answer vocabulary
        answer_vocab_file = self.output_dir / "answer_vocab.json"
        with open(answer_vocab_file, "w", encoding="utf-8") as f:
            json.dump(
                self.answer_processor.answer_to_idx, f, indent=2, ensure_ascii=False
            )

        # Save inverse vocabularies for decoding
        question_idx_to_word_file = self.output_dir / "question_idx_to_word.json"
        with open(question_idx_to_word_file, "w", encoding="utf-8") as f:
            idx_to_word = {
                str(idx): word
                for word, idx in self.question_processor.word_to_idx.items()
            }
            json.dump(idx_to_word, f, indent=2, ensure_ascii=False)

        answer_idx_to_answer_file = self.output_dir / "answer_idx_to_answer.json"
        with open(answer_idx_to_answer_file, "w", encoding="utf-8") as f:
            idx_to_answer = {
                str(idx): answer
                for answer, idx in self.answer_processor.answer_to_idx.items()
            }
            json.dump(idx_to_answer, f, indent=2, ensure_ascii=False)

        # Save comprehensive preprocessing statistics
        stats_file = self.output_dir / "preprocessing_stats.json"
        with open(stats_file, "w", encoding="utf-8") as f:
            json.dump(self.preprocessing_stats, f, indent=2, ensure_ascii=False)

        # Save dataset splits information
        splits_file = self.output_dir / "dataset_splits.json"
        splits_info = {
            "train_samples": self.preprocessing_stats["total_train_samples"],
            "validation_samples": self.preprocessing_stats["total_validation_samples"],
            "test_samples": self.preprocessing_stats["total_test_samples"],
            "total_samples": self.preprocessing_stats["total_samples"],
            "coverage_percentile": self.coverage_percentile,
            "question_max_length": self.preprocessing_stats["question_max_length"],
            "answer_vocab_size": self.preprocessing_stats["answer_vocab_size"],
        }
        with open(splits_file, "w", encoding="utf-8") as f:
            json.dump(splits_info, f, indent=2, ensure_ascii=False)

        # Save class weights for handling imbalanced data
        if hasattr(self.answer_processor, "get_class_weights"):
            class_weights = self.answer_processor.get_class_weights()
            weights_file = self.output_dir / "answer_class_weights.json"
            with open(weights_file, "w", encoding="utf-8") as f:
                weights_dict = {
                    str(i): float(w) for i, w in enumerate(class_weights.tolist())
                }
                json.dump(weights_dict, f, indent=2)

        # Save image processor configuration for DataLoader
        image_config_file = self.output_dir / "image_processor_config.json"
        with open(image_config_file, "w", encoding="utf-8") as f:
            json.dump(self.image_config, f, indent=2)

        print(f"Preprocessing completed:")
        print(f"  - Question vocabulary: {vocab_info['question_vocab_size']} tokens")
        print(
            f"  - Question max length (L_max): {vocab_info['question_max_length']} tokens"
        )
        print(f"  - Answer vocabulary: {vocab_info['answer_vocab_size']} classes")
        print(f"  - Coverage percentile: {self.coverage_percentile}%")
        print(
            f"  - Total samples processed: {self.preprocessing_stats['total_samples']}"
        )


def main() -> int:
    """Main preprocessing pipeline."""
    parser = argparse.ArgumentParser(
        description="Preprocess Medical VQA datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=ModelConfig.IMAGE_SIZE,
        help="Target image size for ViT preprocessing",
    )
    parser.add_argument(
        "--coverage-percentile",
        type=int,
        default=ModelConfig.COVERAGE_PERCENTILE,
        help="Coverage percentile for L_max and top-K selection",
    )
    parser.add_argument(
        "--disable-contrast-stretching",
        action="store_true",
        help="Disable contrast stretching for medical images",
    )
    parser.add_argument(
        "--validate-images",
        action="store_true",
        help="Validate all image paths exist and are readable",
    )

    args = parser.parse_args()

    try:
        # Setup output directory
        project_root = DownloadUtils.project_root(__file__)
        output_dir = project_root / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Medical VQA LRCN - Dataset Preprocessing")
        print(f"Output directory: {output_dir}")
        print(f"Image size: {args.image_size}x{args.image_size}")
        print(f"Coverage percentile: {args.coverage_percentile}%")

        # Initialize preprocessor
        preprocessor = DatasetPreprocessor(
            output_dir=output_dir,
            image_size=args.image_size,
            coverage_percentile=args.coverage_percentile,
        )

        # Load and split datasets
        train_data, val_data, test_data = preprocessor.load_and_split_datasets()

        # Build vocabularies from training data
        vocab_info = preprocessor.build_vocabularies(train_data)

        # Optional image validation
        if args.validate_images:
            validation_results = preprocessor.validate_image_paths(
                (train_data, val_data, test_data)
            )
            if validation_results["invalid_images"] > 0:
                print(
                    f"Warning: {validation_results['invalid_images']} invalid image paths found"
                )

        # Save all preprocessed data
        preprocessor.save_preprocessed_data(vocab_info)

        return 0

    except Exception as e:
        print(f"Preprocessing failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
