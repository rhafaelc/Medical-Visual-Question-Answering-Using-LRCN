"""Dataset preprocessing script for Medical VQA LRCN."""

import argparse
import json
import sys
from pathlib import Path

from ..datamodules.common import load_slake, load_vqa_rad
from ..preprocessing.image_preprocessing import ImagePreprocessor
from ..preprocessing.text_preprocessing import QuestionPreprocessor, AnswerPreprocessor
from ..core.download_utils import DownloadUtils


def main():
    parser = argparse.ArgumentParser(description="Preprocess Medical VQA datasets")
    parser.add_argument(
        "--output-dir", type=str, default="data/processed", help="Output directory"
    )
    args, _ = parser.parse_known_args()

    try:
        project_root = DownloadUtils.project_root(__file__)
        output_dir = project_root / args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        print("Loading datasets...")
        slake_data = load_slake()
        vqa_rad_data = load_vqa_rad()

        train_data = slake_data["train"] + vqa_rad_data["train"]
        val_data = slake_data["validation"] + vqa_rad_data.get("validation", [])
        test_data = slake_data["test"] + vqa_rad_data["test"]

        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Test samples: {len(test_data)}")

        image_processor = ImagePreprocessor(image_size=224)
        question_processor = QuestionPreprocessor()
        answer_processor = AnswerPreprocessor()

        train_questions = [item["question"] for item in train_data]
        train_answers = [item["answer"] for item in train_data]

        question_processor.build_vocab(train_questions)
        question_processor.compute_max_length(train_questions)
        answer_processor.build_vocab(train_answers)

        vocab_stats = {
            "question_vocab_size": question_processor.get_vocab_size(),
            "question_max_length": question_processor.max_length,
            "answer_vocab_size": answer_processor.get_vocab_size(),
            "num_classes": answer_processor.get_vocab_size(),
            "total_train_samples": len(train_data),
            "total_val_samples": len(val_data),
            "total_test_samples": len(test_data),
        }

        print(f"Question vocab size: {vocab_stats['question_vocab_size']}")
        print(f"Question max length: {vocab_stats['question_max_length']}")
        print(f"Answer vocab size: {vocab_stats['answer_vocab_size']}")

        with open(output_dir / "question_vocab.json", "w") as f:
            json.dump(question_processor.get_vocab_dict(), f, indent=2)

        with open(output_dir / "answer_vocab.json", "w") as f:
            json.dump(answer_processor.get_vocab_dict(), f, indent=2)

        with open(output_dir / "preprocessing_stats.json", "w") as f:
            json.dump(vocab_stats, f, indent=2)

        splits = {
            "train": len(train_data),
            "validation": len(val_data),
            "test": len(test_data),
        }

        with open(output_dir / "dataset_splits.json", "w") as f:
            json.dump(splits, f, indent=2)

        print(f"Preprocessing completed. Files saved to: {output_dir}")
        return 0

    except Exception as e:
        print(f"Preprocessing failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
