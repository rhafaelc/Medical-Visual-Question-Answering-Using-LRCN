"""Model evaluation script with attention visualization and results analysis."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image

from ..models.lrcn import LRCN
from ..datamodules.common import load_slake, load_vqa_rad
from ..preprocessing.image_preprocessing import ImagePreprocessor
from ..preprocessing.text_preprocessing import QuestionPreprocessor, AnswerPreprocessor
from ..core.config import ModelConfig


class ModelEvaluator:
    """Comprehensive model evaluation with attention visualization."""

    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize evaluator with trained model."""
        self.model_path = Path(model_path)
        self.device = (
            device
            if device != "auto"
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = None
        self.model_config = None
        self.results = {}

    def load_model(self):
        """Load trained model from checkpoint."""
        print(f"Loading model from: {self.model_path}")

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model_config = checkpoint.get("config", {})

        # Create model instance
        self.model = LRCN(
            num_classes=self.model_config.get("num_classes", 1000),
            hidden_dim=self.model_config.get("hidden_dim", 512),
            num_attention_layers=self.model_config.get("num_attention_layers", 8),
            use_lrm=self.model_config.get("use_lrm", True),
            visual_encoder_type=self.model_config.get("visual_encoder_type", "vit"),
            text_encoder_type=self.model_config.get("text_encoder_type", "biobert"),
        )

        # Load state dict
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded successfully!")
        print(f"Configuration: {self.model_config}")
        print(f"Best validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")

    def evaluate_model(
        self, dataloader: DataLoader, split_name: str = "test"
    ) -> Dict[str, float]:
        """Evaluate model on given dataset."""
        print(f"\nEvaluating on {split_name} set...")

        total_loss = 0.0
        total_acc = 0.0
        total_closed_acc = 0.0
        total_open_acc = 0.0
        num_batches = 0

        criterion = nn.CrossEntropyLoss()

        all_predictions = []
        all_targets = []
        all_question_types = []

        with torch.no_grad():
            progress_bar = tqdm(
                dataloader, desc=f"{split_name.capitalize()} Evaluation"
            )

            for batch in progress_bar:
                images = batch["images"].to(self.device)
                questions = batch["questions"]
                answers = batch["answers"].to(self.device)
                answer_types = batch["answer_types"]

                outputs = self.model(images, questions)
                if isinstance(outputs, dict):
                    outputs = outputs.get(
                        "logits", outputs.get("predictions", list(outputs.values())[0])
                    )

                loss = criterion(outputs, answers)

                # Compute accuracies
                predictions = torch.argmax(outputs, dim=1)
                correct = (predictions == answers).float()

                overall_acc = correct.mean().item()
                closed_mask = torch.tensor([t == "closed" for t in answer_types])
                open_mask = torch.tensor([t == "open" for t in answer_types])

                closed_acc = (
                    correct[closed_mask].mean().item() if closed_mask.sum() > 0 else 0.0
                )
                open_acc = (
                    correct[open_mask].mean().item() if open_mask.sum() > 0 else 0.0
                )

                total_loss += loss.item()
                total_acc += overall_acc
                total_closed_acc += closed_acc
                total_open_acc += open_acc
                num_batches += 1

                # Store for detailed analysis
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(answers.cpu().numpy())
                all_question_types.extend(answer_types)

                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{overall_acc:.4f}",
                        "Closed": f"{closed_acc:.4f}",
                        "Open": f"{open_acc:.4f}",
                    }
                )

        metrics = {
            "loss": total_loss / num_batches,
            "accuracy": total_acc / num_batches,
            "closed_accuracy": total_closed_acc / num_batches,
            "open_accuracy": total_open_acc / num_batches,
        }

        self.results[split_name] = {
            "metrics": metrics,
            "predictions": all_predictions,
            "targets": all_targets,
            "question_types": all_question_types,
        }

        return metrics

    def generate_results_table(self, output_dir: Path):
        """Generate comprehensive results table."""
        results_data = []

        for split_name, split_results in self.results.items():
            metrics = split_results["metrics"]
            results_data.append(
                {
                    "Split": split_name.capitalize(),
                    "Loss": f"{metrics['loss']:.4f}",
                    "Overall Accuracy": f"{metrics['accuracy']:.4f}",
                    "Closed Questions": f"{metrics['closed_accuracy']:.4f}",
                    "Open Questions": f"{metrics['open_accuracy']:.4f}",
                    "Total Samples": len(split_results["predictions"]),
                }
            )

        # Create results table
        df = pd.DataFrame(results_data)

        # Save to CSV
        csv_path = output_dir / "evaluation_results.csv"
        df.to_csv(csv_path, index=False)

        # Create formatted table visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axis("tight")
        ax.axis("off")

        table = ax.table(
            cellText=df.values, colLabels=df.columns, cellLoc="center", loc="center"
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)

        # Style the table
        for i in range(len(df.columns)):
            table[(0, i)].set_facecolor("#40466e")
            table[(0, i)].set_text_props(weight="bold", color="white")

        plt.title(
            f"Model Evaluation Results\n{self.model_config}",
            fontsize=14,
            fontweight="bold",
            pad=20,
        )
        plt.savefig(output_dir / "results_table.png", dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Results table saved to: {csv_path}")
        return df

    def visualize_attention_maps(
        self, dataloader: DataLoader, output_dir: Path, num_samples: int = 10
    ):
        """Generate attention map visualizations."""
        print(f"\nGenerating attention visualizations...")

        attention_dir = output_dir / "attention_maps"
        attention_dir.mkdir(exist_ok=True)

        self.model.eval()
        samples_processed = 0

        with torch.no_grad():
            for batch in dataloader:
                if samples_processed >= num_samples:
                    break

                images = batch["images"].to(self.device)
                questions = batch["questions"]

                # Forward pass
                outputs = self.model(images, questions)

                # Try to extract attention weights
                attention_weights = None
                if hasattr(self.model, "lrcn_attention"):
                    # Check if attention module stored weights
                    if hasattr(self.model.lrcn_attention, "last_attention_weights"):
                        attention_weights = (
                            self.model.lrcn_attention.last_attention_weights
                        )

                batch_size = len(images)
                for i in range(min(batch_size, num_samples - samples_processed)):
                    self._save_attention_sample(
                        images[i],
                        questions[i],
                        attention_weights[i] if attention_weights else None,
                        attention_dir,
                        samples_processed + i + 1,
                    )

                samples_processed += batch_size

        print(f"Attention maps saved to: {attention_dir}")

    def _save_attention_sample(
        self, image_tensor, question, attention_weights, save_dir, sample_id
    ):
        """Save individual attention visualization."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original image
        img_np = image_tensor.cpu().permute(1, 2, 0).numpy()
        # Denormalize if needed
        if img_np.min() < 0:  # Likely normalized
            mean = np.array(ModelConfig.IMAGENET_MEAN)
            std = np.array(ModelConfig.IMAGENET_STD)
            img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)

        axes[0].imshow(img_np)
        axes[0].set_title("Original Medical Image", fontsize=12, fontweight="bold")
        axes[0].axis("off")

        # Attention map
        if attention_weights is not None:
            try:
                # Average across attention heads if multi-head
                if len(attention_weights.shape) > 2:
                    attn_map = attention_weights.cpu().numpy().mean(axis=0)
                else:
                    attn_map = attention_weights.cpu().numpy()

                im = axes[1].imshow(attn_map, cmap="hot", interpolation="bilinear")
                axes[1].set_title("Attention Heatmap", fontsize=12, fontweight="bold")
                axes[1].axis("off")
                plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
            except Exception as e:
                axes[1].text(
                    0.5,
                    0.5,
                    f"Attention visualization\nerror: {str(e)[:50]}...",
                    ha="center",
                    va="center",
                    transform=axes[1].transAxes,
                )
                axes[1].axis("off")
        else:
            axes[1].text(
                0.5,
                0.5,
                "No attention weights\navailable for this model",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
            )
            axes[1].axis("off")

        # Question and model info
        info_text = f"Question:\n{question}\n\n"
        info_text += f"Model Configuration:\n"
        info_text += f"• Attention Layers: {self.model_config.get('num_attention_layers', 'N/A')}\n"
        info_text += f"• LRM: {'Active' if self.model_config.get('use_lrm', False) else 'Inactive'}\n"
        info_text += (
            f"• Visual Encoder: {self.model_config.get('visual_encoder_type', 'N/A')}\n"
        )
        info_text += (
            f"• Text Encoder: {self.model_config.get('text_encoder_type', 'N/A')}"
        )

        axes[2].text(
            0.05,
            0.95,
            info_text,
            ha="left",
            va="top",
            transform=axes[2].transAxes,
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
        )
        axes[2].axis("off")

        plt.tight_layout()
        plt.savefig(
            save_dir / f"attention_sample_{sample_id:03d}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close()

    def generate_analysis_plots(self, output_dir: Path):
        """Generate additional analysis plots."""
        print(f"\nGenerating analysis plots...")

        if not self.results:
            print("No results to analyze. Run evaluation first.")
            return

        # Accuracy comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Plot 1: Overall accuracy by split
        splits = []
        accuracies = []
        for split_name, split_results in self.results.items():
            splits.append(split_name.capitalize())
            accuracies.append(split_results["metrics"]["accuracy"])

        axes[0, 0].bar(
            splits,
            accuracies,
            color=["skyblue", "lightcoral", "lightgreen"][: len(splits)],
        )
        axes[0, 0].set_title("Overall Accuracy by Split")
        axes[0, 0].set_ylabel("Accuracy")
        axes[0, 0].set_ylim(0, 1)

        # Plot 2: Closed vs Open question accuracy
        question_types = ["Closed", "Open"]
        for i, split_name in enumerate(self.results.keys()):
            metrics = self.results[split_name]["metrics"]
            closed_acc = metrics["closed_accuracy"]
            open_acc = metrics["open_accuracy"]

            x_pos = np.arange(len(question_types))
            axes[0, 1].bar(
                x_pos + i * 0.25,
                [closed_acc, open_acc],
                width=0.25,
                label=split_name.capitalize(),
                alpha=0.8,
            )

        axes[0, 1].set_title("Accuracy by Question Type")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(question_types)
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0, 1)

        # Plot 3: Model configuration summary
        config_text = f"Model Configuration Summary:\n\n"
        config_text += f"Architecture: LRCN (Layer-Residual Co-Attention Network)\n"
        config_text += (
            f"Visual Encoder: {self.model_config.get('visual_encoder_type', 'N/A')}\n"
        )
        config_text += (
            f"Text Encoder: {self.model_config.get('text_encoder_type', 'N/A')}\n"
        )
        config_text += (
            f"Hidden Dimension: {self.model_config.get('hidden_dim', 'N/A')}\n"
        )
        config_text += f"Attention Layers: {self.model_config.get('num_attention_layers', 'N/A')}\n"
        config_text += f"Layer-Residual Mechanism: {'Active' if self.model_config.get('use_lrm', False) else 'Inactive'}\n"
        config_text += (
            f"Number of Classes: {self.model_config.get('num_classes', 'N/A')}\n"
        )

        axes[1, 0].text(
            0.1,
            0.9,
            config_text,
            ha="left",
            va="top",
            transform=axes[1, 0].transAxes,
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow"),
        )
        axes[1, 0].axis("off")

        # Plot 4: Loss comparison
        if len(self.results) > 1:
            splits = []
            losses = []
            for split_name, split_results in self.results.items():
                splits.append(split_name.capitalize())
                losses.append(split_results["metrics"]["loss"])

            axes[1, 1].bar(
                splits, losses, color=["orange", "red", "purple"][: len(splits)]
            )
            axes[1, 1].set_title("Loss by Split")
            axes[1, 1].set_ylabel("Loss")
        else:
            axes[1, 1].text(
                0.5,
                0.5,
                "Multiple splits needed\nfor loss comparison",
                ha="center",
                va="center",
                transform=axes[1, 1].transAxes,
            )
            axes[1, 1].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / "analysis_plots.png", dpi=300, bbox_inches="tight")
        plt.close()


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Medical VQA LRCN model")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to saved model (.pth file)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["slake", "vqa-rad", "both"],
        default="both",
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--attention-samples",
        type=int,
        default=10,
        help="Number of attention map samples to generate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Evaluation batch size"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    print("Medical VQA LRCN - Model Evaluation")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_dir}")

    try:
        # Initialize evaluator
        evaluator = ModelEvaluator(args.model_path, args.device)
        evaluator.load_model()

        # Load test data (you'll need to implement proper test set loading)
        # For now, using a simple approach
        from ..scripts.train_lrcn import MedVQADataset, collate_fn
        from ..preprocessing.image_preprocessing import ImagePreprocessor
        from ..preprocessing.text_preprocessing import (
            QuestionPreprocessor,
            AnswerPreprocessor,
        )

        # Load preprocessing artifacts
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            print("Preprocessing artifacts not found. Run preprocess-datasets first.")
            return 1

        # Load datasets for evaluation
        if args.dataset == "slake":
            test_data = [item for item in load_slake() if item["split"] == "test"]
        elif args.dataset == "vqa-rad":
            test_data = [item for item in load_vqa_rad() if item["split"] == "test"]
        else:
            slake_test = [item for item in load_slake() if item["split"] == "test"]
            vqarad_test = [item for item in load_vqa_rad() if item["split"] == "test"]
            test_data = slake_test + vqarad_test

        if not test_data:
            print("No test data found!")
            return 1

        print(f"Test samples: {len(test_data)}")

        # Initialize processors (simplified version for evaluation)
        image_processor = ImagePreprocessor()
        question_processor = QuestionPreprocessor()
        answer_processor = AnswerPreprocessor()

        # Build vocabularies from the test data (not ideal, but for demonstration)
        questions = [item["question"] for item in test_data]
        answers = [item["answer"] for item in test_data]
        question_processor.build_vocab(questions)
        answer_processor.build_vocab(answers)

        # Create dataset and dataloader
        test_dataset = MedVQADataset(
            test_data, image_processor, question_processor, answer_processor
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        # Run evaluation
        test_metrics = evaluator.evaluate_model(test_loader, "test")

        # Generate results table
        results_df = evaluator.generate_results_table(output_dir)

        # Generate analysis plots
        evaluator.generate_analysis_plots(output_dir)

        # Generate attention visualizations
        evaluator.visualize_attention_maps(
            test_loader, output_dir, args.attention_samples
        )

        # Print summary
        print(f"\n" + "=" * 60)
        print("EVALUATION COMPLETE")
        print(f"=" * 60)
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
        print(f"Closed Questions: {test_metrics['closed_accuracy']:.4f}")
        print(f"Open Questions: {test_metrics['open_accuracy']:.4f}")
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"\nResults saved to: {output_dir}")
        print(f"- Results table: evaluation_results.csv")
        print(f"- Analysis plots: analysis_plots.png")
        print(f"- Attention maps: attention_maps/")

        return 0

    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
