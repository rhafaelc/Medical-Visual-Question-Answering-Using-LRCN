"""Results visualization script for Medical VQA LRCN training."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from ..core.download_utils import DownloadUtils
from ..core.config import ModelConfig


def plot_training_curves(history, save_path):
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("Medical VQA LRCN Training Results", fontsize=16, fontweight="bold")

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0, 0].plot(epochs, history["train_loss"], label="Train", linewidth=2)
    axes[0, 0].plot(epochs, history["val_loss"], label="Validation", linewidth=2)
    axes[0, 0].set_title("Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Overall Accuracy
    axes[0, 1].plot(epochs, history["train_accuracy"], label="Train", linewidth=2)
    axes[0, 1].plot(epochs, history["val_accuracy"], label="Validation", linewidth=2)
    axes[0, 1].set_title("Overall Accuracy", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Closed-ended Accuracy
    axes[1, 0].plot(
        epochs, history["train_closed_accuracy"], label="Train", linewidth=2
    )
    axes[1, 0].plot(
        epochs, history["val_closed_accuracy"], label="Validation", linewidth=2
    )
    axes[1, 0].set_title("Closed-ended Accuracy", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Open-ended Accuracy
    axes[1, 1].plot(epochs, history["train_open_accuracy"], label="Train", linewidth=2)
    axes[1, 1].plot(
        epochs, history["val_open_accuracy"], label="Validation", linewidth=2
    )
    axes[1, 1].set_title("Open-ended Accuracy", fontweight="bold")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Accuracy")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_metrics_summary(final_results, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Final Model Performance Summary", fontsize=16, fontweight="bold")

    # Accuracy comparison
    categories = ["Overall", "Closed-ended", "Open-ended"]
    train_accs = [
        final_results["final_train_accuracy"],
        final_results["final_train_closed_acc"],
        final_results["final_train_open_acc"],
    ]
    val_accs = [
        final_results["final_val_accuracy"],
        final_results["final_val_closed_acc"],
        final_results["final_val_open_acc"],
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = axes[0].bar(x - width / 2, train_accs, width, label="Train", alpha=0.8)
    bars2 = axes[0].bar(x + width / 2, val_accs, width, label="Validation", alpha=0.8)

    axes[0].set_title("Final Accuracy by Category", fontweight="bold")
    axes[0].set_xlabel("Question Type")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(categories)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    for bar in bars2:
        height = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.005,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Model statistics
    stats_data = [
        ["Best Val Accuracy", f"{final_results['best_val_accuracy']:.4f}"],
        ["Training Time", f"{final_results['training_time_minutes']:.1f} min"],
        ["Total Parameters", f"{final_results['total_parameters']:,}"],
        ["Trainable Parameters", f"{final_results['trainable_parameters']:,}"],
        ["Epochs Completed", f"{final_results['epochs_completed']}"],
    ]

    axes[1].axis("tight")
    axes[1].axis("off")
    table = axes[1].table(
        cellText=stats_data,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        loc="center",
        colWidths=[0.6, 0.4],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)

    # Style the table
    for i in range(len(stats_data) + 1):
        for j in range(2):
            cell = table[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor("#4472C4")
                cell.set_text_props(weight="bold", color="white")
            else:
                cell.set_facecolor("#F2F2F2" if i % 2 == 0 else "white")

    axes[1].set_title("Training Statistics", fontweight="bold", pad=20)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def generate_report(final_results, results_dir):
    report_path = results_dir / "training_report.txt"

    with open(report_path, "w") as f:
        f.write("MEDICAL VQA LRCN TRAINING REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("FINAL PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(
            f"Best Validation Accuracy:     {final_results['best_val_accuracy']:.4f}\n"
        )
        f.write(
            f"Final Train Accuracy:         {final_results['final_train_accuracy']:.4f}\n"
        )
        f.write(
            f"Final Validation Accuracy:    {final_results['final_val_accuracy']:.4f}\n\n"
        )

        f.write("ACCURACY BY QUESTION TYPE:\n")
        f.write("-" * 30 + "\n")
        f.write(
            f"Closed-ended (Train):         {final_results['final_train_closed_acc']:.4f}\n"
        )
        f.write(
            f"Closed-ended (Validation):    {final_results['final_val_closed_acc']:.4f}\n"
        )
        f.write(
            f"Open-ended (Train):           {final_results['final_train_open_acc']:.4f}\n"
        )
        f.write(
            f"Open-ended (Validation):      {final_results['final_val_open_acc']:.4f}\n\n"
        )

        f.write("TRAINING DETAILS:\n")
        f.write("-" * 30 + "\n")
        f.write(
            f"Total Training Time:          {final_results['training_time_minutes']:.1f} minutes\n"
        )
        f.write(f"Epochs Completed:             {final_results['epochs_completed']}\n")
        f.write(
            f"Total Parameters:             {final_results['total_parameters']:,}\n"
        )
        f.write(
            f"Trainable Parameters:         {final_results['trainable_parameters']:,}\n\n"
        )

        overfitting_indicator = (
            final_results["final_train_accuracy"] - final_results["final_val_accuracy"]
        )
        f.write("ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        if overfitting_indicator > ModelConfig.HIGH_OVERFITTING_THRESHOLD:
            f.write(
                f"⚠️  High overfitting detected (train-val gap > {ModelConfig.HIGH_OVERFITTING_THRESHOLD})\n"
            )
        elif overfitting_indicator > ModelConfig.MODERATE_OVERFITTING_THRESHOLD:
            f.write(
                f"⚡ Moderate overfitting detected (train-val gap > {ModelConfig.MODERATE_OVERFITTING_THRESHOLD})\n"
            )
        else:
            f.write("✅ Good generalization (low train-val gap)\n")

        if final_results["final_val_closed_acc"] > final_results["final_val_open_acc"]:
            f.write("📊 Model performs better on closed-ended questions\n")
        else:
            f.write("📝 Model performs better on open-ended questions\n")

    print(f"Training report saved to: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize Medical VQA LRCN training results"
    )
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots interactively"
    )
    args = parser.parse_args()

    try:
        project_root = DownloadUtils.project_root(__file__)
        results_dir = project_root / args.results_dir

        if not results_dir.exists():
            print(f"Results directory not found: {results_dir}")
            return 1

        history_path = results_dir / "training_history.json"
        results_path = results_dir / "final_results.json"

        if not history_path.exists():
            print(f"Training history not found: {history_path}")
            return 1

        if not results_path.exists():
            print(f"Final results not found: {results_path}")
            return 1

        with open(history_path, "r") as f:
            history = json.load(f)

        with open(results_path, "r") as f:
            final_results = json.load(f)

        print("Generating training curves...")
        plot_training_curves(history, results_dir / "training_curves.png")

        print("Generating performance summary...")
        plot_metrics_summary(final_results, results_dir / "performance_summary.png")

        print("Generating training report...")
        generate_report(final_results, results_dir)

        print("\nFINAL RESULTS SUMMARY:")
        print("=" * 40)
        print(f"Best Validation Accuracy: {final_results['best_val_accuracy']:.4f}")
        print(f"Training Time: {final_results['training_time_minutes']:.1f} minutes")
        print(f"Total Parameters: {final_results['total_parameters']:,}")
        print(f"Files saved to: {results_dir}")

        return 0

    except Exception as e:
        print(f"Results visualization failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
