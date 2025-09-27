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

    # Extract metrics from nested structure
    train_metrics = history["train"]
    val_metrics = history["val"]

    epochs = range(1, len(train_metrics) + 1)

    # Loss
    train_losses = [epoch["loss"] for epoch in train_metrics]
    axes[0, 0].plot(epochs, train_losses, label="Train", linewidth=2)
    if val_metrics:  # Only plot validation if available
        val_losses = [epoch["loss"] for epoch in val_metrics]
        axes[0, 0].plot(
            epochs[: len(val_losses)], val_losses, label="Validation", linewidth=2
        )
    axes[0, 0].set_title("Loss", fontweight="bold")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Overall Accuracy
    train_accuracies = [epoch["accuracy"] for epoch in train_metrics]
    axes[0, 1].plot(epochs, train_accuracies, label="Train", linewidth=2)
    if val_metrics:  # Only plot validation if available
        val_accuracies = [epoch["accuracy"] for epoch in val_metrics]
        axes[0, 1].plot(
            epochs[: len(val_accuracies)],
            val_accuracies,
            label="Validation",
            linewidth=2,
        )
    axes[0, 1].set_title("Overall Accuracy", fontweight="bold")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Accuracy")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Closed-ended Accuracy
    train_closed_accuracies = [epoch["closed_accuracy"] for epoch in train_metrics]
    axes[1, 0].plot(epochs, train_closed_accuracies, label="Train", linewidth=2)
    if val_metrics:
        val_closed_accuracies = [epoch["closed_accuracy"] for epoch in val_metrics]
        axes[1, 0].plot(
            epochs[: len(val_closed_accuracies)],
            val_closed_accuracies,
            label="Validation",
            linewidth=2,
        )
    axes[1, 0].set_title("Closed-ended Accuracy", fontweight="bold")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Open-ended Accuracy
    train_open_accuracies = [epoch["open_accuracy"] for epoch in train_metrics]
    axes[1, 1].plot(epochs, train_open_accuracies, label="Train", linewidth=2)
    if val_metrics:
        val_open_accuracies = [epoch["open_accuracy"] for epoch in val_metrics]
        axes[1, 1].plot(
            epochs[: len(val_open_accuracies)],
            val_open_accuracies,
            label="Validation",
            linewidth=2,
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

    # Extract metrics from nested structure
    train_metrics = final_results.get("final_train_metrics", {})
    val_metrics = final_results.get("final_val_metrics", {})

    # Accuracy comparison
    categories = ["Overall", "Closed-ended", "Open-ended"]
    train_accs = [
        train_metrics.get("accuracy", 0),
        train_metrics.get("closed_accuracy", 0),
        train_metrics.get("open_accuracy", 0),
    ]
    val_accs = [
        val_metrics.get("accuracy", 0),
        val_metrics.get("closed_accuracy", 0),
        val_metrics.get("open_accuracy", 0),
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
    training_time_hours = final_results.get("training_time", 0) / 3600
    stats_data = [
        ["Best Val Accuracy", f"{final_results.get('best_val_accuracy', 0):.4f}"],
        ["Training Time", f"{training_time_hours:.2f} hours"],
        ["Final Train Loss", f"{train_metrics.get('loss', 0):.4f}"],
        ["Final Train Accuracy", f"{train_metrics.get('accuracy', 0):.4f}"],
        [
            "Epochs Completed",
            f"{final_results.get('configuration', {}).get('epochs', 0)}",
        ],
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

    # Extract metrics from nested structure
    train_metrics = final_results.get("final_train_metrics", {})
    val_metrics = final_results.get("final_val_metrics", {})
    config = final_results.get("configuration", {})

    with open(report_path, "w") as f:
        f.write("MEDICAL VQA LRCN TRAINING REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write("FINAL PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        f.write(
            f"Best Validation Accuracy:     {final_results.get('best_val_accuracy', 0):.4f}\n"
        )
        f.write(
            f"Final Train Accuracy:         {train_metrics.get('accuracy', 0):.4f}\n"
        )
        f.write(
            f"Final Validation Accuracy:    {val_metrics.get('accuracy', 0):.4f}\n\n"
        )

        f.write("ACCURACY BY QUESTION TYPE:\n")
        f.write("-" * 30 + "\n")
        f.write(
            f"Closed-ended (Train):         {train_metrics.get('closed_accuracy', 0):.4f}\n"
        )
        f.write(
            f"Closed-ended (Validation):    {val_metrics.get('closed_accuracy', 0):.4f}\n"
        )
        f.write(
            f"Open-ended (Train):           {train_metrics.get('open_accuracy', 0):.4f}\n"
        )
        f.write(
            f"Open-ended (Validation):      {val_metrics.get('open_accuracy', 0):.4f}\n\n"
        )

        training_time_hours = final_results.get("training_time", 0) / 3600
        f.write("TRAINING DETAILS:\n")
        f.write("-" * 30 + "\n")
        f.write(f"Total Training Time:          {training_time_hours:.2f} hours\n")
        f.write(f"Epochs Completed:             {config.get('epochs', 0)}\n")
        f.write(f"Dataset:                      {config.get('dataset', 'unknown')}\n")
        f.write(
            f"Batch Size:                   {config.get('batch_size', 'unknown')}\n"
        )
        f.write(
            f"Learning Rate:                {config.get('learning_rate', 'unknown')}\n\n"
        )

        train_acc = train_metrics.get("accuracy", 0)
        val_acc = val_metrics.get("accuracy", 0)
        overfitting_indicator = train_acc - val_acc

        f.write("ANALYSIS:\n")
        f.write("-" * 30 + "\n")
        if overfitting_indicator > 0.15:  # High threshold
            f.write(
                f"⚠️  High overfitting detected (train-val gap: {overfitting_indicator:.4f})\n"
            )
        elif overfitting_indicator > 0.05:  # Moderate threshold
            f.write(
                f"⚡ Moderate overfitting detected (train-val gap: {overfitting_indicator:.4f})\n"
            )
        else:
            f.write("✅ Good generalization (low train-val gap)\n")

        val_closed_acc = val_metrics.get("closed_accuracy", 0)
        val_open_acc = val_metrics.get("open_accuracy", 0)
        if val_closed_acc > val_open_acc:
            f.write("📊 Model performs better on closed-ended questions\n")
        elif val_open_acc > val_closed_acc:
            f.write("📝 Model performs better on open-ended questions\n")
        else:
            f.write("⚖️ Similar performance on both question types\n")

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
