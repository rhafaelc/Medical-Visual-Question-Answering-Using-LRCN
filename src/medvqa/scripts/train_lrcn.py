"""Training script for Medical VQA LRCN model."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Force using only first GPU to avoid DataParallel issues
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Import required modules at top level
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm
    import numpy as np
except ImportError as e:
    print(f"Missing dependencies: {e}")
    sys.exit(1)

# Import ModelConfig at module level for argument defaults
try:
    from ..core.config import ModelConfig
    from ..models.lrcn import LRCN
    from ..datamodules.common import load_slake, load_vqa_rad
    from ..preprocessing.image_preprocessing import ImagePreprocessor
    from ..preprocessing.text_preprocessing import (
        QuestionPreprocessor,
        AnswerPreprocessor,
    )
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core.config import ModelConfig
    from models.lrcn import LRCN
    from datamodules.common import load_slake, load_vqa_rad
    from preprocessing.image_preprocessing import ImagePreprocessor
    from preprocessing.text_preprocessing import (
        QuestionPreprocessor,
        AnswerPreprocessor,
    )


class MedVQADataset(Dataset):
    """Dataset for Medical VQA with on-the-fly preprocessing."""

    def __init__(self, data, image_processor, question_processor, answer_processor):
        self.data = data
        self.image_processor = image_processor
        self.question_processor = question_processor
        self.answer_processor = answer_processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        try:
            image = self.image_processor.load_and_preprocess(item["image"])
        except Exception as e:
            raise RuntimeError(f"Failed to load image {item['image']}: {e}") from e

        # Pass raw question string to model
        question = item["question"]

        answer_idx = self.answer_processor.encode(item["answer"])

        return {
            "image": image,
            "question": question,
            "answer": torch.tensor(answer_idx, dtype=torch.long),
            "answer_type": item["answer_type"],
            "id": item["id"],
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    images = torch.stack([item["image"] for item in batch])

    # Keep questions as list of strings
    questions = [item["question"] for item in batch]

    answers = torch.stack([item["answer"] for item in batch])
    answer_types = [item["answer_type"] for item in batch]
    ids = [item["id"] for item in batch]

    return {
        "images": images,
        "questions": questions,
        "answers": answers,
        "answer_types": answer_types,
        "ids": ids,
    }


def compute_accuracy(outputs, targets, answer_types):
    """Compute overall, closed, and open question accuracies."""
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == targets).float()

    overall_acc = correct.mean().item()

    closed_mask = torch.tensor([t == "closed" for t in answer_types])
    open_mask = torch.tensor([t == "open" for t in answer_types])

    closed_acc = correct[closed_mask].mean().item() if closed_mask.sum() > 0 else 0.0
    open_acc = correct[open_mask].mean().item() if open_mask.sum() > 0 else 0.0

    return overall_acc, closed_acc, open_acc


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_closed_acc = 0.0
    total_open_acc = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        images = batch["images"].to(device)
        questions = batch["questions"]  # Keep as list of strings
        answers = batch["answers"].to(device)
        answer_types = batch["answer_types"]

        optimizer.zero_grad()

        outputs = model(images, questions)
        if isinstance(outputs, dict):
            outputs = outputs.get(
                "logits", outputs.get("predictions", list(outputs.values())[0])
            )

        loss = criterion(outputs, answers)
        loss.backward()
        optimizer.step()

        if scheduler:
            scheduler.step()

        overall_acc, closed_acc, open_acc = compute_accuracy(
            outputs, answers, answer_types
        )

        total_loss += loss.item()
        total_acc += overall_acc
        total_closed_acc += closed_acc
        total_open_acc += open_acc
        num_batches += 1

        progress_bar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{overall_acc:.4f}",
                "Closed": f"{closed_acc:.4f}",
                "Open": f"{open_acc:.4f}",
            }
        )

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_acc / num_batches,
        "closed_accuracy": total_closed_acc / num_batches,
        "open_accuracy": total_open_acc / num_batches,
    }


def validate_epoch(model, dataloader, criterion, device):
    """Validate model for one epoch."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_closed_acc = 0.0
    total_open_acc = 0.0
    num_batches = 0

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Validation")

        for batch in progress_bar:
            images = batch["images"].to(device)
            questions = batch["questions"]  # Keep as list of strings
            answers = batch["answers"].to(device)
            answer_types = batch["answer_types"]

            outputs = model(images, questions)
            if isinstance(outputs, dict):
                outputs = outputs.get(
                    "logits", outputs.get("predictions", list(outputs.values())[0])
                )

            loss = criterion(outputs, answers)

            overall_acc, closed_acc, open_acc = compute_accuracy(
                outputs, answers, answer_types
            )

            total_loss += loss.item()
            total_acc += overall_acc
            total_closed_acc += closed_acc
            total_open_acc += open_acc
            num_batches += 1

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{overall_acc:.4f}",
                    "Closed": f"{closed_acc:.4f}",
                    "Open": f"{open_acc:.4f}",
                }
            )

    return {
        "loss": total_loss / num_batches,
        "accuracy": total_acc / num_batches,
        "closed_accuracy": total_closed_acc / num_batches,
        "open_accuracy": total_open_acc / num_batches,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Medical VQA LRCN model")

    # Required research parameters
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["slake", "vqa-rad"],
        required=True,
        help="Dataset to use for training",
    )
    parser.add_argument(
        "--attention-layers",
        type=int,
        choices=[4, 6, 8, 10, 12],
        required=True,
        help="Number of attention layers (research parameter)",
    )
    parser.add_argument(
        "--use-lrm", action="store_true", help="Use Layer-Residual Mechanism"
    )
    parser.add_argument(
        "--no-lrm",
        action="store_true",
        help="Disable Layer-Residual Mechanism (for ablation)",
    )
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument(
        "--warmup-epochs", type=int, required=True, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--decay-epochs",
        type=int,
        required=True,
        help="Decay learning rate every N epochs",
    )
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size")
    parser.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument(
        "--early-stopping",
        type=int,
        required=True,
        help="Early stopping patience (epochs without improvement)",
    )

    # Optional parameters
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results",
    )
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save the trained model",
    )

    # Advanced training configuration (with defaults from ModelConfig)
    parser.add_argument(
        "--attention-heads", type=int, default=8, help="Number of attention heads (h)"
    )
    parser.add_argument(
        "--hidden-dim", type=int, default=512, help="Hidden dimension (d)"
    )
    parser.add_argument(
        "--feedforward-dim", type=int, default=2048, help="Feed-forward dimension (dff)"
    )
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument(
        "--coverage-percentile",
        type=float,
        default=95.0,
        help="Percentile for L_max and top-K answer selection",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "adamw", "sgd"],
        default="adam",
        help="Optimizer type",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.01, help="Weight decay for optimizer"
    )

    args = parser.parse_args()

    # Determine LRM setting
    if args.use_lrm and args.no_lrm:
        print("Error: Cannot specify both --use-lrm and --no-lrm")
        return 1
    elif args.use_lrm:
        use_lrm = True
    elif args.no_lrm:
        use_lrm = False
    else:
        use_lrm = ModelConfig.USE_LRM

    # Device setup - Force single GPU (T4)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda:0")  # Use only first GPU
            print(f"🎯 Using single GPU: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("⚠️  No GPU available, using CPU")
    else:
        device = torch.device(args.device)

    print(f"Medical VQA LRCN Training")
    print(f"Dataset: {args.dataset}")
    print(f"Attention Layers: {args.attention_layers}")
    print(f"Layer-Residual Mechanism: {'Active' if use_lrm else 'Inactive'}")
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")

    try:
        # Load and prepare data
        print("Loading datasets...")
        if args.dataset == "slake":
            all_data = list(load_slake())
        else:
            all_data = list(load_vqa_rad())

        print(f"Using {args.dataset} dataset with {len(all_data)} samples")

        # Split data
        train_data = [item for item in all_data if item["split"] == "train"]
        val_data = [item for item in all_data if item["split"] == "validation"]

        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")

        if not train_data:
            print("No training data found!")
            return 1

        # Initialize processors
        print("Initializing processors...")
        image_processor = ImagePreprocessor()
        question_processor = QuestionPreprocessor()
        answer_processor = AnswerPreprocessor()

        # Build vocabularies
        questions = [item["question"] for item in train_data]
        answers = [item["answer"] for item in train_data]

        question_processor.build_vocab(questions)
        answer_processor.build_vocab(answers)

        print(f"Question vocabulary size: {len(question_processor.vocab)}")
        print(f"Answer vocabulary size: {len(answer_processor.answer_vocab)}")

        # Create datasets and dataloaders
        train_dataset = MedVQADataset(
            train_data, image_processor, question_processor, answer_processor
        )
        val_dataset = (
            MedVQADataset(
                val_data, image_processor, question_processor, answer_processor
            )
            if val_data
            else None
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = (
            DataLoader(
                val_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
            if val_dataset
            else None
        )

        # Initialize model
        print("Initializing model...")
        model = LRCN(
            num_classes=len(answer_processor.answer_vocab),
            hidden_dim=ModelConfig.HIDDEN_DIM,
            num_attention_layers=args.attention_layers,
            num_heads=ModelConfig.ATTENTION_HEADS,
            feedforward_dim=ModelConfig.HIDDEN_DIM * 4,
            dropout=0.1,
            use_lrm=use_lrm,
            visual_encoder_type="vit",
            text_encoder_type="biobert",
        )

        # Multi-GPU setup - DISABLED (use single GPU for stability)
        # Force single GPU usage
        print("Using single GPU (multi-GPU disabled for stability)")
        model.to(device)

        # Initialize training components with Cross-Entropy Loss
        criterion = nn.CrossEntropyLoss()

        # Configure optimizer
        if args.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        elif args.optimizer == "adamw":
            optimizer = optim.AdamW(
                model.parameters(), lr=args.lr, weight_decay=args.weight_decay
            )
        elif args.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=args.lr,
                weight_decay=args.weight_decay,
                momentum=0.9,
            )

        # Learning rate scheduler with warmup
        def warmup_schedule(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            else:
                # Decay every decay_epochs
                decay_factor = 0.1 ** (
                    (epoch - args.warmup_epochs) // args.decay_epochs
                )
                return decay_factor

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_schedule)

        # Create results directory
        results_dir = Path(args.results_dir)
        results_dir.mkdir(exist_ok=True)

        # Training loop
        print("Starting training...")
        start_time = time.time()

        best_val_acc = 0.0
        epochs_without_improvement = 0
        history = {"train": [], "val": []}

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")

            # Train
            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device, scheduler
            )
            history["train"].append(train_metrics)

            print(
                f"Train - Loss: {train_metrics['loss']:.4f}, "
                f"Acc: {train_metrics['accuracy']:.4f}, "
                f"Closed: {train_metrics['closed_accuracy']:.4f}, "
                f"Open: {train_metrics['open_accuracy']:.4f}"
            )

            # Validate
            if val_loader:
                val_metrics = validate_epoch(model, val_loader, criterion, device)
                history["val"].append(val_metrics)

                print(
                    f"Val   - Loss: {val_metrics['loss']:.4f}, "
                    f"Acc: {val_metrics['accuracy']:.4f}, "
                    f"Closed: {val_metrics['closed_accuracy']:.4f}, "
                    f"Open: {val_metrics['open_accuracy']:.4f}"
                )

                # Early stopping logic
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    epochs_without_improvement = 0
                    # Save best model
                    best_model_path = results_dir / "best_model.pth"
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "best_val_acc": best_val_acc,
                            "args": vars(args),
                        },
                        best_model_path,
                    )
                    print(f"New best model saved! Val Acc: {best_val_acc:.4f}")
                else:
                    epochs_without_improvement += 1

                # Check early stopping
                if epochs_without_improvement >= args.early_stopping:
                    print(
                        f"\nEarly stopping triggered after {epochs_without_improvement} epochs without improvement"
                    )
                    print(f"Best validation accuracy: {best_val_acc:.4f}")
                    break

        total_time = time.time() - start_time

        # Final results
        final_results = {
            "configuration": {
                "dataset": args.dataset,
                "attention_layers": args.attention_layers,
                "use_lrm": use_lrm,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "warmup_epochs": args.warmup_epochs,
                "decay_epochs": args.decay_epochs,
                "early_stopping": args.early_stopping,
            },
            "final_train_metrics": history["train"][-1] if history["train"] else {},
            "final_val_metrics": history["val"][-1] if history["val"] else {},
            "best_val_accuracy": best_val_acc,
            "training_time": total_time,
        }

        # Save model if requested
        if args.save_model:
            model_config = {
                "num_classes": len(answer_processor.answer_vocab),
                "hidden_dim": args.hidden_dim,
                "num_attention_layers": args.attention_layers,
                "num_heads": args.attention_heads,
                "feedforward_dim": args.feedforward_dim,
                "dropout": args.dropout,
                "use_lrm": use_lrm,
                "visual_encoder_type": "vit",
                "text_encoder_type": "biobert",
                "dataset": args.dataset,
            }

            model_path = (
                results_dir
                / f"model_{args.dataset}_L{args.attention_layers}_{'LRM' if use_lrm else 'NoLRM'}.pth"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": model_config,
                    "train_accuracy": (
                        history["train"][-1]["accuracy"] if history["train"] else 0.0
                    ),
                    "val_accuracy": best_val_acc,
                    "question_vocab": question_processor.vocab,
                    "answer_vocab": answer_processor.answer_vocab,
                    "epoch": args.epochs,
                },
                model_path,
            )

            print(f"Model saved to: {model_path}")

        # Save history and results
        with open(results_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        with open(results_dir / "final_results.json", "w") as f:
            json.dump(final_results, f, indent=2)

        print(f"\nTraining completed in {total_time/60:.1f} minutes")
        print(f"Best validation accuracy: {best_val_acc:.4f}")
        print(f"Results saved to: {results_dir}")

        return 0

    except Exception as e:
        print(f"Training failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
