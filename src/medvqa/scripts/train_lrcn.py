"""Training script for Medical VQA LRCN model."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List


def main():
    def __init__(
        self,
        data: List[Dict],
        image_processor: ImagePreprocessor,
        question_processor: QuestionPreprocessor,
        answer_processor: AnswerPreprocessor,
    ):
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
        except Exception:
            image = torch.randn(3, 224, 224)

        question_tokens = self.question_processor.encode(item["question"])
        question_tensor = torch.tensor(question_tokens, dtype=torch.long)

        answer_idx = self.answer_processor.encode(item["answer"])

        return {
            "image": image,
            "question": question_tensor,
            "answer": torch.tensor(answer_idx, dtype=torch.long),
            "answer_type": item["answer_type"],
            "id": item["id"],
        }


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    questions = torch.stack([item["question"] for item in batch])
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
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == targets).float()

    overall_acc = correct.mean().item()

    closed_mask = torch.tensor([t == "closed" for t in answer_types])
    open_mask = torch.tensor([t == "open" for t in answer_types])

    closed_acc = correct[closed_mask].mean().item() if closed_mask.sum() > 0 else 0.0
    open_acc = correct[open_mask].mean().item() if open_mask.sum() > 0 else 0.0

    return overall_acc, closed_acc, open_acc


def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    total_closed_acc = 0.0
    total_open_acc = 0.0
    num_batches = 0

    progress_bar = tqdm(dataloader, desc="Training")

    for batch in progress_bar:
        images = batch["images"].to(device)
        questions = batch["questions"].to(device)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        acc, closed_acc, open_acc = compute_accuracy(outputs, answers, answer_types)

        total_loss += loss.item()
        total_acc += acc
        total_closed_acc += closed_acc
        total_open_acc += open_acc
        num_batches += 1

        progress_bar.set_postfix(
            {
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{acc:.4f}",
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
            questions = batch["questions"].to(device)
            answers = batch["answers"].to(device)
            answer_types = batch["answer_types"]

            outputs = model(images, questions)
            if isinstance(outputs, dict):
                outputs = outputs.get(
                    "logits", outputs.get("predictions", list(outputs.values())[0])
                )

            loss = criterion(outputs, answers)

            acc, closed_acc, open_acc = compute_accuracy(outputs, answers, answer_types)

            total_loss += loss.item()
            total_acc += acc
            total_closed_acc += closed_acc
            total_open_acc += open_acc
            num_batches += 1

            progress_bar.set_postfix(
                {
                    "Loss": f"{loss.item():.4f}",
                    "Acc": f"{acc:.4f}",
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
    parser = argparse.ArgumentParser(description="Train Medical VQA LRCN model")
    parser.add_argument(
        "--epochs", type=int, default=15, help="Number of training epochs"
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save-model", action="store_true", help="Save trained model")
    parser.add_argument(
        "--results-dir", type=str, default="results", help="Results directory"
    )
    parser.add_argument(
        "--subset", type=int, help="Use subset of data for faster training"
    )
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, Dataset
        import numpy as np
        from tqdm.auto import tqdm

        from ..models.lrcn import LRCN
        from ..datamodules.common import load_slake, load_vqa_rad
        from ..preprocessing.image_preprocessing import ImagePreprocessor
        from ..preprocessing.text_preprocessing import QuestionPreprocessor, AnswerPreprocessor
        from ..core.download_utils import DownloadUtils

        class MedVQADataset(Dataset):
            def __init__(
                self,
                data: List[Dict],
                image_processor: ImagePreprocessor,
                question_processor: QuestionPreprocessor,
                answer_processor: AnswerPreprocessor,
            ):
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
                except Exception:
                    image = torch.randn(3, 224, 224)

                question_tokens = self.question_processor.encode(item["question"])
                question_tensor = torch.tensor(question_tokens, dtype=torch.long)

                answer_idx = self.answer_processor.encode(item["answer"])

                return {
                    "image": image,
                    "question": question_tensor,
                    "answer": torch.tensor(answer_idx, dtype=torch.long),
                    "answer_type": item["answer_type"],
                    "id": item["id"],
                }

        def collate_fn(batch):
            images = torch.stack([item["image"] for item in batch])
            questions = torch.stack([item["question"] for item in batch])
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
            predictions = torch.argmax(outputs, dim=1)
            correct = (predictions == targets).float()

            overall_acc = correct.mean().item()

            closed_mask = torch.tensor([t == "closed" for t in answer_types])
            open_mask = torch.tensor([t == "open" for t in answer_types])

            closed_acc = correct[closed_mask].mean().item() if closed_mask.sum() > 0 else 0.0
            open_acc = correct[open_mask].mean().item() if open_mask.sum() > 0 else 0.0

            return overall_acc, closed_acc, open_acc

        def train_epoch(model, dataloader, optimizer, criterion, device, scheduler=None):
            model.train()
            total_loss = 0.0
            total_acc = 0.0
            total_closed_acc = 0.0
            total_open_acc = 0.0
            num_batches = 0

            progress_bar = tqdm(dataloader, desc="Training")

            for batch in progress_bar:
                images = batch["images"].to(device)
                questions = batch["questions"].to(device)
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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                if scheduler:
                    scheduler.step()

                acc, closed_acc, open_acc = compute_accuracy(outputs, answers, answer_types)

                total_loss += loss.item()
                total_acc += acc
                total_closed_acc += closed_acc
                total_open_acc += open_acc
                num_batches += 1

                progress_bar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Acc": f"{acc:.4f}",
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
                    questions = batch["questions"].to(device)
                    answers = batch["answers"].to(device)
                    answer_types = batch["answer_types"]

                    outputs = model(images, questions)
                    if isinstance(outputs, dict):
                        outputs = outputs.get(
                            "logits", outputs.get("predictions", list(outputs.values())[0])
                        )

                    loss = criterion(outputs, answers)

                    acc, closed_acc, open_acc = compute_accuracy(outputs, answers, answer_types)

                    total_loss += loss.item()
                    total_acc += acc
                    total_closed_acc += closed_acc
                    total_open_acc += open_acc
                    num_batches += 1

                    progress_bar.set_postfix(
                        {
                            "Loss": f"{loss.item():.4f}",
                            "Acc": f"{acc:.4f}",
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
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )

        project_root = DownloadUtils.project_root(__file__)
        results_dir = project_root / args.results_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        print("Loading datasets...")
        slake_data = load_slake()
        vqa_rad_data = load_vqa_rad()

        if args.subset:
            train_data = (
                slake_data["train"][: args.subset // 2]
                + vqa_rad_data["train"][: args.subset // 2]
            )
            val_data = (
                slake_data["validation"][: args.subset // 4]
                + vqa_rad_data.get("validation", [])[: args.subset // 4]
            )
        else:
            train_data = slake_data["train"] + vqa_rad_data["train"]
            val_data = slake_data["validation"] + vqa_rad_data.get("validation", [])

        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")

        image_processor = ImagePreprocessor(image_size=224)
        question_processor = QuestionPreprocessor()
        answer_processor = AnswerPreprocessor()

        train_questions = [item["question"] for item in train_data]
        train_answers = [item["answer"] for item in train_data]

        question_processor.build_vocab(train_questions)
        question_processor.compute_max_length(train_questions)
        answer_processor.build_vocab(train_answers)

        num_classes = answer_processor.get_vocab_size()
        print(f"Number of classes: {num_classes}")

        train_dataset = MedVQADataset(
            train_data, image_processor, question_processor, answer_processor
        )
        val_dataset = MedVQADataset(
            val_data, image_processor, question_processor, answer_processor
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
        )

        model = LRCN(
            num_classes=num_classes,
            hidden_dim=512,
            num_attention_layers=6,
            use_lrm=True,
        ).to(device)

        param_counts = model.count_parameters()
        print(f"Total parameters: {param_counts['total']:,}")
        print(f"Trainable parameters: {param_counts['trainable']:,}")

        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_closed_accuracy": [],
            "train_open_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_closed_accuracy": [],
            "val_open_accuracy": [],
        }

        best_val_acc = 0.0
        start_time = time.time()

        print(f"Starting training for {args.epochs} epochs...")

        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")

            train_metrics = train_epoch(
                model, train_loader, optimizer, criterion, device, scheduler
            )
            val_metrics = validate_epoch(model, val_loader, criterion, device)

            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])
            history["train_closed_accuracy"].append(train_metrics["closed_accuracy"])
            history["train_open_accuracy"].append(train_metrics["open_accuracy"])

            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])
            history["val_closed_accuracy"].append(val_metrics["closed_accuracy"])
            history["val_open_accuracy"].append(val_metrics["open_accuracy"])

            print(
                f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
                f"Closed: {train_metrics['closed_accuracy']:.4f}, Open: {train_metrics['open_accuracy']:.4f}"
            )
            print(
                f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                f"Closed: {val_metrics['closed_accuracy']:.4f}, Open: {val_metrics['open_accuracy']:.4f}"
            )

            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                if args.save_model:
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_accuracy": best_val_acc,
                            "history": history,
                            "config": {
                                "num_classes": num_classes,
                                "hidden_dim": 512,
                                "num_attention_layers": 6,
                                "use_lrm": True,
                            },
                        },
                        results_dir / "best_model.pth",
                    )
                    print(f"Best model saved! Val Acc: {best_val_acc:.4f}")

        total_time = time.time() - start_time

        final_results = {
            "best_val_accuracy": best_val_acc,
            "final_train_accuracy": train_metrics["accuracy"],
            "final_val_accuracy": val_metrics["accuracy"],
            "final_train_closed_acc": train_metrics["closed_accuracy"],
            "final_val_closed_acc": val_metrics["closed_accuracy"],
            "final_train_open_acc": train_metrics["open_accuracy"],
            "final_val_open_acc": val_metrics["open_accuracy"],
            "training_time_minutes": total_time / 60,
            "epochs_completed": args.epochs,
            "total_parameters": param_counts["total"],
            "trainable_parameters": param_counts["trainable"],
        }

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
