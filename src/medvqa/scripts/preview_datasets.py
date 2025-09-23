"""Preview and analyze Medical VQA datasets."""

import argparse
import json
from typing import List, Dict

from ..datamodules.slake_loader import load_slake
from ..datamodules.vqa_rad_loader import load_vqa_rad


class DatasetPreview:
    """Utility for previewing dataset contents."""

    @staticmethod
    def print_dataset_stats(dataset_name: str, data: List[Dict]) -> None:
        """Print statistics for a dataset."""
        if not data:
            print(f"[{dataset_name.upper()}] No data found")
            return

        # Count splits
        split_counts = {}
        for entry in data:
            split = entry.get("split", "unknown")
            split_counts[split] = split_counts.get(split, 0) + 1

        # Format split info
        split_info = ", ".join(
            f"{split}={count}" for split, count in sorted(split_counts.items())
        )

        print(f"[{dataset_name.upper()}] total={len(data)} splits={{{split_info}}}")

    @staticmethod
    def print_samples(
        dataset_name: str, data: List[Dict], num_samples: int, pretty: bool = False
    ) -> None:
        """Print sample entries from dataset."""
        if not data:
            return

        print(f"\n[{dataset_name.upper()}] First {num_samples} samples:")

        samples = data[:num_samples]
        for entry in samples:
            json_str = json.dumps(
                entry, ensure_ascii=False, indent=2 if pretty else None
            )
            print(json_str)


def main() -> int:
    """Main entry point for dataset preview."""
    parser = argparse.ArgumentParser(
        description="Preview standardized Medical VQA datasets"
    )
    parser.add_argument(
        "--dataset",
        choices=["slake", "vqa-rad", "all"],
        default="all",
        help="Dataset to preview",
    )
    parser.add_argument(
        "--head", type=int, default=3, help="Number of examples to show"
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )

    args = parser.parse_known_args()[0]

    datasets_to_load = []

    # Load requested datasets
    if args.dataset in ("slake", "all"):
        try:
            slake_data = load_slake()
            datasets_to_load.append(("SLAKE", slake_data))
        except Exception as e:
            print(f"[ERROR] Failed to load SLAKE: {e}")

    if args.dataset in ("vqa-rad", "all"):
        try:
            vqa_rad_data = load_vqa_rad()
            datasets_to_load.append(("VQA-RAD", vqa_rad_data))
        except Exception as e:
            print(f"[ERROR] Failed to load VQA-RAD: {e}")

    # Display results
    for dataset_name, data in datasets_to_load:
        DatasetPreview.print_dataset_stats(dataset_name, data)
        DatasetPreview.print_samples(dataset_name, data, args.head, args.pretty)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
