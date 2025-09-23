"""Download all datasets for Medical VQA LRCN project."""

import argparse

from .download_slake import main as download_slake_main
from .download_vqa_rad import main as download_vqa_rad_main


def main() -> int:
    """Download all available datasets."""
    parser = argparse.ArgumentParser(
        description="Download all Medical VQA datasets (SLAKE and VQA-RAD)"
    )
    parser.parse_args()

    print("[INFO] Starting download of all datasets...")

    # Download VQA-RAD
    print("\n[1/2] Downloading VQA-RAD...")
    vqa_rad_result = download_vqa_rad_main()

    # Download SLAKE
    print("\n[2/2] Downloading SLAKE...")
    slake_result = download_slake_main()

    # Return combined result
    if vqa_rad_result == 0 and slake_result == 0:
        print("\n[SUCCESS] All datasets downloaded successfully!")
        return 0
    else:
        print("\n[ERROR] Some datasets failed to download.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
