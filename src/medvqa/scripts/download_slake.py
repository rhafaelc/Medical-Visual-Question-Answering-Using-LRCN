"""Download SLAKE dataset from HuggingFace."""

import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

from ..core.base import BaseDatasetDownloader
from ..core.config import DatasetConfig
from ..core.download_utils import DownloadUtils


class SlakeDownloader(BaseDatasetDownloader):
    """Downloader for SLAKE dataset from HuggingFace."""

    @property
    def dataset_name(self) -> str:
        return DatasetConfig.SLAKE_DIR

    def download(self) -> int:
        """Download SLAKE dataset from HuggingFace."""
        try:
            self._ensure_dir(self.temp_dir)

            # Download from HuggingFace
            repo_path = Path(
                snapshot_download(
                    repo_id=DatasetConfig.SLAKE_HF_REPO, repo_type="dataset"
                )
            )

            # Copy needed files to temp directory
            for file_path in repo_path.iterdir():
                if (
                    file_path.is_file()
                    and file_path.name in DatasetConfig.SLAKE_NEEDED_FILES
                ):
                    dest_path = self.temp_dir / file_path.name
                    if not dest_path.exists():
                        dest_path.write_bytes(file_path.read_bytes())

            # Extract images
            images_zip = self.temp_dir / "imgs.zip"
            if not images_zip.exists():
                raise FileNotFoundError("imgs.zip not found in HF snapshot")

            DownloadUtils.extract_zip_once(images_zip, self.images_dir)

            # Move annotation files
            self._ensure_dir(self.annotations_dir)
            for annotation_name in ("train.json", "validation.json", "test.json"):
                src_path = self.temp_dir / annotation_name
                if not src_path.exists():
                    raise FileNotFoundError(f"Missing {annotation_name}")

                if not DownloadUtils.verify_json(src_path):
                    raise ValueError(f"Invalid JSON in {annotation_name}")

                self._safe_move(src_path, self.annotations_dir / annotation_name)

            # Cleanup
            self._cleanup_temp()

            # Print summary
            self._print_summary()

            return 0

        except Exception as e:
            print(f"[ERROR] Failed to download SLAKE: {e}")
            return 1


def main() -> int:
    """Main entry point for SLAKE download."""
    downloader = SlakeDownloader()
    return downloader.download()


if __name__ == "__main__":
    raise SystemExit(main())
