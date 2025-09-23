"""Download VQA-RAD dataset from OSF."""

from pathlib import Path

from ..core.base import BaseDatasetDownloader
from ..core.config import DatasetConfig
from ..core.download_utils import DownloadUtils


class VqaRadDownloader(BaseDatasetDownloader):
    """Downloader for VQA-RAD dataset from OSF."""

    @property
    def dataset_name(self) -> str:
        return DatasetConfig.VQA_RAD_DIR

    def download(self) -> int:
        """Download VQA-RAD dataset from OSF."""
        try:
            self._ensure_dir(self.temp_dir)

            # Download ZIP from OSF
            print("[OSF] Downloading VQA-RAD dataset...")
            content = DownloadUtils.download_file(
                DatasetConfig.VQA_RAD_ZIP_URL,
                None,  # Don't save to file, keep in memory
            )
            print(f"[OSF] Downloaded {len(content)/1e6:.1f} MB")

            # Extract files from ZIP buffer
            DownloadUtils.extract_zip_from_buffer(
                content,
                DatasetConfig.IMAGE_EXTENSIONS,
                DatasetConfig.VQA_RAD_KEEP_FILES,
                self.temp_dir,
            )

            # Organize files into final structure
            self._organize_files()

            # Cleanup
            self._cleanup_temp()

            # Print summary
            self._print_summary()

            return 0

        except Exception as e:
            print(f"[ERROR] Failed to download VQA-RAD: {e}")
            return 1

    def _organize_files(self) -> None:
        """Organize extracted files into final directory structure."""
        self._ensure_dir(self.images_dir)
        self._ensure_dir(self.annotations_dir)

        # Move all files from temp subdirectories
        for temp_path in self.temp_dir.rglob("*"):
            if not temp_path.is_file():
                continue

            filename_lower = temp_path.name.lower()

            # Determine target directory
            if filename_lower.endswith(DatasetConfig.IMAGE_EXTENSIONS):
                target_path = self.images_dir / temp_path.name
            else:
                target_path = self.annotations_dir / temp_path.name

            self._safe_move(temp_path, target_path)


def main() -> int:
    """Main entry point for VQA-RAD download."""
    downloader = VqaRadDownloader()
    return downloader.download()


if __name__ == "__main__":
    raise SystemExit(main())
