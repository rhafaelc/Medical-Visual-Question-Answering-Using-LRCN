"""Download utilities for Medical VQA datasets."""

import io
import json
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable, Tuple

import requests
from tqdm import tqdm

from ..core.config import DownloadConfig, DatasetConfig


class DownloadUtils:
    """Utilities for dataset downloading operations."""

    @staticmethod
    def project_root(file_path: str) -> Path:
        """Get project root directory from file path."""
        return Path(file_path).resolve().parents[3]

    @staticmethod
    def extract_zip_once(
        zip_path: Path, output_dir: Path, marker: str = ".extracted"
    ) -> Path:
        """Extract ZIP file only if not already extracted."""
        output_dir.mkdir(parents=True, exist_ok=True)
        marker_file = output_dir / marker

        if marker_file.exists():
            return output_dir

        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(output_dir)

        marker_file.touch()
        return output_dir

    @staticmethod
    def verify_json(path: Path) -> bool:
        """Verify JSON file integrity."""
        try:
            json.loads(path.read_text(encoding="utf-8"))
            return True
        except (json.JSONDecodeError, FileNotFoundError):
            return False

    @staticmethod
    def download_file(
        url: str, output_path: Path, timeout: int = DownloadConfig.REQUEST_TIMEOUT
    ) -> bytes:
        """Download file from URL."""
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        content = response.content
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(content)

        return content

    @staticmethod
    def extract_zip_from_buffer(
        buffer: bytes, file_extensions: Tuple[str, ...], keep_files: set, temp_dir: Path
    ) -> Tuple[Path, Path]:
        """Extract ZIP from buffer, organizing by file type."""
        images_dir = temp_dir / "images_raw"
        annotations_dir = temp_dir / "annotations_raw"

        images_dir.mkdir(parents=True, exist_ok=True)
        annotations_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(io.BytesIO(buffer)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue

                filename = Path(info.filename).name
                filename_lower = filename.lower()

                # Determine target directory
                if filename_lower.endswith(file_extensions):
                    target_dir = images_dir
                elif (
                    any(
                        filename_lower.endswith(ext)
                        for ext in DatasetConfig.ANNOTATION_EXTENSIONS
                    )
                    or filename in keep_files
                ):
                    target_dir = annotations_dir
                else:
                    continue

                # Extract file
                output_path = target_dir / filename
                with zf.open(info, "r") as src, output_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)

        return images_dir, annotations_dir

    @staticmethod
    def osf_download_many(
        jobs: Iterable[Tuple],
        max_workers: int = DownloadConfig.MAX_WORKERS,
        retries: int = DownloadConfig.RETRY_ATTEMPTS,
        backoff: float = DownloadConfig.BACKOFF_FACTOR,
        desc: str = "Downloading files",
    ):
        """Download multiple files concurrently from OSF."""
        jobs_list = list(jobs)

        def _download_with_retry(file_obj, dest_path):
            dest_path = Path(dest_path)
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            for attempt in range(retries):
                try:
                    with dest_path.open("wb") as fp:
                        file_obj.write_to(fp)
                    if dest_path.stat().st_size > 0:
                        return dest_path
                except Exception:
                    time.sleep(backoff**attempt)

            return dest_path  # May be empty if all retries failed

        with (
            ThreadPoolExecutor(max_workers=max_workers) as executor,
            tqdm(
                total=len(jobs_list),
                unit=DownloadConfig.PROGRESS_BAR_UNIT,
                desc=desc,
                dynamic_ncols=True,
            ) as progress_bar,
        ):
            futures = [
                executor.submit(_download_with_retry, file_obj, dest_path)
                for file_obj, dest_path in jobs_list
            ]

            for _ in as_completed(futures):
                progress_bar.update(1)
