"""Configuration constants for Medical VQA LRCN project."""

from pathlib import Path


# Dataset configuration
class DatasetConfig:
    """Configuration constants for datasets."""

    # Common settings
    DEFAULT_ROOT = Path("data/raw")
    TEMP_DIR_NAME = "_tmp"
    IMAGES_DIR_NAME = "images"
    ANNOTATIONS_DIR_NAME = "annotations"

    # VQA-RAD specific
    VQA_RAD_DIR = "vqa-rad"
    VQA_RAD_ZIP_URL = (
        "https://files.osf.io/v1/resources/89kps/providers/osfstorage/?zip="
    )
    VQA_RAD_PUBLIC_JSON = "VQA_RAD Dataset Public.json"
    VQA_RAD_KEEP_FILES = {
        "VQA_RAD Dataset Public.json",
        "VQA_RAD Dataset Public.xlsx",
        "VQA_RAD Dataset Public.xml",
        "Readme.docx",
    }

    # SLAKE specific
    SLAKE_DIR = "slake_all"
    SLAKE_HF_REPO = "BoKelvin/SLAKE"
    SLAKE_NEEDED_FILES = {"train.json", "validation.json", "test.json", "imgs.zip"}

    # File extensions
    IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    ANNOTATION_EXTENSIONS = (
        ".json",
        ".csv",
        ".tsv",
        ".xlsx",
        ".xls",
        ".xml",
        ".docx",
        ".txt",
    )


class ModelConfig:
    """Configuration constants for model architecture."""

    # Image preprocessing
    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    NORMALIZE_RANGE = (0, 1)

    # Text preprocessing
    ANSWER_TYPE_CLOSED_KEYWORDS = {"yes", "no"}
    COVERAGE_PERCENTILE = 95  # For Lmax and top-K selection

    # Model architecture
    HIDDEN_DIM = 512
    ATTENTION_HEADS = 8

    # Training
    DEFAULT_SPLITS = {
        "vqa_rad": {"train": 0.72, "validation": 0.08, "test": 0.20},
        "slake": {"train": 0.70, "validation": 0.15, "test": 0.15},
    }


class DownloadConfig:
    """Configuration for download operations."""

    MAX_WORKERS = 16
    RETRY_ATTEMPTS = 3
    BACKOFF_FACTOR = 1.5
    REQUEST_TIMEOUT = 300
    PROGRESS_BAR_UNIT = "file"
