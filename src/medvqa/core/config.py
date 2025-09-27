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

    IMAGE_SIZE = 224
    PATCH_SIZE = 16
    NORMALIZE_RANGE = (0, 1)
    ANSWER_TYPE_CLOSED_KEYWORDS = {"yes", "no"}
    COVERAGE_PERCENTILE = 95
    HIDDEN_DIM = 512
    ATTENTION_HEADS = 8
    RANDOM_SEED = 42
    VQA_RAD_VAL_SPLIT_RATIO = 0.1  # VQA-RAD validation split ratio from training data

    # Image preprocessing constants
    PIXEL_MAX_VALUE = 255
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    MEDICAL_NORMALIZE_MEAN = [0.5, 0.5, 0.5]
    MEDICAL_NORMALIZE_STD = [0.5, 0.5, 0.5]

    # Text preprocessing constants
    MAX_TEXT_LENGTH = 128
    PERCENTAGE_DIVISOR = 100.0
    TOP_ANSWERS_PREVIEW = 10
    STATISTICS_TOP_ANSWERS = 20

    # Weight constant
    DEFAULT_WEIGHT = 1.0

    # Training constants
    DEFAULT_EPOCHS = 200  # Updated for research configurations
    DEFAULT_BATCH_SIZE = 64  # Default for dataloaders
    DEFAULT_TRAINING_BATCH_SIZE = 64  # Updated for research configurations
    DEFAULT_LEARNING_RATE = 1e-4
    DEFAULT_NUM_WORKERS = 4
    DEFAULT_WEIGHT_DECAY = 1e-5
    DEFAULT_ATTENTION_LAYERS = 8
    GRADIENT_CLIP_NORM = 1.0

    # Architecture flags
    USE_LRM = True
    VISUAL_FEATURE_DIM = 768  # ViT base feature dimension
    TEXT_FEATURE_DIM = 768  # BioBERT base feature dimension
    TEXT_ENCODER_NAME = "dmis-lab/biobert-base-cased-v1.1"

    # Overfitting detection thresholds
    HIGH_OVERFITTING_THRESHOLD = 0.1
    MODERATE_OVERFITTING_THRESHOLD = 0.05

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
