# Medical Visual Question Answering using LRCN

An adaptation of Layer-Residual Co-Attention Network (LRCN) for Medical Visual Question Answering (MedVQA). This research addresses information dispersion in deep co-attention VQA models by integrating ViT backbone and BioBERT text encoding while preserving the core co-attention and Layer-Residual Mechanism (LRM).

## Features

- **Modern Architecture**: ViT backbone (replaces ResNeXt-152) + BioBERT (replaces GloVe)
- **Layer-Residual Mechanism**: Addresses information dispersion in deep attention networks
- **Dual Dataset Support**: SLAKE and VQA-RAD benchmark datasets
- **Standardized Pipeline**: Consistent data loading and preprocessing
- **CLI Tools**: Easy dataset download and exploration

## Quick Start

### Environment Setup

This project uses Nix for reproducible development environments:

```bash
# Enter development environment
nix develop

# Environment auto-activates Python virtual environment
```

### Dataset Download

```bash
# Download all datasets
download-all-datasets

# Or download individually
download-vqa-rad    # Downloads VQA-RAD from OSF
download-slake      # Downloads SLAKE from HuggingFace
```

### Data Exploration

```bash
# Preview datasets
preview-datasets --dataset all --head 5 --pretty
preview-datasets --dataset slake --head 10
```

## Project Structure

```
src/medvqa/
├── core/                    # Core utilities and base classes
│   ├── config.py           # Centralized configuration
│   ├── base.py             # Abstract base classes
│   └── download_utils.py   # Download utilities
├── datamodules/            # Dataset loaders
│   ├── common.py           # Unified loading interface
│   ├── slake_loader.py     # SLAKE dataset loader
│   └── vqa_rad_loader.py   # VQA-RAD dataset loader
└── scripts/                # CLI commands
    ├── download_all.py     # Download all datasets
    ├── download_slake.py   # Download SLAKE
    ├── download_vqa_rad.py # Download VQA-RAD
    └── preview_datasets.py # Dataset preview tool
```

## Datasets

### SLAKE (English subset)

- **Images**: 642 radiological images
- **QA Pairs**: 7,033 question-answer pairs
- **Split**: 70% train, 15% validation, 15% test (official)

### VQA-RAD

- **Images**: 315 unique radiological images
- **QA Pairs**: 2,248 question-answer pairs
- **Split**: 72% train, 8% validation (stratified), 20% test

## Data Format

All loaders return standardized entries:

```python
{
    "id": "dataset_split_00001",
    "dataset": "slake_all" | "vqa-rad",
    "split": "train" | "validation" | "test",
    "image": "/absolute/path/to/image.jpg",
    "question": "What is shown in the image?",
    "answer": "normalized lowercase answer",
    "answer_type": "open" | "closed"  # auto-detected
}
```

## Research Focus

### Layer-Residual Co-Attention Network (LRCN)

- **Problem**: Information dispersion in deep attention networks
- **Solution**: Layer-Residual Mechanism (LRM) with inter-layer residual connections
- **Architecture**: Encoder-decoder variant with Self-Attention and Guided-Attention blocks

### Key Innovations

- **Information Preservation**: LRM maintains early-layer features in deeper layers
- **Medical Domain Adaptation**: BioBERT for medical terminology, ViT for radiological images
- **Ablation Study**: Evaluate LRM effectiveness across different attention layer depths

## Development

### Adding New Datasets

1. Create loader class inheriting from `BaseDatasetLoader`
2. Implement in `src/medvqa/datamodules/{name}_loader.py`
3. Follow standardized return format
4. Add CLI download command following `download-{name}` pattern

### Configuration

All constants centralized in `src/medvqa/core/config.py`:

- `DatasetConfig`: Dataset-specific settings
- `ModelConfig`: Architecture parameters
- `DownloadConfig`: Download operation settings

## Requirements

- Python 3.12+
- Nix (for development environment)
- See `pyproject.toml` for Python dependencies

## CLI Commands

- `download-all-datasets`: Download both datasets
- `download-vqa-rad`: Download VQA-RAD from OSF
- `download-slake`: Download SLAKE from HuggingFace
- `preview-datasets`: Analyze and preview datasets
