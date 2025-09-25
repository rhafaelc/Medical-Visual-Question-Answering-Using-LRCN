"""Image preprocessing for Medical VQA LRCN."""

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageEnhance
import numpy as np
from typing import Union, Tuple

from ..core.config import ModelConfig


class ImagePreprocessor:
    """Medical image preprocessing: resize, contrast stretching, normalization."""

    def __init__(
        self,
        image_size: int = ModelConfig.IMAGE_SIZE,
        apply_contrast_stretching: bool = True,
        normalize_range: Tuple[float, float] = ModelConfig.NORMALIZE_RANGE,
    ):
        """Initialize image preprocessor.

        Args:
            image_size: Target image size for ViT
            apply_contrast_stretching: Apply contrast enhancement
            normalize_range: Normalization range (0,1)
        """
        self.image_size = image_size
        self.apply_contrast_stretching = apply_contrast_stretching
        self.normalize_range = normalize_range

        self.transforms = transforms.Compose(
            [transforms.Resize((image_size, image_size)), transforms.ToTensor()]
        )

    def apply_contrast_stretching(self, image: Image.Image) -> Image.Image:
        """Apply min-max contrast stretching to enhance medical image visibility."""
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)

        if len(img_array.shape) == 3:  # RGB image
            # Apply min-max stretching to each channel
            stretched_array = np.zeros_like(img_array)
            for channel in range(img_array.shape[2]):
                channel_data = img_array[:, :, channel]
                min_val = np.min(channel_data)
                max_val = np.max(channel_data)

                # Avoid division by zero
                if max_val > min_val:
                    stretched_array[:, :, channel] = (
                        (channel_data - min_val) / (max_val - min_val)
                    ) * 255
                else:
                    stretched_array[:, :, channel] = channel_data
        else:  # Grayscale image
            min_val = np.min(img_array)
            max_val = np.max(img_array)

            # Min-max normalization to [0, 255] range
            if max_val > min_val:
                stretched_array = ((img_array - min_val) / (max_val - min_val)) * 255
            else:
                stretched_array = img_array

        # Convert back to uint8 and PIL Image
        stretched_array = np.clip(stretched_array, 0, 255).astype(np.uint8)
        return Image.fromarray(stretched_array)

    def preprocess(self, image: Union[Image.Image, str, np.ndarray]) -> torch.Tensor:
        """Preprocess medical image: resize, contrast stretch, normalize to [0,1]."""
        # Load image if path provided
        if isinstance(image, str):
            try:
                image = Image.open(image).convert("RGB")
            except Exception as e:
                raise ValueError(f"Failed to load image from {image}: {e}")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")

        # Ensure RGB format
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Apply contrast stretching for medical images
        if self.apply_contrast_stretching:
            image = self.apply_contrast_stretching(image)

        # Apply transforms (resize only, no augmentation)
        tensor = self.transforms(image)

        # Normalize to [0,1] range
        tensor = tensor.clamp(0, 1)

        return tensor

    def get_transforms(self) -> transforms.Compose:
        """Get torchvision transforms for DataLoader integration."""
        return transforms.Compose(
            [
                transforms.Lambda(
                    lambda img: (
                        self.apply_contrast_stretching(img)
                        if self.apply_contrast_stretching
                        else img
                    )
                ),
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )


class MedicalImageNormalizer:
    """Specialized normalizer for medical images."""

    @staticmethod
    def imagenet_normalize():
        """ImageNet normalization for pretrained models."""
        return transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    @staticmethod
    def medical_normalize():
        """Medical-specific normalization to [0,1] range."""
        return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    @staticmethod
    def zero_one_normalize():
        """Explicit [0,1] normalization."""
        return lambda x: x.clamp(0, 1)
