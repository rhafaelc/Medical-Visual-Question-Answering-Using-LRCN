"""Vision Transformer encoder for medical images."""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

from ..core.config import ModelConfig


class ViTVisualEncoder(nn.Module):
    """Vision Transformer encoder for medical image feature extraction.

    Replaces ResNeXt-152 from original LRCN with ViT backbone
    optimized for medical radiological images.
    """

    def __init__(
        self,
        image_size: int = ModelConfig.IMAGE_SIZE,
        patch_size: int = ModelConfig.PATCH_SIZE,
        hidden_dim: int = ModelConfig.HIDDEN_DIM,
        num_layers: int = 12,
        num_heads: int = ModelConfig.ATTENTION_HEADS,
        pretrained: bool = True,
    ):
        super().__init__()

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim

        if pretrained:
            # Use pre-trained ViT-Base
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
            vit_hidden_dim = self.vit.config.hidden_size
        else:
            # Create custom ViT configuration
            config = ViTConfig(
                image_size=image_size,
                patch_size=patch_size,
                hidden_size=hidden_dim,
                num_hidden_layers=num_layers,
                num_attention_heads=num_heads,
                num_channels=3,
            )
            self.vit = ViTModel(config)
            vit_hidden_dim = hidden_dim

        # Projection layer to match LRCN hidden dimension
        if vit_hidden_dim != hidden_dim:
            self.projection = nn.Linear(vit_hidden_dim, hidden_dim)
        else:
            self.projection = nn.Identity()

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract visual features from medical images.

        Args:
            images: Batch of images [batch_size, 3, H, W]

        Returns:
            Visual features [batch_size, num_patches + 1, hidden_dim]
            Note: +1 for [CLS] token
        """
        # Extract features using ViT
        outputs = self.vit(pixel_values=images)

        # Get sequence of patch embeddings (including [CLS] token)
        visual_features = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

        # Project to target dimension if needed
        visual_features = self.projection(visual_features)

        # Apply layer normalization and dropout
        visual_features = self.layer_norm(visual_features)
        visual_features = self.dropout(visual_features)

        return visual_features

    def get_cls_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract global image features using [CLS] token.

        Args:
            images: Batch of images [batch_size, 3, H, W]

        Returns:
            Global features [batch_size, hidden_dim]
        """
        visual_features = self.forward(images)
        return visual_features[:, 0]  # [CLS] token

    def get_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Extract patch-level features (excluding [CLS] token).

        Args:
            images: Batch of images [batch_size, 3, H, W]

        Returns:
            Patch features [batch_size, num_patches, hidden_dim]
        """
        visual_features = self.forward(images)
        return visual_features[:, 1:]  # Exclude [CLS] token

    @property
    def num_patches(self) -> int:
        """Number of image patches."""
        return (self.image_size // self.patch_size) ** 2
