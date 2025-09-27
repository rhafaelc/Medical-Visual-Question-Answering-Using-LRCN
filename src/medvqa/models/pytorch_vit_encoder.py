"""Simplified PyTorch ViT Visual Encoder."""

import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_b_32, ViT_B_16_Weights, ViT_B_32_Weights

from ..core.config import ModelConfig


class PyTorchViTEncoder(nn.Module):
    """Clean PyTorch native ViT encoder for medical images.

    Much simpler than HuggingFace - no dependency hell!
    """

    def __init__(
        self,
        image_size: int = ModelConfig.IMAGE_SIZE,
        hidden_dim: int = ModelConfig.HIDDEN_DIM,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        variant: str = "b32",  # "b16" (patch 16) or "b32" (patch 32)
    ):
        super().__init__()

        self.image_size = image_size
        self.hidden_dim = hidden_dim
        self.variant = variant

        # Load PyTorch ViT - much cleaner!
        if variant == "b32":
            weights = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = vit_b_32(weights=weights)
            self.patch_size = 32
            print(
                f"✅ Loaded PyTorch ViT-B/32 ({'pretrained' if pretrained else 'random init'})"
            )
        else:  # b16
            weights = ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
            self.vit = vit_b_16(weights=weights)
            self.patch_size = 16
            print(
                f"✅ Loaded PyTorch ViT-B/16 ({'pretrained' if pretrained else 'random init'})"
            )

        # Remove classifier head - we want features, not ImageNet classes
        self.vit.heads = nn.Identity()

        # ViT outputs 768-dim features, project to target dimension
        vit_hidden_dim = 768
        if vit_hidden_dim != hidden_dim:
            self.projection = nn.Linear(vit_hidden_dim, hidden_dim)
        else:
            self.projection = nn.Identity()

        # Layer norm and dropout
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)

        # Freeze backbone if requested - MASSIVE speedup!
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False
            print(f"🧊 Frozen ViT backbone - only projection trainable")

        # Only unfreeze the final projection
        for param in self.projection.parameters():
            param.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from medical images.

        Args:
            images: [batch_size, 3, 224, 224]

        Returns:
            features: [batch_size, num_patches+1, hidden_dim]
                     (includes CLS token at position 0)
        """
        # Get features from ViT encoder (before final pooling)
        features = self._extract_patch_features(images)

        # Project to target dimension
        features = self.projection(features)

        # Layer norm and dropout
        features = self.layer_norm(features)
        features = self.dropout(features)

        return features

    def _extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch-level features from ViT encoder."""
        # This extracts features before the final global average pooling
        # Following torchvision ViT implementation

        # Patch embedding
        x = self.vit.conv_proj(
            x
        )  # [batch_size, hidden_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [batch_size, num_patches, hidden_dim]

        # Add class token
        batch_size = x.shape[0]
        cls_tokens = self.vit.class_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # [batch_size, num_patches+1, hidden_dim]

        # Add positional embedding
        x = x + self.vit.encoder.pos_embedding
        x = self.vit.encoder.dropout(x)

        # Pass through transformer layers
        x = self.vit.encoder.layers(x)
        x = self.vit.encoder.ln(x)

        return x

    def get_cls_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get global image features using [CLS] token."""
        features = self.forward(images)
        return features[:, 0]  # CLS token

    def get_patch_features(self, images: torch.Tensor) -> torch.Tensor:
        """Get patch-level features (excluding CLS token)."""
        features = self.forward(images)
        return features[:, 1:]  # Skip CLS token

    @property
    def num_patches(self) -> int:
        """Number of patches per image."""
        return (self.image_size // self.patch_size) ** 2


# Alias for backward compatibility
class ViTVisualEncoder(PyTorchViTEncoder):
    """Backward compatibility alias."""

    pass
