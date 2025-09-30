"""LRCN attention mechanisms with Layer-Residual Mechanism."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from ..core.config import ModelConfig


class SelfAttentionBlock(nn.Module):
    """Self-attention block for LRCN."""

    def __init__(
        self,
        hidden_dim: int = ModelConfig.HIDDEN_DIM,
        num_heads: int = ModelConfig.ATTENTION_HEADS,
        feedforward_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 4

        # Multi-head self-attention (PyTorch native)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply self-attention block.

        Args:
            x: Input tensor [batch_size, seq_len, hidden_dim]
            attention_mask: Optional mask [batch_size, seq_len]

        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual connection
        key_padding_mask = (
            ~attention_mask.bool() if attention_mask is not None else None
        )
        attn_output, attention_weights = self.self_attention(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        x = self.norm1(x + attn_output)

        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)

        return x, attention_weights


class GuidedAttentionBlock(nn.Module):
    """Guided attention block for cross-modal interaction."""

    def __init__(
        self,
        hidden_dim: int = ModelConfig.HIDDEN_DIM,
        num_heads: int = ModelConfig.ATTENTION_HEADS,
        feedforward_dim: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 4

        # Cross-attention (visual attends to text) - PyTorch native
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
            bias=True,
        )

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, hidden_dim),
            nn.Dropout(dropout),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        visual_features: torch.Tensor,
        text_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply guided attention block.

        Args:
            visual_features: Visual features [batch_size, num_patches, hidden_dim]
            text_features: Text features [batch_size, seq_len, hidden_dim]
            text_mask: Optional text mask [batch_size, seq_len]

        Returns:
            Tuple of (output, attention_weights)
        """
        # Guided attention: visual queries attend to text keys/values
        key_padding_mask = ~text_mask.bool() if text_mask is not None else None
        attn_output, attention_weights = self.cross_attention(
            query=visual_features,
            key=text_features,
            value=text_features,
            key_padding_mask=key_padding_mask,
            need_weights=True,
        )
        visual_features = self.norm1(visual_features + attn_output)

        # Feed-forward with residual connection
        ffn_output = self.ffn(visual_features)
        visual_features = self.norm2(visual_features + ffn_output)

        return visual_features, attention_weights


class LayerResidualMechanism(nn.Module):
    """Layer-Residual Mechanism (LRM) - Core innovation of LRCN.

    Addresses information dispersion by maintaining residual connections
    between attention layers, preserving early-layer features in deeper layers.
    """

    def __init__(
        self,
        num_layers: int = 6,
        hidden_dim: int = ModelConfig.HIDDEN_DIM,
        num_heads: int = ModelConfig.ATTENTION_HEADS,
        feedforward_dim: int = None,
        dropout: float = 0.1,
        use_lrm: bool = True,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.use_lrm = use_lrm

        if feedforward_dim is None:
            feedforward_dim = hidden_dim * 4

        # Self-attention layers for text
        self.text_sa_layers = nn.ModuleList(
            [
                SelfAttentionBlock(hidden_dim, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Self-attention + Guided-attention layers for visual
        self.visual_sa_layers = nn.ModuleList(
            [
                SelfAttentionBlock(hidden_dim, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        self.visual_ga_layers = nn.ModuleList(
            [
                GuidedAttentionBlock(hidden_dim, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )

        # Layer normalization for LRM residuals
        if use_lrm:
            self.text_lrm_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
            )
            self.visual_lrm_norms = nn.ModuleList(
                [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
            )

    def forward(
        self,
        text_features: torch.Tensor,
        visual_features: torch.Tensor,
        text_mask: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Apply Layer-Residual Mechanism.

        Args:
            text_features: Text features [batch_size, seq_len, hidden_dim]
            visual_features: Visual features [batch_size, num_patches, hidden_dim]
            text_mask: Optional text mask [batch_size, seq_len]

        Returns:
            Dictionary containing:
            - 'text': Final text features
            - 'visual': Final visual features
            - 'attention_weights': Attention weights from each layer
        """
        # Store previous layer outputs for LRM
        prev_text_sa = None
        prev_visual_sa = None

        attention_weights = []

        for i in range(self.num_layers):
            # Text self-attention
            text_out, text_attn = self.text_sa_layers[i](text_features, text_mask)

            # Apply LRM for text (residual from previous SA layer)
            if self.use_lrm and prev_text_sa is not None:
                text_out = self.text_lrm_norms[i](text_out + prev_text_sa)

            prev_text_sa = text_out
            text_features = text_out

            # Visual self-attention
            visual_sa_out, visual_sa_attn = self.visual_sa_layers[i](visual_features)

            # Apply LRM for visual SA (residual from previous SA layer)
            if self.use_lrm and prev_visual_sa is not None:
                visual_sa_out = self.visual_lrm_norms[i](visual_sa_out + prev_visual_sa)

            prev_visual_sa = visual_sa_out

            # Visual guided-attention (attends to text)
            visual_out, visual_ga_attn = self.visual_ga_layers[i](
                visual_sa_out, text_features, text_mask
            )

            visual_features = visual_out

            # Store attention weights
            attention_weights.append(
                {
                    "text_sa": text_attn,
                    "visual_sa": visual_sa_attn,
                    "visual_ga": visual_ga_attn,
                }
            )

        return {
            "text": text_features,
            "visual": visual_features,
            "attention_weights": attention_weights,
        }
