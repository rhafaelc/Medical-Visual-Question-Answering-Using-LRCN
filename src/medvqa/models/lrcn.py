"""Main LRCN model for Medical Visual Question Answering."""

import torch
import torch.nn as nn
from typing import Optional, Union

from ..core.config import ModelConfig
from .visual_encoder import ViTVisualEncoder
from .text_encoder import BioBERTTextEncoder
from .attention import LayerResidualMechanism


class LRCN(nn.Module):
    """Layer-Residual Co-Attention Network for Medical VQA.

    Implementation of Han et al.'s LRCN adapted for medical domain:
    - ViT backbone (replaces ResNeXt-152) for medical image encoding
    - BioBERT (replaces GloVe) for medical question encoding
    - Layer-Residual Mechanism to address information dispersion
    - Co-attention with Self-Attention and Guided-Attention blocks
    """

    def __init__(
        self,
        # Model architecture
        hidden_dim: int = ModelConfig.HIDDEN_DIM,
        num_attention_layers: int = ModelConfig.DEFAULT_ATTENTION_LAYERS,
        num_heads: int = ModelConfig.ATTENTION_HEADS,
        feedforward_dim: int = None,
        # Visual encoder
        image_size: int = ModelConfig.IMAGE_SIZE,
        patch_size: int = ModelConfig.PATCH_SIZE,
        vit_pretrained: bool = True,
        visual_encoder_type: str = "vit",  # "vit" or "vit-linear"
        # Text encoder
        biobert_model: str = ModelConfig.TEXT_ENCODER_NAME,
        max_text_length: int = ModelConfig.MAX_TEXT_LENGTH,
        freeze_biobert: bool = False,
        text_encoder_type: str = "biobert",  # "biobert" or "biobert-lstm"
        # Answer vocabulary
        num_classes: int = 1000,  # Will be set based on answer vocabulary
        # Layer-Residual Mechanism
        use_lrm: bool = ModelConfig.USE_LRM,
        # Efficiency settings
        freeze_backbones: bool = True,  # Freeze ViT + BioBERT for faster training
        # Regularization
        dropout: float = 0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_attention_layers = num_attention_layers
        self.num_classes = num_classes
        self.use_lrm = use_lrm
        self.visual_encoder_type = visual_encoder_type
        self.text_encoder_type = text_encoder_type

        # Visual encoder (ViT-B/32 with linear projection to 512)
        if visual_encoder_type == "vit-linear":
            self.visual_encoder = ViTVisualEncoder(
                image_size=image_size,
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                pretrained=vit_pretrained,
                use_linear_projection=True,
                freeze_backbone=freeze_backbones,
                variant="b32",  # Use ViT-B/32
            )
        else:  # vit
            self.visual_encoder = ViTVisualEncoder(
                image_size=image_size,
                patch_size=patch_size,
                hidden_dim=hidden_dim,
                pretrained=vit_pretrained,
                freeze_backbone=freeze_backbones,
                variant="b32",  # Use ViT-B/32
            )

        # Text encoder (BioBERT + LSTM → 512 dimensions)
        self.text_encoder = BioBERTTextEncoder(
            model_name=biobert_model,
            hidden_dim=hidden_dim,
            max_length=max_text_length,
            freeze_bert=freeze_backbones,
            use_lstm=True,  # Always use LSTM for 512-dim output
        )

        # Layer-Residual Co-Attention Mechanism
        self.lrcn_attention = LayerResidualMechanism(
            num_layers=num_attention_layers,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            feedforward_dim=feedforward_dim,
            dropout=dropout,
            use_lrm=use_lrm,
        )

        # Answer decoder
        self.answer_decoder = nn.Sequential(
            # Global pooling of visual features
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            # Classification head
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        images: torch.Tensor,
        questions: Union[list[str], torch.Tensor],
        return_attention: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass of LRCN model.

        Args:
            images: Batch of images [batch_size, 3, H, W]
            questions: List of question strings or pre-tokenized tensor
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
            - 'logits': Answer logits [batch_size, num_classes]
            - 'attention_weights': Attention weights (if requested)
        """
        batch_size = images.shape[0]

        # Encode visual features
        visual_features = self.visual_encoder(
            images
        )  # [batch_size, num_patches+1, hidden_dim]
        visual_features = visual_features[:, 1:]  # Remove [CLS] token, keep patches

        # Encode text features
        if isinstance(questions, list):
            # String input - use BioBERT directly
            text_outputs = self.text_encoder(questions)
            text_features = text_outputs[
                "sequence"
            ]  # [batch_size, seq_len, hidden_dim]
            text_mask = text_outputs["attention_mask"]  # [batch_size, seq_len]
        else:
            # Assume pre-processed tensor input
            text_features = questions
            text_mask = None

        # Apply Layer-Residual Co-Attention
        lrcn_outputs = self.lrcn_attention(
            text_features=text_features,
            visual_features=visual_features,
            text_mask=text_mask,
        )

        # Get final visual features after co-attention
        final_visual_features = lrcn_outputs[
            "visual"
        ]  # [batch_size, num_patches, hidden_dim]

        # Decode answer
        # Transpose for AdaptiveAvgPool1d: [batch_size, hidden_dim, num_patches]
        pooled_features = final_visual_features.transpose(1, 2)
        answer_logits = self.answer_decoder(
            pooled_features
        )  # [batch_size, num_classes]

        outputs = {
            "logits": answer_logits,
        }

        if return_attention:
            outputs["attention_weights"] = lrcn_outputs["attention_weights"]
            outputs["visual_features"] = final_visual_features
            outputs["text_features"] = lrcn_outputs["text"]

        return outputs

    def predict(
        self,
        images: torch.Tensor,
        questions: Union[list[str], torch.Tensor],
        answer_vocab: Optional[dict] = None,
    ) -> dict[str, torch.Tensor]:
        """Generate predictions with post-processing.

        Args:
            images: Batch of images [batch_size, 3, H, W]
            questions: List of question strings or pre-tokenized tensor
            answer_vocab: Optional answer vocabulary for decoding

        Returns:
            Dictionary containing predictions and confidence scores
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, questions)

            logits = outputs["logits"]
            probabilities = torch.softmax(logits, dim=-1)
            predictions = torch.argmax(logits, dim=-1)
            confidence = torch.max(probabilities, dim=-1)[0]

            results = {
                "predictions": predictions,
                "probabilities": probabilities,
                "confidence": confidence,
                "logits": logits,
            }

            # Decode predictions if vocabulary is provided
            if answer_vocab is not None:
                idx_to_answer = {v: k for k, v in answer_vocab.items()}
                decoded_answers = [
                    idx_to_answer.get(pred.item(), "<UNK>") for pred in predictions
                ]
                results["decoded_answers"] = decoded_answers

            return results

    def get_attention_maps(
        self,
        images: torch.Tensor,
        questions: Union[list[str], torch.Tensor],
        layer_idx: int = -1,
    ) -> dict[str, torch.Tensor]:
        """Extract attention maps for visualization.

        Args:
            images: Batch of images [batch_size, 3, H, W]
            questions: List of question strings or pre-tokenized tensor
            layer_idx: Which layer to extract attention from (-1 for last layer)

        Returns:
            Dictionary containing attention maps
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, questions, return_attention=True)

            attention_weights = outputs["attention_weights"][layer_idx]

            return {
                "visual_self_attention": attention_weights["visual_sa"],
                "visual_guided_attention": attention_weights["visual_ga"],
                "text_self_attention": attention_weights["text_sa"],
            }

    def count_parameters(self) -> dict[str, int]:
        """Count model parameters."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        # Count parameters by component
        visual_params = sum(p.numel() for p in self.visual_encoder.parameters())
        text_params = sum(p.numel() for p in self.text_encoder.parameters())
        attention_params = sum(p.numel() for p in self.lrcn_attention.parameters())
        decoder_params = sum(p.numel() for p in self.answer_decoder.parameters())

        return {
            "total": total_params,
            "trainable": trainable_params,
            "visual_encoder": visual_params,
            "text_encoder": text_params,
            "attention": attention_params,
            "decoder": decoder_params,
        }
