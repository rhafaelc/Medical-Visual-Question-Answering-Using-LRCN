"""BioBERT text encoder for medical questions."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from ..core.config import ModelConfig


class BioBERTTextEncoder(nn.Module):
    """BioBERT encoder for medical question understanding.

    Replaces GloVe + LSTM from original LRCN with BioBERT
    pre-trained on medical literature.
    """

    def __init__(
        self,
        model_name: str = "dmis-lab/biobert-base-cased-v1.1",
        hidden_dim: int = ModelConfig.HIDDEN_DIM,
        max_length: int = ModelConfig.MAX_TEXT_LENGTH,
        freeze_bert: bool = False,
    ):
        super().__init__()

        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.max_length = max_length

        # Load BioBERT tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)

        # Get BERT hidden dimension
        bert_hidden_dim = self.bert.config.hidden_size

        # Freeze BERT parameters if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        # Projection layer to match LRCN hidden dimension
        if bert_hidden_dim != hidden_dim:
            self.projection = nn.Linear(bert_hidden_dim, hidden_dim)
        else:
            self.projection = nn.Identity()

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.1)

    def forward(self, questions: list[str]) -> dict[str, torch.Tensor]:
        """Encode medical questions using BioBERT.

        Args:
            questions: List of question strings

        Returns:
            Dictionary containing:
            - 'sequence': Token-level features [batch_size, seq_len, hidden_dim]
            - 'pooled': Sentence-level features [batch_size, hidden_dim]
            - 'attention_mask': Attention mask [batch_size, seq_len]
        """
        # Tokenize questions
        encoding = self.tokenizer(
            questions,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move to device
        input_ids = encoding["input_ids"].to(next(self.parameters()).device)
        attention_mask = encoding["attention_mask"].to(next(self.parameters()).device)

        # Encode with BioBERT
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        # Get sequence and pooled outputs
        sequence_output = (
            outputs.last_hidden_state
        )  # [batch_size, seq_len, bert_hidden]
        pooled_output = outputs.pooler_output  # [batch_size, bert_hidden]

        # Project to target dimension
        sequence_features = self.projection(sequence_output)
        pooled_features = self.projection(pooled_output)

        # Apply layer normalization and dropout
        sequence_features = self.layer_norm(sequence_features)
        sequence_features = self.dropout(sequence_features)

        pooled_features = self.layer_norm(pooled_features)
        pooled_features = self.dropout(pooled_features)

        return {
            "sequence": sequence_features,  # Token-level features
            "pooled": pooled_features,  # Sentence-level features
            "attention_mask": attention_mask,  # For masking in attention
        }

    def encode_questions(self, questions: list[str]) -> torch.Tensor:
        """Simple interface to get question embeddings.

        Args:
            questions: List of question strings

        Returns:
            Question embeddings [batch_size, hidden_dim]
        """
        outputs = self.forward(questions)
        return outputs["pooled"]

    def get_sequence_features(
        self, questions: list[str]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get token-level features and attention mask.

        Args:
            questions: List of question strings

        Returns:
            Tuple of (features, attention_mask)
            - features: [batch_size, seq_len, hidden_dim]
            - attention_mask: [batch_size, seq_len]
        """
        outputs = self.forward(questions)
        return outputs["sequence"], outputs["attention_mask"]
