"""BioBERT text encoder for medical questions."""

import torch
import torch.nn as nn
import warnings
from transformers import AutoTokenizer, AutoModel
from transformers.utils import logging

from ..core.config import ModelConfig

# Suppress transformers warnings for cleaner output
logging.set_verbosity_warning()


class BioBERTTextEncoder(nn.Module):
    """BioBERT encoder for medical question understanding.

    Replaces GloVe + LSTM from original LRCN with BioBERT
    pre-trained on medical literature.
    """

    def __init__(
        self,
        model_name: str = ModelConfig.TEXT_ENCODER_NAME,
        hidden_dim: int = ModelConfig.HIDDEN_DIM,
        max_length: int = ModelConfig.MAX_TEXT_LENGTH,
        freeze_bert: bool = False,
        use_lstm: bool = False,
        lstm_layers: int = 2,
    ):
        super().__init__()

        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.max_length = max_length
        self.use_lstm = use_lstm

        # Load BioBERT tokenizer and model
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert = AutoModel.from_pretrained(model_name)

        print(f"✅ Loaded BioBERT ({model_name.split('/')[-1]}) pretrained weights")

        # Get BERT hidden dimension
        bert_hidden_dim = self.bert.config.hidden_size

        # Freeze BERT parameters if requested
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
            print(f"❄️ Frozen BioBERT backbone - only projection layers trainable")

        # Optional LSTM layer after BioBERT
        if use_lstm:
            self.lstm = nn.LSTM(
                input_size=bert_hidden_dim,
                hidden_size=hidden_dim // 2,  # Bidirectional LSTM
                num_layers=lstm_layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.1 if lstm_layers > 1 else 0,
            )
            lstm_output_dim = hidden_dim
        else:
            self.lstm = None
            lstm_output_dim = bert_hidden_dim

        # Projection layer to match LRCN hidden dimension
        if lstm_output_dim != hidden_dim:
            self.projection = nn.Linear(lstm_output_dim, hidden_dim)
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

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]

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

        # Optional LSTM processing
        if self.lstm is not None:
            # Pass sequence through LSTM
            lstm_output, _ = self.lstm(
                sequence_output
            )  # [batch_size, seq_len, hidden_dim]
            sequence_features = lstm_output
            # Use mean pooling for sentence-level representation
            mask_expanded = (
                attention_mask.unsqueeze(-1).expand(sequence_features.size()).float()
            )
            sum_embeddings = torch.sum(sequence_features * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled_features = sum_embeddings / sum_mask
        else:
            sequence_features = sequence_output
            pooled_features = pooled_output

        # Project to target dimension
        sequence_features = self.projection(sequence_features)
        pooled_features = self.projection(pooled_features)

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
