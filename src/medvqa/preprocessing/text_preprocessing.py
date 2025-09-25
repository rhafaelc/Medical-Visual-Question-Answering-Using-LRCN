"""Text preprocessing for Medical VQA LRCN."""

import string
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch

from ..core.config import ModelConfig


class QuestionPreprocessor:
    """Question preprocessing with tokenization and padding."""

    def __init__(
        self,
        max_length: Optional[int] = None,
        coverage_percentile: int = ModelConfig.COVERAGE_PERCENTILE,
    ):
        """Initialize question preprocessor.

        Args:
            max_length: Maximum question length. If None, computed from data.
            coverage_percentile: Percentile for automatic max_length calculation.
        """
        self.max_length = max_length
        self.coverage_percentile = coverage_percentile
        self.vocab = None
        self.word_to_idx = None
        self.idx_to_word = None

        # Special tokens
        self.PAD_TOKEN = "<PAD>"
        self.UNK_TOKEN = "<UNK>"

    def build_vocab(self, questions: List[str]) -> Dict[str, int]:
        """Build vocabulary from questions.

        Args:
            questions: List of question strings

        Returns:
            word_to_idx mapping
        """
        # Tokenize all questions
        all_tokens = []
        for question in questions:
            tokens = self.tokenize(question)
            all_tokens.extend(tokens)

        # Count token frequencies
        token_counts = Counter(all_tokens)

        # Create vocabulary (keep all tokens for questions)
        special_tokens = [self.PAD_TOKEN, self.UNK_TOKEN]
        vocab_tokens = special_tokens + list(token_counts.keys())

        self.vocab = vocab_tokens
        self.word_to_idx = {token: idx for idx, token in enumerate(vocab_tokens)}
        self.idx_to_word = {idx: token for token, idx in self.word_to_idx.items()}

        return self.word_to_idx

    def compute_max_length(self, questions: List[str]) -> int:
        """Compute max_length based on percentile of question lengths."""
        lengths = []
        for question in questions:
            tokens = self.tokenize(question)
            lengths.append(len(tokens))

        lmax = int(np.percentile(lengths, self.coverage_percentile))
        print(f"Computed max_length at {self.coverage_percentile}th percentile: {lmax}")
        print(
            f"Question length stats: min={min(lengths)}, max={max(lengths)}, "
            f"mean={np.mean(lengths):.1f}, std={np.std(lengths):.1f}"
        )

        self.max_length = lmax
        return lmax

    def tokenize(self, question: str) -> List[str]:
        """Tokenize question: lowercase, remove punctuation, split."""
        question = question.lower().strip()
        question = question.translate(str.maketrans("", "", string.punctuation))
        question = re.sub(r"\s+", " ", question)
        tokens = question.split()

        return tokens

    def encode(self, question: str) -> List[int]:
        """Encode question to token indices.

        Args:
            question: Raw question string

        Returns:
            List of token indices
        """
        if self.word_to_idx is None:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        tokens = self.tokenize(question)

        # Convert to indices
        indices = []
        for token in tokens:
            if token in self.word_to_idx:
                indices.append(self.word_to_idx[token])
            else:
                indices.append(self.word_to_idx[self.UNK_TOKEN])

        return indices

    def pad_or_truncate(
        self, indices: List[int], max_length: Optional[int] = None
    ) -> List[int]:
        """Pad or truncate to maximum length.

        Args:
            indices: List of token indices
            max_length: Maximum length to pad/truncate to

        Returns:
            Padded/truncated indices
        """
        if max_length is None:
            max_length = self.max_length

        if max_length is None:
            raise ValueError("max_length not specified and not computed from data")

        pad_idx = self.word_to_idx[self.PAD_TOKEN]

        if len(indices) >= max_length:
            # Truncate
            return indices[:max_length]
        else:
            # Pad
            return indices + [pad_idx] * (max_length - len(indices))

    def preprocess(self, question: str) -> torch.Tensor:
        """Complete preprocessing pipeline for question."""
        indices = self.encode(question)
        padded_indices = self.pad_or_truncate(indices)
        return torch.tensor(padded_indices, dtype=torch.long)

    def decode(self, indices: List[int]) -> str:
        """Decode token indices back to text.

        Args:
            indices: List of token indices

        Returns:
            Decoded text
        """
        if self.idx_to_word is None:
            raise ValueError("Vocabulary not built. Call build_vocab() first.")

        tokens = []
        for idx in indices:
            if idx in self.idx_to_word:
                token = self.idx_to_word[idx]
                if token != self.PAD_TOKEN:
                    tokens.append(token)

        return " ".join(tokens)


class AnswerPreprocessor:
    """Answer preprocessing with top-K vocabulary and label encoding."""

    def __init__(
        self,
        coverage_percentile: int = ModelConfig.COVERAGE_PERCENTILE,
        top_k: Optional[int] = None,
    ):
        """Initialize answer preprocessor.

        Args:
            coverage_percentile: Coverage percentile for top-K selection.
            top_k: Fixed number of top answers. If None, computed from coverage.
        """
        self.coverage_percentile = coverage_percentile
        self.top_k = top_k
        self.answer_vocab = None
        self.answer_to_idx = None
        self.idx_to_answer = None
        self.answer_counts = None

        # Special answer for out-of-vocabulary
        self.OTHER_TOKEN = "<OTHER>"

    def build_vocab(self, answers: List[str]) -> Dict[str, int]:
        """Build answer vocabulary with top-K selection.

        Args:
            answers: List of answer strings

        Returns:
            answer_to_idx mapping
        """
        # Step 1: Normalize answers (lowercase)
        normalized_answers = [answer.lower().strip() for answer in answers]

        # Step 2: Count answer frequencies
        self.answer_counts = Counter(normalized_answers)
        total_answers = len(normalized_answers)

        # Step 3: Determine top-K for coverage
        if self.top_k is None:
            sorted_answers = self.answer_counts.most_common()
            cumulative_count = 0
            coverage_threshold = (self.coverage_percentile / 100.0) * total_answers

            for i, (answer, count) in enumerate(sorted_answers):
                cumulative_count += count
                if cumulative_count >= coverage_threshold:
                    self.top_k = i + 1
                    break

            print(
                f"Selected top-{self.top_k} answers for {self.coverage_percentile}% coverage"
            )
            print(
                f"Coverage: {cumulative_count}/{total_answers} = {100*cumulative_count/total_answers:.1f}%"
            )

        # Step 4: Create vocabulary
        top_answers = [
            answer for answer, _ in self.answer_counts.most_common(self.top_k)
        ]
        self.answer_vocab = [self.OTHER_TOKEN] + top_answers  # <OTHER> at index 0

        # Step 5: Create mappings
        self.answer_to_idx = {
            answer: idx for idx, answer in enumerate(self.answer_vocab)
        }
        self.idx_to_answer = {idx: answer for answer, idx in self.answer_to_idx.items()}

        print(f"Answer vocabulary size: {len(self.answer_vocab)}")
        print(f"Top answers: {top_answers[:10]}...")  # Show first 10

        return self.answer_to_idx

    def encode(self, answer: str) -> int:
        """Encode answer to class index.

        Args:
            answer: Raw answer string

        Returns:
            Class index
        """
        if self.answer_to_idx is None:
            raise ValueError("Answer vocabulary not built. Call build_vocab() first.")

        # Normalize answer
        normalized_answer = answer.lower().strip()

        # Map to index
        if normalized_answer in self.answer_to_idx:
            return self.answer_to_idx[normalized_answer]
        else:
            return self.answer_to_idx[self.OTHER_TOKEN]  # Index 0

    def decode(self, index: int) -> str:
        """Decode class index back to answer.

        Args:
            index: Class index

        Returns:
            Answer string
        """
        if self.idx_to_answer is None:
            raise ValueError("Answer vocabulary not built. Call build_vocab() first.")

        if index in self.idx_to_answer:
            return self.idx_to_answer[index]
        else:
            return self.OTHER_TOKEN

    def preprocess(self, answer: str) -> torch.Tensor:
        """Complete preprocessing pipeline for answer."""
        index = self.encode(answer)
        return torch.tensor(index, dtype=torch.long)

    def get_vocab_size(self) -> int:
        """Get vocabulary size for model initialization."""
        if self.answer_vocab is None:
            raise ValueError("Answer vocabulary not built. Call build_vocab() first.")
        return len(self.answer_vocab)

    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalanced data.

        Returns:
            Class weights tensor
        """
        if self.answer_counts is None:
            raise ValueError("Answer vocabulary not built. Call build_vocab() first.")

        # Get counts for vocabulary answers
        vocab_counts = []
        total_other = 0

        for answer in self.answer_vocab:
            if answer == self.OTHER_TOKEN:
                # Count all answers not in top-K
                other_count = sum(
                    count
                    for ans, count in self.answer_counts.items()
                    if ans not in self.answer_vocab
                )
                vocab_counts.append(other_count)
                total_other = other_count
            else:
                vocab_counts.append(self.answer_counts.get(answer, 0))

        # Compute inverse frequency weights
        total_samples = sum(vocab_counts)
        weights = []
        for count in vocab_counts:
            if count > 0:
                weight = total_samples / (len(vocab_counts) * count)
                weights.append(weight)
            else:
                weights.append(1.0)

        return torch.tensor(weights, dtype=torch.float32)
