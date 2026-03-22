"""PyTorch Dataset for Social Media Sentiment Analysis.

This module provides a custom Dataset class for loading and preprocessing
social media text data for sentiment analysis tasks.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from app.dataset.text_cleaner import TextCleaner, get_default_cleaner


class SentimentLabel(Enum):
    """Sentiment label enumeration."""

    POSITIVE = 0
    NEGATIVE = 1
    NEUTRAL = 2

    @classmethod
    def from_string(cls, label: str) -> SentimentLabel:
        """Convert string label to SentimentLabel enum.

        Args:
            label: String label (positive, negative, neutral or 0, 1, 2).

        Returns:
            Corresponding SentimentLabel enum value.

        Raises:
            ValueError: If label is not a valid sentiment label.
        """
        label_lower = label.lower().strip()
        mapping: dict[str, SentimentLabel] = {
            "positive": cls.POSITIVE,
            "0": cls.POSITIVE,
            "pos": cls.POSITIVE,
            "negative": cls.NEGATIVE,
            "1": cls.NEGATIVE,
            "neg": cls.NEGATIVE,
            "neutral": cls.NEUTRAL,
            "2": cls.NEUTRAL,
            "neu": cls.NEUTRAL,
        }
        if label_lower not in mapping:
            raise ValueError(
                f"Invalid sentiment label: {label}. "
                f"Expected one of: {list(mapping.keys())}"
            )
        return mapping[label_lower]

    def to_index(self) -> int:
        """Get the integer index of this label."""
        return self.value


@dataclass
class SentimentSample:
    """A single sample for sentiment analysis."""

    text: str
    label: int
    raw_text: str | None = None
    sample_id: str | None = None


@dataclass
class TokenizedSample:
    """Tokenized sample with model inputs."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    label: torch.Tensor


class SentimentDataset(Dataset[Any]):
    """PyTorch Dataset for Social Media Sentiment Analysis.

    This dataset handles text cleaning, tokenization, and label encoding
    for social media sentiment analysis tasks.

    Args:
        data: List of SentimentSample or dict with 'text' and 'label' keys.
        tokenizer: HuggingFace AutoTokenizer instance.
        max_length: Maximum sequence length (default: 128).
        text_cleaner: TextCleaner instance for preprocessing.

    Attributes:
        samples: List of preprocessed SentimentSample objects.
        tokenizer: Tokenizer used for encoding texts.
        max_length: Maximum sequence length.
        text_cleaner: Cleaner used for preprocessing.

    Example:
        >>> from transformers import AutoTokenizer
        >>> from app.dataset.dataset import SentimentDataset, SentimentSample
        >>> tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
        >>> samples = [
        ...     SentimentSample(text="今天很开心！", label=0),
        ...     SentimentSample(text="太失望了", label=1),
        ... ]
        >>> dataset = SentimentDataset(samples, tokenizer)
        >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
        >>> for batch in dataloader:
        ...     print(batch["input_ids"].shape)  # (batch_size, 128)
    """

    def __init__(
        self,
        data: list[SentimentSample] | list[dict[str, Any]],
        tokenizer: AutoTokenizer,
        max_length: int = 128,
        text_cleaner: TextCleaner | None = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            data: List of samples (SentimentSample or dict).
            tokenizer: HuggingFace AutoTokenizer instance.
            max_length: Maximum sequence length.
            text_cleaner: Optional TextCleaner instance.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.text_cleaner = text_cleaner or get_default_cleaner()

        self.samples: list[SentimentSample] = []
        for item in data:
            if isinstance(item, dict):
                sample = SentimentSample(
                    text=item["text"],
                    label=int(item["label"]),
                    raw_text=item.get("raw_text"),
                    sample_id=item.get("id"),
                )
            else:
                sample = item
            self.samples.append(sample)

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        """Get a single tokenized sample.

        Args:
            index: Sample index.

        Returns:
            Dictionary containing input_ids, attention_mask, and label tensors.
        """
        sample = self.samples[index]

        cleaned_text = self.text_cleaner.clean(sample.text).cleaned_text

        encoding: dict[str, Any] = self.tokenizer(  # type: ignore[reportCallIssue]
            cleaned_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids: torch.Tensor = encoding["input_ids"].squeeze(0)  # type: ignore[reportUnknownVariableType]
        attention_mask: torch.Tensor = encoding["attention_mask"].squeeze(0)  # type: ignore[reportUnknownVariableType]
        label: torch.Tensor = torch.tensor(sample.label, dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": label,
        }

    def get_raw_sample(self, index: int) -> SentimentSample:
        """Get the original (un-tokenized) sample.

        Args:
            index: Sample index.

        Returns:
            Original SentimentSample.
        """
        return self.samples[index]


class SentimentDataCollator:
    """Data collator for batching tokenized samples.

    This collator ensures consistent tensor shapes within a batch
    by using the tokenizer's built-in padding mechanism.

    Attributes:
        tokenizer: Tokenizer for decoding and padding.
        padding: Padding strategy ('longest' or 'max_length').
        max_length: Maximum length for padding.
        label_name: Name of the label key in output dict.
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        padding: str = "longest",
        max_length: int | None = None,
        label_name: str = "label",
    ) -> None:
        """Initialize the data collator.

        Args:
            tokenizer: Tokenizer for padding.
            padding: Padding strategy.
            max_length: Maximum length for padding.
            label_name: Key name for labels in output.
        """
        self.tokenizer = tokenizer
        self.padding = padding
        self.max_length: int = max_length or tokenizer.model_max_length  # type: ignore[reportAttributeAccessIssue]
        self.label_name = label_name

    def __call__(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            batch: List of sample dicts from __getitem__.

        Returns:
            Collated batch dict with stacked tensors.
        """
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": labels,
        }


def create_dummy_data(num_samples: int = 100) -> list[SentimentSample]:
    """Create dummy sentiment data for testing.

    Args:
        num_samples: Number of samples to generate.

    Returns:
        List of SentimentSample objects.
    """
    texts = [
        "今天心情很好！",
        "这个产品太棒了，强烈推荐！",
        "服务态度差，强烈差评",
        "一般般吧，没什么特别的",
        "太失望了，完全不值这个价",
        "还不错，值得购买",
        "垃圾产品，不要买",
        "一般般，中规中矩",
        "太棒了！完美！",
        "不怎么样，浪费钱",
    ]
    labels = [0, 0, 1, 2, 1, 0, 1, 2, 0, 1]  # 0=positive, 1=negative, 2=neutral

    samples: list[SentimentSample] = []
    for i in range(num_samples):
        text = texts[i % len(texts)]
        label = labels[i % len(labels)]
        samples.append(
            SentimentSample(
                text=text,
                label=label,
                sample_id=f"sample_{i}",
            )
        )
    return samples
