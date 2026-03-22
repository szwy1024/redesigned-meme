"""Social Media Sentiment Fusion Model.

This module implements a dual-channel late fusion architecture for
sentiment analysis, combining text features with social features.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoModel  # type: ignore[reportUnknownVariableType]


@dataclass
class ModelConfig:
    """Configuration for SocialSentimentFusionModel."""

    pretrained_model_name: str = "hfl/chinese-roberta-wwm-ext"
    text_hidden_size: int | None = None
    social_feature_dim: int = 10
    social_hidden_dim: int = 32
    fusion_hidden_dim: int = 128
    num_labels: int = 3
    dropout_prob: float = 0.1
    freeze_text_encoder: bool = False


class SocialFeatureExtractor(nn.Module):
    """MLP-based social feature extractor.

    Maps social features (emoji count, punctuation intensity, etc.)
    to a fixed-dimensional hidden representation.

    Attributes:
        mlp: Multi-layer perceptron for feature transformation.
        layer_norm: Layer normalization for stabilized outputs.
    """

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        """Initialize the social feature extractor.

        Args:
            input_dim: Dimension of input social features.
            hidden_dim: Dimension of output hidden representation.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, social_features: torch.Tensor) -> torch.Tensor:
        """Extract features from social signals.

        Args:
            social_features: Tensor of shape (batch_size, input_dim).

        Returns:
            Tensor of shape (batch_size, hidden_dim).
        """
        return self.mlp(social_features)


class FusionClassifier(nn.Module):
    """Fusion and classification head.

    Concatenates text and social features, then classifies into
    sentiment categories.

    Attributes:
        classifier: Classification MLP with dropout.
        output: Final linear layer for logits.
    """

    def __init__(
        self,
        text_feature_dim: int,
        social_feature_dim: int,
        fusion_hidden_dim: int,
        num_labels: int,
        dropout_prob: float,
    ) -> None:
        """Initialize the fusion classifier.

        Args:
            text_feature_dim: Dimension of text [CLS] vector.
            social_feature_dim: Dimension of social feature vector.
            fusion_hidden_dim: Hidden dimension in fusion MLP.
            num_labels: Number of output classes.
            dropout_prob: Dropout probability.
        """
        super().__init__()
        fusion_input_dim = text_feature_dim + social_feature_dim
        self.classifier = nn.Sequential(
            nn.Linear(fusion_input_dim, fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        )
        self.output = nn.Linear(fusion_hidden_dim, num_labels)

    def forward(self, text_features: torch.Tensor, social_features: torch.Tensor) -> torch.Tensor:
        """Fuse features and classify.

        Args:
            text_features: Text [CLS] vectors (batch_size, text_feature_dim).
            social_features: Social feature vectors (batch_size, social_feature_dim).

        Returns:
            Logits of shape (batch_size, num_labels).
        """
        fused = torch.cat([text_features, social_features], dim=1)
        hidden = self.classifier(fused)
        logits = self.output(hidden)
        return logits


class SocialSentimentFusionModel(nn.Module):
    """Dual-channel late fusion model for social media sentiment analysis.

    This model combines:
    - Channel A: Pre-trained text encoder (RoBERTa) extracting [CLS] vector
    - Channel B: MLP extracting social features from dense vectors
    - Fusion & Classification head for sentiment prediction

    Attributes:
        text_encoder: Pre-trained language model for text encoding.
        social_extractor: MLP for social feature extraction.
        classifier: Fusion and classification head.
        config: Model configuration.
        freeze_text_encoder: Whether to freeze text encoder weights.

    Example:
        >>> config = ModelConfig(social_feature_dim=10, num_labels=3)
        >>> model = SocialSentimentFusionModel(config)
        >>> input_ids = torch.randint(0, 1000, (4, 128))
        >>> attention_mask = torch.ones(4, 128)
        >>> social_features = torch.randn(4, 10)
        >>> logits = model(input_ids, attention_mask, social_features)
        >>> print(logits.shape)  # torch.Size([4, 3])
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize the fusion model.

        Args:
            config: Model configuration. Uses defaults if not provided.
        """
        super().__init__()
        self.config = config or ModelConfig()
        cfg = self.config

        # Channel A: Text Encoder
        self.text_encoder = AutoModel.from_pretrained(cfg.pretrained_model_name)  # type: ignore[reportUnknownVariableType]
        text_hidden_size: int = cfg.text_hidden_size or self.text_encoder.config.hidden_size  # type: ignore[reportUnknownVariableType]

        # Channel B: Social Feature Extractor
        self.social_extractor = SocialFeatureExtractor(
            input_dim=cfg.social_feature_dim,
            hidden_dim=cfg.fusion_hidden_dim,
        )

        # Fusion & Classification Head
        self.classifier = FusionClassifier(
            text_feature_dim=text_hidden_size,  # type: ignore[reportArgumentType]
            social_feature_dim=cfg.fusion_hidden_dim,
            fusion_hidden_dim=cfg.fusion_hidden_dim,
            num_labels=cfg.num_labels,
            dropout_prob=cfg.dropout_prob,
        )

        if cfg.freeze_text_encoder:
            for param in self.text_encoder.parameters():  # type: ignore[reportUnknownVariableType]
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        social_features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass of the fusion model.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length).
            attention_mask: Attention mask of shape (batch_size, seq_length).
            social_features: Social features of shape (batch_size, social_feature_dim).

        Returns:
            Logits of shape (batch_size, num_labels).
        """
        # Channel A: Text encoding
        text_outputs = self.text_encoder(  # type: ignore[reportCallIssue]
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Extract [CLS] vector (first token)
        text_cls: torch.Tensor = text_outputs.last_hidden_state[:, 0, :]  # type: ignore[reportUnknownVariableType]

        # Channel B: Social feature extraction
        social_hidden: torch.Tensor = self.social_extractor(social_features)

        # Fusion & Classification
        logits: torch.Tensor = self.classifier(text_cls, social_hidden)

        return logits

    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Extract text features without social features.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_length).
            attention_mask: Attention mask of shape (batch_size, seq_length).

        Returns:
            [CLS] vectors of shape (batch_size, text_hidden_size).
        """
        outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)  # type: ignore[reportCallIssue]
        return outputs.last_hidden_state[:, 0, :]  # type: ignore[reportUnknownVariableType]

    def get_social_features(self, social_features: torch.Tensor) -> torch.Tensor:
        """Extract social features.

        Args:
            social_features: Raw social features of shape (batch_size, social_feature_dim).

        Returns:
            Extracted social features of shape (batch_size, fusion_hidden_dim).
        """
        return self.social_extractor(social_features)


def create_social_features_from_cleaning_results(
    num_emojis: int,
    num_hashtags: int,
    num_mentions: int,
    exclamation_count: int,
    question_count: int,
    has_url: int,
    has_html: int,
    text_length: int,
    emoji_polarity_sum: float,
) -> list[float]:
    """Create a social feature vector from text cleaning statistics.

    This is a helper function to generate social feature vectors
    based on text cleaning results.

    Args:
        num_emojis: Number of emojis in text.
        num_hashtags: Number of hashtags.
        num_mentions: Number of mentions.
        exclamation_count: Count of exclamation marks.
        question_count: Count of question marks.
        has_url: Binary indicator (0/1) for URL presence.
        has_html: Binary indicator (0/1) for HTML tag presence.
        text_length: Length of cleaned text.
        emoji_polarity_sum: Sum of emoji polarity values (-1 to 1 per emoji).

    Returns:
        List of 10 social feature values.
    """
    return [
        float(num_emojis),
        float(num_hashtags),
        float(num_mentions),
        float(exclamation_count),
        float(question_count),
        float(has_url),
        float(has_html),
        float(text_length) / 128.0,  # normalize to [0, 1]
        emoji_polarity_sum,  # already in [-N, N] range
        float(exclamation_count + question_count),  # punctuation intensity
    ]


if __name__ == "__main__":
    # Test script for SocialSentimentFusionModel
    print("=" * 60)
    print("SocialSentimentFusionModel Test")
    print("=" * 60)

    # Configuration
    config = ModelConfig(
        pretrained_model_name="hfl/chinese-roberta-wwm-ext",
        social_feature_dim=10,
        social_hidden_dim=32,
        fusion_hidden_dim=128,
        num_labels=3,
        dropout_prob=0.1,
        freeze_text_encoder=True,  # Freeze for faster testing
    )

    # Initialize model
    print("\n[1] Initializing model...")
    model = SocialSentimentFusionModel(config)
    print(f"    Text encoder hidden size: {config.text_hidden_size or 'auto-detected'}")
    print(f"    Social feature dim: {config.social_feature_dim}")
    print(f"    Fusion hidden dim: {config.fusion_hidden_dim}")
    print(f"    Num labels: {config.num_labels}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")

    # Create dummy inputs
    print("\n[2] Creating dummy inputs...")
    batch_size = 4
    seq_length = 128

    input_ids = torch.randint(0, 21128, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    social_features = torch.randn(batch_size, config.social_feature_dim)

    print(f"    input_ids shape: {input_ids.shape}")
    print(f"    attention_mask shape: {attention_mask.shape}")
    print(f"    social_features shape: {social_features.shape}")

    # Forward pass
    print("\n[3] Running forward pass...")
    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask, social_features)

    print(f"    logits shape: {logits.shape}")
    print(f"    logits:\\n{logits}")

    # Verify output
    expected_shape = (batch_size, config.num_labels)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
    print(f"\\n    ✓ Output shape verified: {logits.shape}")

    # Test text feature extraction
    print("\n[4] Testing text feature extraction...")
    with torch.no_grad():
        text_features = model.get_text_features(input_ids, attention_mask)
    print(f"    text_features shape: {text_features.shape}")

    # Test social feature extraction
    print("\n[5] Testing social feature extraction...")
    with torch.no_grad():
        social_hidden = model.get_social_features(social_features)
    print(f"    social_hidden shape: {social_hidden.shape}")

    # Test create_social_features_from_cleaning_results
    print("\n[6] Testing social feature creation helper...")
    features = create_social_features_from_cleaning_results(
        num_emojis=3,
        num_hashtags=2,
        num_mentions=1,
        exclamation_count=2,
        question_count=1,
        has_url=0,
        has_html=1,
        text_length=64,
        emoji_polarity_sum=1.5,
    )
    print(f"    Generated social feature vector: {features}")
    print(f"    Vector length: {len(features)}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
