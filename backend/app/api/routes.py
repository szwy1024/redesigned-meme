"""API Routes for Sentiment Analysis.

This module defines the API endpoints for text analysis,
including request/response models and the prediction logic.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.dataset.text_cleaner import get_default_cleaner
from app.main import get_model, get_tokenizer


# ============== Pydantic Models ==============


class SentimentLabel(str, Enum):
    """Sentiment classification labels."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class AnalyzeRequest(BaseModel):
    """Request body for text analysis."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Raw social media text to analyze",
        examples=["今天心情很好！终于完成了这个项目 😊 #开心"],
    )


class SocialFeaturesResponse(BaseModel):
    """Extracted social features from text cleaning."""

    emoji_count: int = Field(..., description="Number of emojis found")
    emoji_descriptions: list[str] = Field(
        default_factory=list,
        description="List of converted emoji descriptions",
    )
    hashtag_count: int = Field(..., description="Number of hashtags found")
    hashtags: list[str] = Field(
        default_factory=list,
        description="Extracted hashtags",
    )
    mention_count: int = Field(..., description="Number of mentions found")
    mentions: list[str] = Field(
        default_factory=list,
        description="Extracted @mentions",
    )
    url_count: int = Field(..., description="Number of URLs removed")
    exclamation_count: int = Field(..., description="Exclamation mark count")
    question_count: int = Field(..., description="Question mark count")
    cleaned_text: str = Field(..., description="Cleaned text for analysis")


class AnalyzeResponse(BaseModel):
    """Response body for text analysis."""

    sentiment: SentimentLabel = Field(..., description="Predicted sentiment")
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence score",
    )
    confidence_per_class: dict[str, float] = Field(
        ...,
        description="Confidence scores for each class",
    )
    social_features: SocialFeaturesResponse = Field(
        ...,
        description="Extracted social media features",
    )
    raw_text: str = Field(..., description="Original input text")


# ============== Helper Functions ==============

_LABEL_MAP: dict[int, SentimentLabel] = {
    0: SentimentLabel.POSITIVE,
    1: SentimentLabel.NEGATIVE,
    2: SentimentLabel.NEUTRAL,
}


def create_social_features_vector(
    emoji_count: int,
    hashtag_count: int,
    mention_count: int,
    exclamation_count: int,
    question_count: int,
    url_count: int,
    html_count: int,
    text_length: int,
    emoji_polarity_sum: float,
) -> list[float]:
    """Create social features vector from cleaning results.

    Args:
        emoji_count: Number of emojis.
        hashtag_count: Number of hashtags.
        mention_count: Number of mentions.
        exclamation_count: Exclamation marks.
        question_count: Question marks.
        url_count: Number of URLs.
        html_count: Number of HTML tags.
        text_length: Length of cleaned text.
        emoji_polarity_sum: Sum of emoji polarity values.

    Returns:
        List of 10 social feature values.
    """
    return [
        float(emoji_count),
        float(hashtag_count),
        float(mention_count),
        float(exclamation_count),
        float(question_count),
        float(url_count),
        float(html_count),
        float(text_length) / 128.0,
        emoji_polarity_sum,
        float(exclamation_count + question_count),
    ]


# Emoji polarity mapping for social features
_EMOJI_POLARITY: dict[str, float] = {
    "[笑脸]": 0.5,
    "[微笑]": 0.4,
    "[大笑]": 0.7,
    "[苦笑]": -0.2,
    "[笑哭]": 0.6,
    "[爱慕]": 0.8,
    "[崇拜]": 0.8,
    "[飞吻]": 0.6,
    "[亲亲]": 0.5,
    "[馋嘴]": 0.3,
    "[调皮]": 0.3,
    "[得意]": 0.4,
    "[愤怒]": -0.8,
    "[生气]": -0.7,
    "[失望]": -0.6,
    "[忧郁]": -0.5,
    "[尴尬]": -0.3,
    "[心碎]": -0.7,
    "[爱]": 0.8,
    "[红心]": 0.7,
    "[点赞]": 0.6,
    "[点踩]": -0.5,
    "[酷]": 0.5,
    "[加油]": 0.5,
    "[庆祝]": 0.7,
    "[烟花]": 0.6,
}


def get_emoji_polarity_sum(emoji_descriptions: list[str]) -> float:
    """Calculate sum of emoji polarity values.

    Args:
        emoji_descriptions: List of emoji descriptions.

    Returns:
        Sum of polarity values.
    """
    return sum(_EMOJI_POLARITY.get(desc, 0.0) for desc in emoji_descriptions)


# ============== Router ==============

router = APIRouter()


@router.post("/predict", response_model=AnalyzeResponse)
async def analyze_text(request: AnalyzeRequest) -> AnalyzeResponse:
    """Analyze sentiment of social media text.

    This endpoint performs the full pipeline:
    1. Text cleaning (HTML, URL, emoji conversion)
    2. Tokenization
    3. Model inference
    4. Response formatting

    Args:
        request: AnalyzeRequest with raw text.

    Returns:
        AnalyzeResponse with sentiment prediction and social features.

    Raises:
        HTTPException: If model or tokenizer is not loaded.
    """
    try:
        model = get_model()
        tokenizer = get_tokenizer()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Step 1: Text cleaning
    cleaner = get_default_cleaner()
    cleaning_result = cleaner.clean(request.text)

    # Extract emoji descriptions for response
    import re
    emoji_pattern = re.compile(r"\[.*?\]")
    emoji_descriptions = emoji_pattern.findall(cleaning_result.cleaned_text)
    emoji_descriptions = [
        desc for desc in emoji_descriptions
        if desc in _EMOJI_POLARITY or desc.startswith("[")
    ]

    # Calculate emoji polarity
    emoji_polarity_sum = get_emoji_polarity_sum(emoji_descriptions)

    # Create social features vector
    social_features_vec = create_social_features_vector(
        emoji_count=cleaning_result.converted_emojis,
        hashtag_count=len(cleaning_result.extracted_hashtags),
        mention_count=len(cleaning_result.extracted_mentions),
        exclamation_count=cleaning_result.cleaned_text.count("!"),
        question_count=cleaning_result.cleaned_text.count("?"),
        url_count=cleaning_result.removed_urls,
        html_count=cleaning_result.removed_html_tags,
        text_length=len(cleaning_result.cleaned_text),
        emoji_polarity_sum=emoji_polarity_sum,
    )

    # Step 2: Tokenization
    import torch
    encoding: dict[str, Any] = tokenizer(  # type: ignore[reportCallIssue]
        cleaning_result.cleaned_text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids: torch.Tensor = encoding["input_ids"]  # type: ignore[reportUnknownVariableType]
    attention_mask: torch.Tensor = encoding["attention_mask"]  # type: ignore[reportUnknownVariableType]

    # Move to same device as model
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)  # type: ignore[reportUnknownMemberType]
    attention_mask = attention_mask.to(device)  # type: ignore[reportUnknownMemberType]

    # Step 3: Model inference
    social_features_tensor = torch.tensor(
        social_features_vec,
        dtype=torch.float32,
    ).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(input_ids, attention_mask, social_features_tensor)
        probabilities = torch.softmax(logits, dim=1)

    # Step 4: Parse results
    probs = probabilities.squeeze(0).cpu().numpy()  # type: ignore[reportUnknownVariableType]
    pred_idx = int(probs.argmax())
    confidence = float(probs[pred_idx])

    # Build confidence per class
    confidence_per_class = {
        "positive": float(probs[0]),
        "negative": float(probs[1]),
        "neutral": float(probs[2]),
    }

    # Build response
    social_features_response = SocialFeaturesResponse(
        emoji_count=cleaning_result.converted_emojis,
        emoji_descriptions=emoji_descriptions,
        hashtag_count=len(cleaning_result.extracted_hashtags),
        hashtags=cleaning_result.extracted_hashtags,
        mention_count=len(cleaning_result.extracted_mentions),
        mentions=cleaning_result.extracted_mentions,
        url_count=cleaning_result.removed_urls,
        exclamation_count=cleaning_result.cleaned_text.count("!"),
        question_count=cleaning_result.cleaned_text.count("?"),
        cleaned_text=cleaning_result.cleaned_text,
    )

    return AnalyzeResponse(
        sentiment=_LABEL_MAP[pred_idx],
        confidence=confidence,
        confidence_per_class=confidence_per_class,
        social_features=social_features_response,
        raw_text=request.text,
    )


@router.get("/labels")
async def get_labels() -> dict[str, list[str]]:
    """Get available sentiment labels.

    Returns:
        Dictionary with label names.
    """
    return {
        "labels": [label.value for label in SentimentLabel]
    }
