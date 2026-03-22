"""FastAPI Application Entry Point.

This module initializes the FastAPI application with CORS,
lifespan management for model loading, and route registration.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer  # type: ignore[reportUnknownVariableType]

from app.models.model import ModelConfig, SocialSentimentFusionModel


# Global model and tokenizer instances
_model: SocialSentimentFusionModel | None = None
_tokenizer: AutoTokenizer | None = None


def get_model() -> SocialSentimentFusionModel:
    """Get the global model instance.

    Returns:
        SocialSentimentFusionModel instance.

    Raises:
        RuntimeError: If model is not loaded.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Server may be starting up.")
    return _model


def get_tokenizer() -> AutoTokenizer:
    """Get the global tokenizer instance.

    Returns:
        AutoTokenizer instance.

    Raises:
        RuntimeError: If tokenizer is not loaded.
    """
    if _tokenizer is None:
        raise RuntimeError("Tokenizer not loaded. Server may be starting up.")
    return _tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for model loading on startup.

    Loads the trained model weights and tokenizer when the
    FastAPI server starts, making them available for inference.
    """
    global _model, _tokenizer

    print("=" * 60)
    print("Loading model and tokenizer...")
    print("=" * 60)

    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[reportUnknownVariableType]
        "hfl/chinese-roberta-wwm-ext"
    )

    # Initialize model
    model_config = ModelConfig(
        pretrained_model_name="hfl/chinese-roberta-wwm-ext",
        social_feature_dim=10,
        social_hidden_dim=32,
        fusion_hidden_dim=128,
        num_labels=3,
        dropout_prob=0.1,
        freeze_text_encoder=True,
    )
    _model = SocialSentimentFusionModel(model_config)

    # Load trained weights
    import torch
    import os

    checkpoint_path = "checkpoints/best_model.pt"
    if os.path.exists(checkpoint_path):
        _model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        print(f"Loaded checkpoint from {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using untrained model for inference")

    _model.eval()

    # Detect device
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    _model.to(device)

    print(f"Model loaded successfully on device: {device}")
    print("=" * 60)

    yield

    # Cleanup on shutdown
    print("Shutting down... Model unloaded.")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title="Social Sentiment Analysis API",
        description="Deep Learning based sentiment analysis for social media text",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://localhost:3000",
            "http://127.0.0.1:5173",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    from app.api.routes import router as api_router
    app.include_router(api_router, prefix="/api")

    @app.get("/")
    async def root() -> dict[str, str]:  # type: ignore[reportUnusedFunction]
        """Health check endpoint."""
        return {"status": "ok", "service": "Social Sentiment Analysis API"}

    @app.get("/health")
    async def health() -> dict[str, str]:  # type: ignore[reportUnusedFunction]
        """Detailed health check."""
        return {
            "status": "ok",
            "model_loaded": str(_model is not None),
            "tokenizer_loaded": str(_tokenizer is not None),
        }

    return app


# Create the app instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
