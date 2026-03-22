"""Training Script for Social Sentiment Analysis Model.

This script trains the SocialSentimentFusionModel on the prepared Weibo dataset.

Usage:
    uv run python scripts/train.py

Requirements:
    - Run prepare_data.py first to generate train.csv, val.csv, test.csv
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.trainer import ModelTrainer, TrainingConfig
from app.dataset.dataset import SentimentDataCollator, SentimentDataset, SentimentSample
from app.models.model import ModelConfig, SocialSentimentFusionModel
from transformers import AutoTokenizer


def load_csv_data(csv_path: str) -> list[SentimentSample]:
    """Load CSV file into list of SentimentSample.

    Args:
        csv_path: Path to CSV file.

    Returns:
        List of SentimentSample.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    samples = []
    for _, row in df.iterrows():
        samples.append(
            SentimentSample(
                text=str(row["text"]),
                label=int(row["label"]),
            )
        )
    return samples


def create_data_loaders(
    train_path: str,
    val_path: str,
    tokenizer: AutoTokenizer,
    batch_size: int = 32,
    max_length: int = 128,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders.

    Args:
        train_path: Path to training CSV.
        val_path: Path to validation CSV.
        tokenizer: HuggingFace tokenizer.
        batch_size: Batch size.
        max_length: Max sequence length.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    print("Loading datasets...")

    # Load data
    train_samples = load_csv_data(train_path)
    val_samples = load_csv_data(val_path)

    print(f"  Train samples: {len(train_samples):,}")
    print(f"  Val samples: {len(val_samples):,}")

    # Create datasets
    train_dataset = SentimentDataset(
        data=train_samples,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    val_dataset = SentimentDataset(
        data=val_samples,
        tokenizer=tokenizer,
        max_length=max_length,
    )

    # Create collator
    collator = SentimentDataCollator(tokenizer=tokenizer)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,  # Set > 0 for multiprocessing
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    return train_loader, val_loader


def main() -> None:
    """Main training function."""
    print("=" * 60)
    print("Social Sentiment Analysis Model Training")
    print("=" * 60)

    # Configuration
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
    BATCH_SIZE = 32
    EPOCHS = 5
    MAX_LENGTH = 128
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    processed_dir = project_dir / "data" / "processed"
    checkpoint_dir = project_dir / "checkpoints"

    train_csv = processed_dir / "train.csv"
    val_csv = processed_dir / "val.csv"

    if not train_csv.exists():
        print(f"ERROR: Training data not found at {train_csv}")
        print("Please run prepare_data.py first!")
        sys.exit(1)

    if not val_csv.exists():
        print(f"ERROR: Validation data not found at {val_csv}")
        print("Please run prepare_data.py first!")
        sys.exit(1)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load tokenizer
    print(f"\n[1] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create data loaders
    print("\n[2] Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        str(train_csv),
        str(val_csv),
        tokenizer,
        batch_size=BATCH_SIZE,
        max_length=MAX_LENGTH,
    )

    # Initialize model
    print("\n[3] Initializing model...")
    model_config = ModelConfig(
        pretrained_model_name=MODEL_NAME,
        social_feature_dim=10,
        social_hidden_dim=32,
        fusion_hidden_dim=128,
        num_labels=2,  # Binary classification (positive/negative)
        dropout_prob=0.1,
        freeze_text_encoder=False,  # Fine-tune the encoder
    )

    model = SocialSentimentFusionModel(model_config)
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Training config
    print("\n[4] Training configuration:")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Max length: {MAX_LENGTH}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Warmup ratio: {WARMUP_RATIO}")
    print(f"  Gradient clip: 1.0")

    training_config = TrainingConfig(
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        warmup_ratio=WARMUP_RATIO,
        gradient_clip_norm=1.0,
        save_dir=str(checkpoint_dir),
        metric_for_best="f1_weighted",
        greater_is_better=True,
    )

    # Create trainer
    print("\n[5] Initializing trainer...")
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
    )

    # Train!
    print("\n[6] Starting training...")
    history = trainer.train()

    # Print summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print("\nFinal Results:")
    for metrics in history:
        val_f1 = metrics.val_metrics.f1_weighted if metrics.val_metrics else 0
        val_acc = metrics.val_metrics.accuracy if metrics.val_metrics else 0
        print(f"  Epoch {metrics.epoch}: Train Loss={metrics.train_loss:.4f}, "
              f"Val Acc={val_acc:.4f}, Val F1={val_f1:.4f}")

    print(f"\nBest model saved at: {trainer.best_model_path}")
    print(f"\nTo start the API server:")
    print(f"  uv run uvicorn app.main:app --reload --port 8000")


if __name__ == "__main__":
    main()
