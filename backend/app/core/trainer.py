"""Model Trainer for Social Media Sentiment Analysis.

This module provides a structured ModelTrainer class for training and evaluating
the SocialSentimentFusionModel with comprehensive metrics.

NOTE: Ensure the following dependencies are installed:
    uv add scikit-learn numpy torch
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

# sklearn metrics - already in pyproject.toml
from sklearn.metrics import (  # type: ignore[reportUnknownVariableType, reportUnusedImport]
    accuracy_score,  # type: ignore[reportUnknownVariableType]
    precision_score,  # type: ignore[reportUnknownVariableType]
    recall_score,  # type: ignore[reportUnknownVariableType]
    f1_score,  # type: ignore[reportUnknownVariableType]
    confusion_matrix,  # type: ignore[reportUnknownVariableType]
)

from app.models.model import SocialSentimentFusionModel


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""

    accuracy: float
    precision_macro: float
    precision_weighted: float
    recall_macro: float
    recall_weighted: float
    f1_macro: float
    f1_weighted: float
    confusion_mat: Any  # numpy.ndarray
    predictions: Any  # numpy.ndarray
    targets: Any  # numpy.ndarray


@dataclass
class TrainingConfig:
    """Configuration for training."""

    epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    save_dir: str = "checkpoints"
    metric_for_best: str = "f1_weighted"
    greater_is_better: bool = True


@dataclass
class TrainingMetrics:
    """Container for training epoch metrics."""

    epoch: int
    train_loss: float
    val_loss: float | None = None
    val_metrics: EvaluationMetrics | None = None


class ModelTrainer:
    """Trainer for SocialSentimentFusionModel.

    Handles training loop, evaluation with comprehensive metrics,
    and best model checkpoint saving.

    Example:
        >>> trainer = ModelTrainer(model, train_loader, val_loader)
        >>> trainer.train()
    """

    def __init__(
        self,
        model: SocialSentimentFusionModel,
        train_loader: DataLoader[Any],
        val_loader: DataLoader[Any] | None,
        config: TrainingConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            model: SocialSentimentFusionModel instance.
            train_loader: Training DataLoader.
            val_loader: Optional validation DataLoader.
            config: Training configuration.
            device: Target device. Auto-detected if None.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        self.device = device or self._get_device()

        self.model.to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        self.criterion = nn.CrossEntropyLoss()

        self.best_f1: float = 0.0
        self.best_model_path: str = ""

        os.makedirs(self.config.save_dir, exist_ok=True)

    @staticmethod
    def _get_device() -> torch.device:
        """Auto-detect the best available device.

        Returns:
            torch.device: cuda, mps, or cpu.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch.

        Args:
            epoch: Current epoch number (for logging).

        Returns:
            Average training loss.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            leave=False,
        )

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            social_features = self._get_social_features(batch, input_ids.size(0))
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(input_ids, attention_mask, social_features)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            )
            self.optimizer.step()  # type: ignore[reportUnknownMemberType]

            total_loss += loss.item()
            num_batches += 1

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})  # type: ignore[reportUnknownMemberType]

        return total_loss / num_batches

    def _get_social_features(
        self, batch: dict[str, torch.Tensor], batch_size: int
    ) -> torch.Tensor:
        """Get social features from batch or create dummy features.

        Args:
            batch: Data batch dict.
            batch_size: Batch size.

        Returns:
            Social features tensor.
        """
        if "social_features" in batch:
            return batch["social_features"].to(self.device)
        return torch.randn(batch_size, 10).to(self.device)

    @torch.no_grad()
    def evaluate(self) -> tuple[float, EvaluationMetrics]:
        """Evaluate the model on validation set.

        Returns:
            Tuple of (validation loss, EvaluationMetrics).
        """
        if self.val_loader is None:
            raise ValueError("Validation loader is not provided")

        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        all_predictions: list[int] = []
        all_targets: list[int] = []

        for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            social_features = self._get_social_features(batch, input_ids.size(0))
            labels = batch["label"].to(self.device)

            logits = self.model(input_ids, attention_mask, social_features)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()
            num_batches += 1

            predictions = torch.argmax(logits, dim=1).cpu().numpy().tolist()  # type: ignore[reportUnknownVariableType]
            targets = labels.cpu().numpy().tolist()  # type: ignore[reportUnknownVariableType]

            all_predictions.extend(predictions)  # type: ignore[reportArgumentType]
            all_targets.extend(targets)  # type: ignore[reportArgumentType]

        avg_loss = total_loss / num_batches

        metrics = EvaluationMetrics(
            accuracy=accuracy_score(all_targets, all_predictions),
            precision_macro=precision_score(
                all_targets, all_predictions, average="macro", zero_division=0  # type: ignore
            ),
            precision_weighted=precision_score(
                all_targets, all_predictions, average="weighted", zero_division=0  # type: ignore
            ),
            recall_macro=recall_score(
                all_targets, all_predictions, average="macro", zero_division=0  # type: ignore
            ),
            recall_weighted=recall_score(
                all_targets, all_predictions, average="weighted", zero_division=0  # type: ignore
            ),
            f1_macro=f1_score(
                all_targets, all_predictions, average="macro", zero_division=0  # type: ignore
            ),
            f1_weighted=f1_score(
                all_targets, all_predictions, average="weighted", zero_division=0  # type: ignore
            ),
            confusion_mat=confusion_matrix(all_targets, all_predictions),
            predictions=np.array(all_predictions),
            targets=np.array(all_targets),
        )

        return avg_loss, metrics

    def print_metrics(self, metrics: EvaluationMetrics, prefix: str = "") -> None:
        """Print evaluation metrics in a formatted way.

        Args:
            metrics: EvaluationMetrics to print.
            prefix: Optional prefix for each line.
        """
        print(f"\n{prefix}===== Evaluation Metrics =====")
        print(f"{prefix}Accuracy:           {metrics.accuracy:.4f}")
        print(f"{prefix}Precision (macro):  {metrics.precision_macro:.4f}")
        print(f"{prefix}Precision (weight): {metrics.precision_weighted:.4f}")
        print(f"{prefix}Recall (macro):     {metrics.recall_macro:.4f}")
        print(f"{prefix}Recall (weighted):  {metrics.recall_weighted:.4f}")
        print(f"{prefix}F1 (macro):         {metrics.f1_macro:.4f}")
        print(f"{prefix}F1 (weighted):      {metrics.f1_weighted:.4f}")
        print(f"{prefix}Confusion Matrix:")
        print(f"{prefix}{metrics.confusion_mat}")
        print(f"{prefix}==============================\n")

    def save_checkpoint(
        self, epoch: int, metrics: EvaluationMetrics, is_best: bool
    ) -> None:
        """Save model checkpoint.

        Args:
            epoch: Current epoch number.
            metrics: Current evaluation metrics.
            is_best: Whether this is the best model so far.
        """
        checkpoint_path = os.path.join(
            self.config.save_dir,
            f"checkpoint_epoch_{epoch}.pt",
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "f1_weighted": metrics.f1_weighted,
                "accuracy": metrics.accuracy,
            },
            checkpoint_path,
        )

        if is_best:
            best_path = os.path.join(self.config.save_dir, "best_model.pt")
            torch.save(self.model.state_dict(), best_path)
            self.best_model_path = best_path
            print(f"  ✓ New best model saved! F1: {metrics.f1_weighted:.4f}")

    def train(self) -> list[TrainingMetrics]:
        """Run the full training loop.

        Returns:
            List of TrainingMetrics for each epoch.
        """
        print("=" * 60)
        print("Training Configuration")
        print("=" * 60)
        print(f"  Device:           {self.device}")
        print(f"  Epochs:           {self.config.epochs}")
        print(f"  Learning Rate:   {self.config.learning_rate}")
        print(f"  Weight Decay:    {self.config.weight_decay}")
        print(f"  Gradient Clip:  {self.config.gradient_clip_norm}")
        print(f"  Save Directory: {self.config.save_dir}")
        print(f"  Metric for Best: {self.config.metric_for_best}")
        print("=" * 60)

        history: list[TrainingMetrics] = []

        for epoch in range(1, self.config.epochs + 1):
            print(f"\nEpoch {epoch}/{self.config.epochs}")

            train_loss = self.train_epoch(epoch)
            print(f"  Train Loss: {train_loss:.4f}")

            if self.val_loader is not None:
                val_loss, metrics = self.evaluate()
                print(f"  Val Loss:   {val_loss:.4f}")
                self.print_metrics(metrics, prefix="  ")

                current_f1 = (
                    metrics.f1_weighted
                    if self.config.metric_for_best == "f1_weighted"
                    else metrics.f1_macro
                    if self.config.metric_for_best == "f1_macro"
                    else metrics.accuracy
                )

                is_best = (
                    current_f1 > self.best_f1
                    if self.config.greater_is_better
                    else current_f1 < self.best_f1
                )

                if is_best:
                    self.best_f1 = current_f1

                self.save_checkpoint(epoch, metrics, is_best)

                history.append(
                    TrainingMetrics(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        val_metrics=metrics,
                    )
                )
            else:
                checkpoint_path = os.path.join(
                    self.config.save_dir,
                    f"checkpoint_epoch_{epoch}.pt",
                )
                torch.save(self.model.state_dict(), checkpoint_path)
                history.append(
                    TrainingMetrics(epoch=epoch, train_loss=train_loss)
                )

        print("\n" + "=" * 60)
        print("Training Complete!")
        print(f"Best {self.config.metric_for_best}: {self.best_f1:.4f}")
        if self.best_model_path:
            print(f"Best model saved at: {self.best_model_path}")
        print("=" * 60)

        return history

    def load_best_model(self) -> None:
        """Load the best model checkpoint."""
        if self.best_model_path and os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print(f"Loaded best model from {self.best_model_path}")
        else:
            raise FileNotFoundError("Best model checkpoint not found")


if __name__ == "__main__":
    # Example usage of ModelTrainer
    print("=" * 60)
    print("ModelTrainer Example")
    print("=" * 60)

    from transformers import AutoTokenizer  # type: ignore

    from app.dataset.dataset import (
        SentimentDataCollator,
        SentimentDataset,
        create_dummy_data,
    )
    from app.models.model import ModelConfig

    # Configuration
    MODEL_NAME = "hfl/chinese-roberta-wwm-ext"
    BATCH_SIZE = 8
    EPOCHS = 2
    MAX_LENGTH = 64  # Smaller for faster testing

    print(f"\n[1] Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)  # type: ignore[reportUnknownVariableType]

    print("\n[2] Creating dummy dataset...")
    samples = create_dummy_data(num_samples=32)
    dataset = SentimentDataset(
        data=samples,
        tokenizer=tokenizer,  # type: ignore[reportArgumentType]
        max_length=MAX_LENGTH,
    )

    print("\n[3] Creating data loaders...")
    collator = SentimentDataCollator(tokenizer=tokenizer)  # type: ignore[reportArgumentType]
    train_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collator,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collator,
    )

    print("\n[4] Initializing model...")
    model_config = ModelConfig(
        pretrained_model_name=MODEL_NAME,
        social_feature_dim=10,
        social_hidden_dim=32,
        fusion_hidden_dim=64,  # Smaller for faster testing
        num_labels=3,
        dropout_prob=0.1,
        freeze_text_encoder=True,
    )
    model = SocialSentimentFusionModel(model_config)

    print("\n[5] Initializing trainer...")
    training_config = TrainingConfig(
        epochs=EPOCHS,
        learning_rate=1e-3,
        weight_decay=0.01,
        gradient_clip_norm=1.0,
        save_dir="checkpoints",
        metric_for_best="f1_weighted",
    )

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
    )

    print("\n[6] Starting training...")
    history = trainer.train()

    print("\n[7] Training Summary:")
    for metrics in history:
        val_str = ""
        if metrics.val_metrics is not None:
            val_str = f", Val F1: {metrics.val_metrics.f1_weighted:.4f}"
        print(f"  Epoch {metrics.epoch}: Train Loss={metrics.train_loss:.4f}{val_str}")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
