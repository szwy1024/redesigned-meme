"""Export model computation graph for visualization with Netron.

This script exports the SocialSentimentFusionModel to ONNX format
so it can be visualized using Netron (https://netron.app).

Usage:
    uv run python scripts/export_model_graph.py
"""

import torch
import netron

from app.models.model import ModelConfig, SocialSentimentFusionModel


def export_model_to_onnx(output_path: str = "model_graph.onnx") -> None:
    """Export the model computation graph to ONNX format.

    Args:
        output_path: Path to save the ONNX model file.
    """
    # Configuration
    config = ModelConfig(
        pretrained_model_name="hfl/chinese-roberta-wwm-ext",
        social_feature_dim=10,
        social_hidden_dim=32,
        fusion_hidden_dim=128,
        num_labels=3,
        dropout_prob=0.1,
        freeze_text_encoder=True,
    )

    # Initialize model
    print("[1] Initializing model...")
    model = SocialSentimentFusionModel(config)
    model.eval()

    # Create dummy inputs matching the expected shapes
    print("[2] Creating dummy inputs...")
    batch_size = 1
    seq_length = 128
    vocab_size = 21128  # RoBERTa Chinese vocab size

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)
    social_features = torch.randn(batch_size, config.social_feature_dim)

    # Export to ONNX
    print(f"[3] Exporting model to ONNX format: {output_path}")
    torch.onnx.export(
        model,
        (input_ids, attention_mask, social_features),
        output_path,
        input_names=["input_ids", "attention_mask", "social_features"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "social_features": {0: "batch_size"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    print(f"[4] Model exported successfully to: {output_path}")
    print("\nYou can visualize the model using Netron:")
    print(f"  1. Open https://netron.app")
    print(f"  2. Upload the file: {output_path}")
    print(f"  3. Or run: netron.start('{output_path}')")

    # Start netron server to view the model
    print("\n[5] Starting Netron server to view the model...")
    netron.start(output_path, port=8088)


if __name__ == "__main__":
    # Export to backend directory (will be moved to project root later)
    export_model_to_onnx("model_graph.onnx")