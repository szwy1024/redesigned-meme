"""Test script for SentimentDataset and DataLoader integration."""

from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from app.dataset.dataset import (
    SentimentDataCollator,
    SentimentDataset,
    SentimentLabel,
    SentimentSample,
    create_dummy_data,
)


def main() -> None:
    print("=" * 60)
    print("SentimentDataset Test")
    print("=" * 60)

    # Load a lightweight tokenizer for testing
    print("\n[1] Loading tokenizer (hfl/chinese-roberta-wwm-ext)...")
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    print(f"    Vocab size: {tokenizer.vocab_size}")

    # Create dummy data
    print("\n[2] Creating dummy dataset (100 samples)...")
    samples = create_dummy_data(num_samples=100)
    print(f"    Created {len(samples)} samples")

    # Initialize dataset
    print("\n[3] Initializing SentimentDataset...")
    dataset = SentimentDataset(
        data=samples,
        tokenizer=tokenizer,
        max_length=128,
    )
    print(f"    Dataset length: {len(dataset)}")
    print(f"    Max length: {dataset.max_length}")

    # Test single item access
    print("\n[4] Testing single item access...")
    sample = dataset[0]
    print(f"    input_ids shape: {sample['input_ids'].shape}")
    print(f"    attention_mask shape: {sample['attention_mask'].shape}")
    print(f"    label: {sample['label'].item()}")

    # Test with DataLoader
    print("\n[5] Testing DataLoader integration...")
    collator = SentimentDataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        collate_fn=collator,
    )
    print(f"    Number of batches: {len(dataloader)}")

    # Iterate through first batch
    print("\n[6] Iterating first batch...")
    first_batch = next(iter(dataloader))
    print(f"    input_ids shape: {first_batch['input_ids'].shape}")
    print(f"    attention_mask shape: {first_batch['attention_mask'].shape}")
    print(f"    label shape: {first_batch['label'].shape}")
    print(f"    label values: {first_batch['label'].tolist()}")

    # Test label encoding
    print("\n[7] Testing SentimentLabel encoding...")
    for label_str in ["positive", "negative", "neutral", "0", "1", "2"]:
        label = SentimentLabel.from_string(label_str)
        print(f"    '{label_str}' -> {label.name} ({label.value})")

    # Test with dict input format
    print("\n[8] Testing dict input format...")
    dict_data: list[dict[str, Any]] = [
        {"text": "太开心了！", "label": 0},
        {"text": "很差的产品", "label": 1},
        {"text": "一般般", "label": 2},
    ]
    dict_dataset = SentimentDataset(
        data=dict_data,
        tokenizer=tokenizer,
        max_length=64,
    )
    for i in range(len(dict_dataset)):
        item = dict_dataset[i]
        print(f"    Sample {i}: label={item['label'].item()}")

    # Decode a sample to verify tokenization
    print("\n[9] Decoding sample to verify tokenization...")
    decoded = tokenizer.decode(first_batch["input_ids"][0], skip_special_tokens=True)
    print(f"    Decoded text: {decoded}")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
