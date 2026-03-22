"""Data Preparation Script for Weibo Sentiment Dataset.

This script:
1. Loads raw CSV data from data/raw/
2. Cleans missing values and validates text/label pairs
3. Maps labels to our standard format (0=positive, 1=negative)
4. Splits into train/val/test (8:1:1)
5. Saves processed CSVs to data/processed/

Usage:
    uv run python scripts/prepare_data.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw_data(csv_path: str) -> pd.DataFrame:
    """Load raw CSV data.

    Args:
        csv_path: Path to raw CSV file.

    Returns:
        Loaded DataFrame.
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"  Loaded {len(df):,} rows")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean data by removing missing values and invalid entries.

    Args:
        df: Raw DataFrame.

    Returns:
        Cleaned DataFrame.
    """
    print("\nCleaning data...")

    initial_count = len(df)

    # Remove rows with missing values
    df = df.dropna()
    print(f"  After dropping NaN: {len(df):,} rows")

    # Ensure text is string and not empty
    df["review"] = df["review"].astype(str)
    df = df[df["review"].str.strip().str.len() > 0]
    print(f"  After removing empty text: {len(df):,} rows")

    # Ensure label is integer
    df["label"] = df["label"].astype(int)

    # Remove rows with labels not in {0, 1}
    df = df[df["label"].isin([0, 1])]
    print(f"  After filtering valid labels: {len(df):,} rows")

    print(f"  Removed {initial_count - len(df):,} invalid rows")
    return df


def map_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map Weibo labels to our standard format.

    Weibo format:
        0 = negative (负面)
        1 = positive (正面)

    Our format (to match model output):
        0 = POSITIVE (positive)
        1 = NEGATIVE (negative)

    Args:
        df: DataFrame with Weibo labels.

    Returns:
        DataFrame with mapped labels.
    """
    print("\nMapping labels...")
    print("  Weibo 0 (negative) → 1 (NEGATIVE)")
    print("  Weibo 1 (positive) → 0 (POSITIVE)")

    # Map: Weibo's 0 -> 1, Weibo's 1 -> 0
    df["label"] = df["label"].map({0: 1, 1: 0})

    print(f"  Label distribution after mapping:")
    label_counts = df["label"].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = "POSITIVE" if label == 0 else "NEGATIVE"
        print(f"    {label} ({label_name}): {count:,}")

    return df


def split_data(df: pd.DataFrame, test_size: float = 0.1, val_size: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train/val/test sets.

    Args:
        df: DataFrame to split.
        test_size: Proportion for test set.
        val_size: Proportion for validation set.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    print(f"\nSplitting data (train:{1-test_size*2:.0%}, val:{val_size:.0%}, test:{test_size:.0%})...")

    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=42,
        stratify=df["label"],
    )

    # Second split: separate val from train_val
    val_ratio = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio,
        random_state=42,
        stratify=train_val_df["label"],
    )

    print(f"  Train: {len(train_df):,} rows")
    print(f"  Val:   {len(val_df):,} rows")
    print(f"  Test:  {len(test_df):,} rows")

    return train_df, val_df, test_df


def save_processed_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save processed data to CSV files.

    Args:
        train_df: Training set.
        val_df: Validation set.
        test_df: Test set.
        output_dir: Output directory path.
    """
    print(f"\nSaving to {output_dir}...")

    os.makedirs(output_dir, exist_ok=True)

    # Rename columns to standard format
    train_df = train_df.rename(columns={"review": "text"})
    val_df = val_df.rename(columns={"review": "text"})
    test_df = test_df.rename(columns={"review": "text"})

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"  Saved train.csv ({len(train_df):,} rows)")
    print(f"  Saved val.csv ({len(val_df):,} rows)")
    print(f"  Saved test.csv ({len(test_df):,} rows)")


def main() -> None:
    """Main execution function."""
    print("=" * 60)
    print("Weibo Sentiment Data Preparation")
    print("=" * 60)

    # Paths
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    raw_data_path = project_dir / "data" / "raw" / "weibo_senti_100k.csv"
    processed_dir = project_dir / "data" / "processed"

    # Check raw data exists
    if not raw_data_path.exists():
        print(f"ERROR: Raw data not found at {raw_data_path}")
        sys.exit(1)

    # Load
    df = load_raw_data(str(raw_data_path))

    # Clean
    df = clean_data(df)

    # Map labels (Weibo -> our format)
    df = map_labels(df)

    # Split
    train_df, val_df, test_df = split_data(df, test_size=0.1, val_size=0.1)

    # Save
    save_processed_data(train_df, val_df, test_df, str(processed_dir))

    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"  1. Train model: uv run python scripts/train.py")
    print(f"  2. Start API:   uv run uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()
