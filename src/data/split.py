"""
Dataset splitting logic to create reproducible, stratified Train / Val / Test partitions
with fixed random seed = 42, eliminating data leakage.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split


def create_splits(
    samples: List[Dict[str, Any]],
    output_dir: Path,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Creates stratified Train / Val / Test partitions based on condition_label.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels = [s["condition_label"] for s in samples]

    # First split train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_samples, temp_samples, train_labels, temp_labels = train_test_split(
        samples, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed
    )

    # Second split val vs test
    relative_test_ratio = test_ratio / val_test_ratio
    val_samples, test_samples, val_labels, test_labels = train_test_split(
        temp_samples, temp_labels,
        test_size=relative_test_ratio,
        stratify=temp_labels,
        random_state=seed
    )

    # Save to disk
    for name, split_data in [("train", train_samples), ("val", val_samples), ("test", test_samples)]:
        split_path = output_dir / f"{name}.json"
        with open(split_path, "w") as f:
            json.dump({"samples": split_data}, f, indent=2)

    return train_samples, val_samples, test_samples


def load_split(split_path: Path) -> List[Dict[str, Any]]:
    """Loads split samples from JSON."""
    with open(split_path, "r") as f:
        data = json.load(f)
    return data.get("samples", data)
