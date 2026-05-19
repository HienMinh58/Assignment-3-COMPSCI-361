"""Backward-compatible wrapper around the Naive Bayes metrics service."""

from pathlib import Path

from src.metrics_service import evaluate_nb, load_local_assignment_data


def load_train_test_data(train_path: str | Path, test_path: str | Path):
    """Load and validate local train and test CSV datasets."""
    return load_local_assignment_data(train_path, test_path)


def compute_baseline_metrics(train_path: str | Path, test_path: str | Path) -> dict:
    """Compute local Naive Bayes metrics."""
    return evaluate_nb(train_path, test_path)
