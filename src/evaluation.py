"""Evaluation helpers for comparing model predictions."""

from sklearn.metrics import f1_score


def compute_f1_score(y_true, y_pred) -> float:
    """Compute weighted F1 score for multi-class classification results."""
    return f1_score(y_true, y_pred, average="weighted")
