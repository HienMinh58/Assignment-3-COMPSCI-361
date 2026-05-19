"""Evaluation helpers for comparing model predictions."""

from collections import Counter


def compute_f1_score(y_true, y_pred) -> float:
    """Compute weighted F1 score for multi-class classification results."""
    return _weighted_f1(list(y_true), list(y_pred))


def compute_classification_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute accuracy, macro F1, and weighted F1 from scratch."""
    y_true = list(y_true)
    y_pred = list(y_pred)
    labels = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    correct = sum(true == pred for true, pred in zip(y_true, y_pred))
    supports = Counter(y_true)

    f1_values = {
        label: _f1_for_label(label, y_true, y_pred)
        for label in labels
    }
    macro_f1 = sum(f1_values.values()) / len(labels) if labels else 0.0
    weighted_f1 = _weighted_f1(y_true, y_pred)

    return {
        "accuracy": round(correct / total if total else 0.0, 4),
        "f1_macro": round(macro_f1, 4),
        "f1_weighted": round(weighted_f1, 4),
    }


def _f1_for_label(label, y_true, y_pred) -> float:
    true_positive = sum(
        true == label and pred == label
        for true, pred in zip(y_true, y_pred)
    )
    false_positive = sum(
        true != label and pred == label
        for true, pred in zip(y_true, y_pred)
    )
    false_negative = sum(
        true == label and pred != label
        for true, pred in zip(y_true, y_pred)
    )

    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _weighted_f1(y_true, y_pred) -> float:
    labels = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    supports = Counter(y_true)
    if total == 0:
        return 0.0
    return sum(
        _f1_for_label(label, y_true, y_pred) * supports[label]
        for label in labels
    ) / total
