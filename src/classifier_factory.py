"""Factory for creating assignment text classifiers by name."""

from src.classifiers import (
    ANNTextClassifier,
    KNNTextClassifier,
    NaiveBayesTextClassifier,
    SVMTextClassifier,
)
from src.classifiers.base import BaseTextClassifier


SUPPORTED_MODEL_NAMES = ("nb", "knn", "svm", "ann")


def create_classifier(model_name: str) -> BaseTextClassifier:
    """Create a classifier instance for a supported model name."""
    normalized_name = model_name.strip().lower()

    if normalized_name == "nb":
        return NaiveBayesTextClassifier()
    if normalized_name == "knn":
        return KNNTextClassifier()
    if normalized_name == "svm":
        return SVMTextClassifier()
    if normalized_name == "ann":
        return ANNTextClassifier()

    supported = ", ".join(SUPPORTED_MODEL_NAMES)
    raise ValueError(
        f"Unsupported model name '{model_name}'. Supported models: {supported}."
    )
