"""Placeholder kNN classifier for teammates to complete later."""

from src.classifiers.base import BaseTextClassifier


class KNNTextClassifier(BaseTextClassifier):
    """Skeleton for the assignment k-nearest neighbors implementation."""

    def fit(self, train_texts, train_labels):
        """Train the kNN classifier."""
        raise NotImplementedError("This classifier has not been implemented yet.")

    def predict(self, test_texts):
        """Predict labels with the kNN classifier."""
        raise NotImplementedError("This classifier has not been implemented yet.")

    def evaluate(self, test_texts, test_labels):
        """Evaluate kNN predictions."""
        raise NotImplementedError("This classifier has not been implemented yet.")
