"""Placeholder ANN classifier for teammates to complete later."""

from src.classifiers.base import BaseTextClassifier


class ANNTextClassifier(BaseTextClassifier):
    """Skeleton for the assignment artificial neural network implementation."""

    def fit(self, train_texts, train_labels):
        """Train the ANN classifier."""
        raise NotImplementedError("This classifier has not been implemented yet.")

    def predict(self, test_texts):
        """Predict labels with the ANN classifier."""
        raise NotImplementedError("This classifier has not been implemented yet.")

    def evaluate(self, test_texts, test_labels):
        """Evaluate ANN predictions."""
        raise NotImplementedError("This classifier has not been implemented yet.")
