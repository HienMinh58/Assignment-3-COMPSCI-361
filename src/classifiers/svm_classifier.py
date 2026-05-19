"""Placeholder SVM classifier for teammates to complete later."""

from src.classifiers.base import BaseTextClassifier


class SVMTextClassifier(BaseTextClassifier):
    """Skeleton for the assignment support vector machine implementation."""

    def fit(self, train_texts, train_labels):
        """Train the SVM classifier."""
        raise NotImplementedError("This classifier has not been implemented yet.")

    def predict(self, test_texts):
        """Predict labels with the SVM classifier."""
        raise NotImplementedError("This classifier has not been implemented yet.")

    def evaluate(self, test_texts, test_labels):
        """Evaluate SVM predictions."""
        raise NotImplementedError("This classifier has not been implemented yet.")
