"""Shared classifier interface for text classification models."""


class BaseTextClassifier:
    """Simple interface that every assignment classifier should follow."""

    def fit(self, train_texts, train_labels):
        """Train the classifier on text examples and labels."""
        raise NotImplementedError

    def predict(self, test_texts):
        """Predict labels for text examples."""
        raise NotImplementedError

    def evaluate(self, test_texts, test_labels):
        """Evaluate predictions against true labels."""
        raise NotImplementedError
