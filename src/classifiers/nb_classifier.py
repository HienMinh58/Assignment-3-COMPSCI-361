"""Naive Bayes text classifier for BBC news article classification."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import MultinomialNB

from src.classifiers.base import BaseTextClassifier


class NaiveBayesTextClassifier(BaseTextClassifier):
    """Multinomial Naive Bayes classifier using TF-IDF features."""

    def __init__(self, alpha: float = 0.1) -> None:
        """
        Create a Naive Bayes classifier with configurable smoothing.

        Parameters
        ----------
        alpha:
            Additive smoothing parameter. Defaults to 0.1 to match the
            notebook's final cross-validated model.
        """
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer()
        self.model = MultinomialNB(alpha=alpha)

    def fit(self, train_texts, train_labels) -> "NaiveBayesTextClassifier":
        """Fit TF-IDF on training texts, then train MultinomialNB."""
        X_train = self.vectorizer.fit_transform(train_texts)
        self.model.fit(X_train, train_labels)
        return self

    def predict(self, test_texts):
        """Predict labels for new texts using the fitted vectorizer and model."""
        X_test = self.vectorizer.transform(test_texts)
        return self.model.predict(X_test)

    def evaluate(self, test_texts, test_labels) -> dict[str, float]:
        """Evaluate model predictions with accuracy and F1 scores."""
        predictions = self.predict(test_texts)

        return {
            "accuracy": round(float(accuracy_score(test_labels, predictions)), 4),
            "f1_macro": round(
                float(f1_score(test_labels, predictions, average="macro")),
                4,
            ),
            "f1_weighted": round(
                float(f1_score(test_labels, predictions, average="weighted")),
                4,
            ),
        }
