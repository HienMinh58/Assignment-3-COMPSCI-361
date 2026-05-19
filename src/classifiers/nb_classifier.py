"""Reference Naive Bayes text classifier implemented from scratch.

This file is the demo implementation for teammates. The other classifiers
should eventually follow the same public fit/predict/evaluate interface.
"""

from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score

from src.classifiers.base import BaseTextClassifier


class NaiveBayesTextClassifier(BaseTextClassifier):
    """From-scratch Multinomial Naive Bayes classifier for BBC news articles."""

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Create a Naive Bayes classifier with configurable Laplace smoothing.

        Parameters
        ----------
        alpha:
            Smoothing parameter. alpha=1.0 corresponds to Laplace smoothing.
        """
        self.alpha = alpha
        self.vectorizer = TfidfVectorizer()

        self.classes_: np.ndarray | None = None
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None

    def fit(self, train_texts, train_labels) -> "NaiveBayesTextClassifier":
        """
        Fit TF-IDF on training texts, then train Naive Bayes from scratch.

        Training computes:
        - log class priors: log P(class)
        - log feature likelihoods: log P(feature | class)
        """
        X_train = self.vectorizer.fit_transform(train_texts)
        y_train = np.asarray(train_labels)

        self.classes_, class_counts = np.unique(y_train, return_counts=True)

        n_samples = X_train.shape[0]
        n_features = X_train.shape[1]
        n_classes = len(self.classes_)

        # log P(class)
        self.class_log_prior_ = np.log(class_counts / n_samples)

        # Store feature sums for each class
        feature_counts = np.zeros((n_classes, n_features), dtype=float)

        for class_index, class_label in enumerate(self.classes_):
            class_rows = X_train[y_train == class_label]
            feature_counts[class_index, :] = np.asarray(
                class_rows.sum(axis=0)
            ).ravel()

        # Laplace smoothing
        smoothed_feature_counts = feature_counts + self.alpha

        # Total feature mass per class after smoothing
        smoothed_class_totals = smoothed_feature_counts.sum(axis=1)

        # log P(feature | class)
        self.feature_log_prob_ = np.log(
            smoothed_feature_counts / smoothed_class_totals[:, np.newaxis]
        )

        return self

    def predict(self, test_texts):
        """
        Predict labels for new texts using the fitted vectorizer and NB model.
        """
        if (
            self.classes_ is None
            or self.class_log_prior_ is None
            or self.feature_log_prob_ is None
        ):
            raise ValueError("The Naive Bayes classifier must be fitted before prediction.")

        X_test = self.vectorizer.transform(test_texts)

        # For each document and class:
        # score = log P(class) + sum(feature_value * log P(feature | class))
        class_scores = X_test @ self.feature_log_prob_.T
        class_scores = class_scores + self.class_log_prior_

        predicted_class_indices = np.asarray(class_scores).argmax(axis=1)

        return self.classes_[predicted_class_indices]

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