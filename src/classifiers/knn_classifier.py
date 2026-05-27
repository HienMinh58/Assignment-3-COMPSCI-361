"""k-nearest neighbors text classifier for BBC news article classification."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier

from src.classifiers.base import BaseTextClassifier


class KNNTextClassifier(BaseTextClassifier):
    """kNN classifier using TF-IDF features."""

    def __init__(
        self,
        n_neighbors: int = 12,
        metric: str = "euclidean",
        max_features: int = 10000,
    ) -> None:
        """
        Create a kNN classifier with configurable neighbor count and distance.

        Parameters
        ----------
        n_neighbors:
            Number of neighbors to use. The fitted model caps this at the number
            of training samples so small test fixtures still work.
        metric:
            Distance metric used by kNN. Defaults to Euclidean distance to match
            the notebook implementation.
        max_features:
            Maximum TF-IDF vocabulary size.
        """
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be at least 1.")
        if max_features < 1:
            raise ValueError("max_features must be at least 1.")

        self.n_neighbors = n_neighbors
        self.metric = metric
        self.max_features = max_features

        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model: KNeighborsClassifier | None = None

    def fit(self, train_texts, train_labels):
        """Fit TF-IDF on training texts, then train the kNN classifier."""
        X_train = self.vectorizer.fit_transform(train_texts)
        effective_neighbors = min(self.n_neighbors, X_train.shape[0])

        self.model = KNeighborsClassifier(
            n_neighbors=effective_neighbors,
            metric=self.metric,
        )
        self.model.fit(X_train, train_labels)
        return self

    def predict(self, test_texts):
        """Predict labels for new texts using the fitted vectorizer and model."""
        if self.model is None:
            raise ValueError("The kNN classifier must be fitted before prediction.")

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
