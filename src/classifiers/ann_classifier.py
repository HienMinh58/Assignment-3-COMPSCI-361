"""Artificial neural network text classifier for BBC news classification."""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier

from src.classifiers.base import BaseTextClassifier


class _PositiveInitMLPClassifier(MLPClassifier):
    """MLPClassifier with the assignment's [0, 0.1] weight initialization."""

    def _init_coef(self, fan_in, fan_out, dtype=None):
        coef_init = self._random_state.uniform(0.0, 0.1, (fan_in, fan_out))
        intercept_init = np.zeros(fan_out)

        if dtype is not None:
            coef_init = coef_init.astype(dtype, copy=False)
            intercept_init = intercept_init.astype(dtype, copy=False)

        return coef_init, intercept_init


class ANNTextClassifier(BaseTextClassifier):
    """Single hidden-layer ANN classifier using TF-IDF features."""

    def __init__(
        self,
        hidden_units: int = 20,
        hidden_unit_options: tuple[int, ...] = (2, 5, 20, 40),
        learning_rate: float = 0.01,
        epochs: int = 100,
        max_features: int = 5000,
        random_state: int = 42,
    ) -> None:
        if hidden_units < 1:
            raise ValueError("hidden_units must be at least 1.")
        if not hidden_unit_options:
            raise ValueError("hidden_unit_options must not be empty.")
        if any(option < 1 for option in hidden_unit_options):
            raise ValueError("Every hidden-unit option must be at least 1.")
        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if max_features < 1:
            raise ValueError("max_features must be at least 1.")

        self.hidden_units = hidden_units
        self.hidden_unit_options = hidden_unit_options
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.max_features = max_features
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.model = self._create_model(hidden_units)
        self.hidden_unit_losses_: dict[int, float] = {}

    def fit(self, train_texts, train_labels) -> "ANNTextClassifier":
        """Fit TF-IDF on training texts, then train the ANN classifier."""
        X_train = self.vectorizer.fit_transform(train_texts)

        self.hidden_unit_losses_ = {}
        for hidden_units in self.hidden_unit_options:
            model = self._create_model(hidden_units)
            self._fit_model(model, X_train, train_labels)
            self.hidden_unit_losses_[hidden_units] = round(float(model.loss_), 4)

            if hidden_units == self.hidden_units:
                self.model = model

        if self.hidden_units not in self.hidden_unit_options:
            self.model = self._create_model(self.hidden_units)
            self._fit_model(self.model, X_train, train_labels)

        return self

    def predict(self, test_texts):
        """Predict labels for new texts using the fitted ANN model."""
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

    def plot_hidden_unit_losses(self, ax=None):
        """Plot training loss against hidden-unit count for Task 2(d)."""
        if not self.hidden_unit_losses_:
            raise ValueError("Fit the ANN before plotting hidden-unit losses.")

        if ax is None:
            import matplotlib.pyplot as plt

            _, ax = plt.subplots()

        hidden_units = list(self.hidden_unit_losses_)
        losses = [self.hidden_unit_losses_[hidden_unit] for hidden_unit in hidden_units]

        ax.plot(hidden_units, losses, marker="o")
        ax.set_xlabel("Number of hidden units")
        ax.set_ylabel("Binary cross-entropy loss")
        ax.set_title("ANN training loss vs hidden units")
        ax.grid(True, alpha=0.3)
        return ax

    def _create_model(self, hidden_units: int) -> MLPClassifier:
        return _PositiveInitMLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            activation="logistic",
            solver="sgd",
            learning_rate="constant",
            learning_rate_init=self.learning_rate,
            max_iter=self.epochs,
            batch_size=1,
            tol=0.0,
            n_iter_no_change=self.epochs + 1,
            random_state=self.random_state + hidden_units,
        )

    @staticmethod
    def _fit_model(model: MLPClassifier, X_train, train_labels) -> None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            model.fit(X_train, train_labels)
