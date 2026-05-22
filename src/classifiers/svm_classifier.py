"""Support Vector Machine text classifier for BBC news article classification."""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from src.classifiers.base import BaseTextClassifier


class SVMTextClassifier(BaseTextClassifier):
    """Support Vector Machine classifier for BBC news articles using RBF kernel."""

    def __init__(self, kernel: str = 'rbf', C: float = 1.0, gamma: str = 'scale') -> None:
        """
        Create an SVM classifier with configurable kernel and hyperparameters.

        Parameters
        ----------
        kernel:
            SVM kernel type. Options: 'linear', 'rbf', 'poly', 'sigmoid'.
            Default: 'rbf' for non-linear classification.
        C:
            Regularization parameter. Higher values mean stricter fitting.
            Default: 1.0
        gamma:
            Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
            Default: 'scale' (1 / (n_features * X.var()))
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        
        self.vectorizer = CountVectorizer(max_features=5000, stop_words='english')
        self.model = SVC(kernel=kernel, C=C, gamma=gamma)

    def fit(self, train_texts, train_labels) -> "SVMTextClassifier":
        """
        Fit the CountVectorizer and train the SVM model on training texts.

        Parameters
        ----------
        train_texts:
            List of training text documents.
        train_labels:
            List of training labels corresponding to texts.
        """
        X_train = self.vectorizer.fit_transform(train_texts)
        self.model.fit(X_train, train_labels)
        return self

    def predict(self, test_texts):
        """
        Predict labels for new texts using the fitted vectorizer and SVM model.

        Parameters
        ----------
        test_texts:
            List of test text documents to classify.

        Returns
        -------
        Predicted labels for the test texts.
        """
        X_test = self.vectorizer.transform(test_texts)
        return self.model.predict(X_test)

    def evaluate(self, test_texts, test_labels) -> dict[str, float]:
        """
        Evaluate SVM model predictions with accuracy and F1 scores.

        Parameters
        ----------
        test_texts:
            List of test text documents.
        test_labels:
            List of true labels for test documents.

        Returns
        -------
        Dictionary containing accuracy, f1_macro, and f1_weighted scores.
        """
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
