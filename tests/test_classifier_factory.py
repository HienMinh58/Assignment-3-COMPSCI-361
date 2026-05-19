import pytest

from src.classifier_factory import create_classifier
from src.classifiers import (
    ANNTextClassifier,
    KNNTextClassifier,
    NaiveBayesTextClassifier,
    SVMTextClassifier,
)


@pytest.mark.parametrize(
        ("model_name", "expected_type"),
        [
        ("nb", NaiveBayesTextClassifier),
        ("knn", KNNTextClassifier),
        ("svm", SVMTextClassifier),
        ("ann", ANNTextClassifier),
    ],
)
def test_create_classifier_returns_expected_type(model_name, expected_type):
    classifier = create_classifier(model_name)

    assert isinstance(classifier, expected_type)


def test_create_classifier_raises_for_unknown_model():
    with pytest.raises(ValueError, match="Unsupported model name"):
        create_classifier("unknown_model")
