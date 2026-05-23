import pytest

from src.classifiers import (
    ANNTextClassifier,
    KNNTextClassifier,
    NaiveBayesTextClassifier,
    SVMTextClassifier,
)


TRAIN_TEXTS = [
    "football team wins match",
    "team scores football goal",
    "coach celebrates sport win",
    "government announces election policy",
    "minister debates government policy",
    "election campaign discusses law",
    "new phone software technology",
    "computer chip technology update",
]
TRAIN_LABELS = [
    "sport",
    "sport",
    "sport",
    "politics",
    "politics",
    "politics",
    "tech",
    "tech",
]
TEST_TEXTS = [
    "football goal win",
    "government election speech",
    "technology software update",
]
TEST_LABELS = ["sport", "politics", "tech"]


@pytest.mark.parametrize(
    "classifier_cls",
    [
        NaiveBayesTextClassifier,
        SVMTextClassifier,
    ],
)
def test_implemented_classifier_fit_predict_evaluate_interface(classifier_cls):
    classifier = classifier_cls()

    classifier.fit(TRAIN_TEXTS, TRAIN_LABELS)
    predictions = classifier.predict(TEST_TEXTS)
    metrics = classifier.evaluate(TEST_TEXTS, TEST_LABELS)

    assert len(predictions) == len(TEST_TEXTS)
    assert set(metrics) == {"accuracy", "f1_macro", "f1_weighted"}
    for value in metrics.values():
        assert isinstance(value, float)
        assert 0 <= value <= 1


@pytest.mark.parametrize(
    "classifier_cls",
    [
        KNNTextClassifier,
        ANNTextClassifier,
    ],
)
def test_placeholder_classifiers_raise_not_implemented(classifier_cls):
    classifier = classifier_cls()

    with pytest.raises(NotImplementedError, match="not been implemented"):
        classifier.fit(TRAIN_TEXTS, TRAIN_LABELS)
