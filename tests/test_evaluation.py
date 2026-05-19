import pytest

from evaluation import compute_f1_score


def test_f1_score_is_between_zero_and_one():
    score = compute_f1_score(
        y_true=["business", "sport", "tech"],
        y_pred=["business", "sport", "business"],
    )

    assert 0 <= score <= 1


def test_f1_score_matches_deterministic_example():
    score = compute_f1_score(
        y_true=["business", "business", "sport", "sport"],
        y_pred=["business", "sport", "sport", "sport"],
    )

    assert score == pytest.approx(0.7333333333)
