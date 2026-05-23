from pathlib import Path

import pandas as pd
import pytest

from src.metrics_service import evaluate_ann, evaluate_knn, evaluate_nb, evaluate_svm


def _write_dataset(path: Path) -> None:
    df = pd.DataFrame(
        {
            "Id": [1, 2, 3, 4],
            "Article": [
                "team wins football match",
                "government election policy",
                "football team scores goal",
                "minister announces policy",
            ],
            "Category": ["sport", "politics", "sport", "politics"],
        }
    )
    df.to_csv(path, index=False)


def test_evaluate_nb_returns_expected_structure(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_dataset(train_path)
    _write_dataset(test_path)

    result = evaluate_nb(train_path, test_path)

    assert result["model"] == "nb"
    assert result["dataset"] == {"train_rows": 4, "test_rows": 4}
    assert set(result["metrics"]) == {"accuracy", "f1_macro", "f1_weighted"}
    for value in result["metrics"].values():
        assert isinstance(value, float)
        assert 0 <= value <= 1


def test_evaluate_svm_returns_expected_structure(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_dataset(train_path)
    _write_dataset(test_path)

    result = evaluate_svm(train_path, test_path)

    assert result["model"] == "svm"
    assert result["dataset"] == {"train_rows": 4, "test_rows": 4}
    assert set(result["metrics"]) == {"accuracy", "f1_macro", "f1_weighted"}
    for value in result["metrics"].values():
        assert isinstance(value, float)
        assert 0 <= value <= 1


@pytest.mark.parametrize(
    "evaluate_function",
    [evaluate_knn, evaluate_ann],
)
def test_placeholder_evaluators_raise_not_implemented(evaluate_function, tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_dataset(train_path)
    _write_dataset(test_path)

    with pytest.raises(NotImplementedError, match="not been implemented"):
        evaluate_function(train_path, test_path)


def test_missing_csv_file_raises_clear_error(tmp_path):
    train_path = tmp_path / "missing_train.csv"
    test_path = tmp_path / "missing_test.csv"

    with pytest.raises(FileNotFoundError, match="Training dataset not found"):
        evaluate_nb(train_path, test_path)


def test_missing_required_column_raises_clear_error(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    pd.DataFrame({"Article": ["hello"]}).to_csv(train_path, index=False)
    _write_dataset(test_path)

    with pytest.raises(ValueError, match="Category"):
        evaluate_nb(train_path, test_path)
