from pathlib import Path

import pandas as pd
import pytest

from src.baseline_metrics import compute_baseline_metrics


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


def test_baseline_evaluation_returns_nb_structure(tmp_path):
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    _write_dataset(train_path)
    _write_dataset(test_path)

    result = compute_baseline_metrics(train_path, test_path)

    assert result["model"] == "nb"
    assert result["dataset"] == {"train_rows": 4, "test_rows": 4}
    assert set(result["metrics"]) == {"accuracy", "f1_macro", "f1_weighted"}


def test_missing_input_files_raise_clear_error(tmp_path):
    train_path = tmp_path / "missing_train.csv"
    test_path = tmp_path / "missing_test.csv"

    with pytest.raises(FileNotFoundError, match="Training dataset not found"):
        compute_baseline_metrics(train_path, test_path)
