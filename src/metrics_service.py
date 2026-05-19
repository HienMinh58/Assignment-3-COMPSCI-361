"""Reusable local metrics service for classifier evaluation."""

from pathlib import Path

import pandas as pd

from src.classifier_factory import create_classifier


TEXT_COLUMN = "Article"
LABEL_COLUMN = "Category"
REQUIRED_METRIC_COLUMNS = {TEXT_COLUMN, LABEL_COLUMN}


def load_local_assignment_data(
    train_path: str | Path,
    test_path: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load local train/test CSV files and validate required metric columns."""
    train_path = Path(train_path)
    test_path = Path(test_path)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found at {train_path}. "
            "Place the file at data/train.csv."
        )
    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found at {test_path}. Place the file at data/test.csv."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    _validate_metric_columns(train_df, "training")
    _validate_metric_columns(test_df, "test")
    return train_df, test_df


def evaluate_nb(train_path: str | Path, test_path: str | Path) -> dict:
    """Fit and evaluate the Naive Bayes demo classifier on local CSV files."""
    return _evaluate_model("nb", train_path, test_path)


def evaluate_knn(train_path: str | Path, test_path: str | Path) -> dict:
    """Fit and evaluate the kNN classifier on local CSV files."""
    return _evaluate_model("knn", train_path, test_path)


def evaluate_svm(train_path: str | Path, test_path: str | Path) -> dict:
    """Fit and evaluate the SVM classifier on local CSV files."""
    return _evaluate_model("svm", train_path, test_path)


def evaluate_ann(train_path: str | Path, test_path: str | Path) -> dict:
    """Fit and evaluate the ANN classifier on local CSV files."""
    return _evaluate_model("ann", train_path, test_path)


def _evaluate_model(model_name: str, train_path: str | Path, test_path: str | Path) -> dict:
    """Fit and evaluate one classifier using the shared local metrics flow."""
    train_df, test_df = load_local_assignment_data(train_path, test_path)
    classifier = create_classifier(model_name)
    classifier.fit(train_df[TEXT_COLUMN].fillna(""), train_df[LABEL_COLUMN])
    metrics = classifier.evaluate(test_df[TEXT_COLUMN].fillna(""), test_df[LABEL_COLUMN])

    return {
        "model": model_name,
        "dataset": {
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
        },
        "metrics": metrics,
    }


def _validate_metric_columns(df: pd.DataFrame, dataset_name: str) -> None:
    """Raise a clear error if required text or label columns are missing."""
    missing_columns = REQUIRED_METRIC_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(
            f"The {dataset_name} dataset is missing required column(s): {missing}"
        )
