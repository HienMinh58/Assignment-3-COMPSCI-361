"""Utilities for loading and validating the BBC news dataset."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"Id", "Article", "Category"}


def validate_dataset(df: pd.DataFrame) -> None:
    """Validate that a dataframe has the columns required by the project.

    Args:
        df: Dataset to validate.

    Raises:
        ValueError: If one or more required columns are missing.
    """
    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a CSV dataset from disk and validate its required columns."""
    df = pd.read_csv(path)
    validate_dataset(df)
    return df
