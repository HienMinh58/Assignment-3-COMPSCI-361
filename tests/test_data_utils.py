import pandas as pd
import pytest

from data_utils import validate_dataset


def test_valid_dataframe_passes_validation():
    df = pd.DataFrame(
        {
            "Id": [1, 2],
            "Article": ["News article one", "News article two"],
            "Category": ["business", "sport"],
        }
    )

    validate_dataset(df)


def test_missing_required_columns_raises_value_error():
    df = pd.DataFrame(
        {
            "Id": [1],
            "Article": ["News article one"],
        }
    )

    with pytest.raises(ValueError, match="Category"):
        validate_dataset(df)
