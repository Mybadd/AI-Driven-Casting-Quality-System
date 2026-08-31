from typing import Any

import pandas as pd


def get_dataset_summary(
    dataframe: pd.DataFrame,
) -> dict[str, Any]:
    """
    Generate a general summary of a dataset.
    """

    return {
        "rows": dataframe.shape[0],
        "columns": dataframe.shape[1],
        "column_names": dataframe.columns.tolist(),
        "data_types": {
            column: str(dtype)
            for column, dtype in dataframe.dtypes.items()
        },
        "duplicate_rows": int(dataframe.duplicated().sum()),
    }


def get_missing_values(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    """
    Count missing values for every column.
    """

    return {
        column: int(count)
        for column, count
        in dataframe.isnull().sum().items()
    }


def get_unique_value_counts(
    dataframe: pd.DataFrame,
) -> dict[str, int]:
    """
    Count unique values for every column.
    """

    return {
        column: int(dataframe[column].nunique())
        for column in dataframe.columns
    }


def get_numeric_statistics(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, float]]:
    """
    Generate descriptive statistics for numeric columns.
    """

    numeric_columns = dataframe.select_dtypes(
        include="number"
    ).columns

    statistics = {}

    for column in numeric_columns:
        statistics[column] = {
            "minimum": float(dataframe[column].min()),
            "maximum": float(dataframe[column].max()),
            "mean": float(dataframe[column].mean()),
            "median": float(dataframe[column].median()),
            "standard_deviation": float(
                dataframe[column].std()
            ),
        }

    return statistics


def get_categorical_statistics(
    dataframe: pd.DataFrame,
) -> dict[str, dict[str, int]]:
    """
    Generate value counts for categorical columns.
    """

    categorical_columns = dataframe.select_dtypes(
        exclude="number"
    ).columns

    statistics = {}

    for column in categorical_columns:

        value_counts = dataframe[column].value_counts(
            dropna=False
        )

        statistics[column] = {
            str(value): int(count)
            for value, count in value_counts.items()
        }

    return statistics