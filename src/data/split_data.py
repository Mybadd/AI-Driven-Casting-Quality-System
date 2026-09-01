"""
Reusable data splitting utilities for the
AI-Driven Casting Quality System.
"""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


# The five approved ML input features.
ML_FEATURES = [
    "Alloy",
    "Pour_Temp",
    "Mold_Moisture",
    "Cooling_Time",
    "Riser",
]


# Classification targets supported by the project.
CLASSIFICATION_TARGETS = [
    "Defect",
    "Porosity",
    "Scrap",
]


# Regression targets supported by the project.
REGRESSION_TARGETS = [
    "Yield",
]


SUPPORTED_TARGETS = CLASSIFICATION_TARGETS + REGRESSION_TARGETS


def validate_target(target: str) -> None:
    """
    Validate that the requested prediction target is supported.

    Parameters
    ----------
    target:
        Name of the target column.

    Raises
    ------
    ValueError:
        If the target is not supported.
    """
    if target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Unsupported target: '{target}'. "
            f"Supported targets are: {SUPPORTED_TARGETS}"
        )


def prepare_features_and_target(
    dataframe: pd.DataFrame,
    target: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select the five approved ML features and the requested target.

    Parameters
    ----------
    dataframe:
        Validated casting dataset.
    target:
        Prediction target.

    Returns
    -------
    X:
        Input feature dataframe.
    y:
        Target series.
    """
    validate_target(target)

    required_columns = ML_FEATURES + [target]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}"
        )

    X = dataframe[ML_FEATURES].copy()
    y = dataframe[target].copy()

    return X, y


def split_dataset(
    dataframe: pd.DataFrame,
    target: str,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Create a reproducible train/test split.

    Classification targets use stratified splitting to preserve
    class distribution. Regression targets use a standard random split.

    Parameters
    ----------
    dataframe:
        Validated casting dataset.
    target:
        Prediction target.
    test_size:
        Fraction of data reserved for testing.
    random_state:
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    X, y = prepare_features_and_target(
        dataframe=dataframe,
        target=target,
    )

    stratify = None

    if target in CLASSIFICATION_TARGETS:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X_train, X_test, y_train, y_test