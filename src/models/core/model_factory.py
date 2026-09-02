"""
Factory functions for creating machine learning models.

This module centralises model creation so that training modules do not
need to directly construct model instances.
"""

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
)

from src.models.core.model_config import (
    BALANCED_RANDOM_FOREST_CONFIG,
    RANDOM_FOREST_CONFIG,
    RANDOM_STATE,
)


def create_classification_model(model_name: str):
    """
    Create and return a classification model.
    """

    if model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
        )

    if model_name == "balanced_logistic_regression":
        return LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

    if model_name == "random_forest_classifier":
        return RandomForestClassifier(
            **RANDOM_FOREST_CONFIG,
        )

    if model_name == "balanced_random_forest_classifier":
        return RandomForestClassifier(
            **BALANCED_RANDOM_FOREST_CONFIG,
        )

    raise ValueError(
        f"Unsupported classification model: {model_name}"
    )

def create_regression_model(model_name: str):
    """
    Create and return a regression model.

    Parameters
    ----------
    model_name : str
        Name of the regression model.

    Returns
    -------
    sklearn estimator
        Configured regression estimator.

    Raises
    ------
    ValueError
        If the requested model is not supported.
    """

    if model_name == "linear_regression":
        return LinearRegression()

    if model_name == "random_forest_regressor":
        return RandomForestRegressor(
            **RANDOM_FOREST_CONFIG,
        )

    raise ValueError(
        f"Unsupported regression model: {model_name}"
    )