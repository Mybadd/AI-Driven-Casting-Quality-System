"""
Central configuration for machine learning models.

This module defines reproducible configuration values used across
baseline and advanced model training.
"""

from typing import Final


RANDOM_STATE: Final[int] = 42


CLASSIFICATION_TARGETS: Final[tuple[str, ...]] = (
    "Defect",
    "Porosity",
    "Scrap",
)


REGRESSION_TARGETS: Final[tuple[str, ...]] = (
    "Yield",
)


BASELINE_CLASSIFICATION_MODELS: Final[tuple[str, ...]] = (
    "logistic_regression",
    "random_forest_classifier",
)


BASELINE_REGRESSION_MODELS: Final[tuple[str, ...]] = (
    "linear_regression",
    "random_forest_regressor",
)


RANDOM_FOREST_CONFIG: Final[dict[str, int]] = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
}
BALANCED_RANDOM_FOREST_CONFIG: Final[dict[str, int | str]] = {
    "n_estimators": 200,
    "max_depth": 12,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}


BALANCED_CLASSIFICATION_MODELS: Final[tuple[str, ...]] = (
    "balanced_logistic_regression",
    "balanced_random_forest_classifier",
)
