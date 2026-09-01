"""
Central evaluation interface for the
AI-Driven Casting Quality System.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.evaluation.classification_metrics import (
    calculate_classification_metrics,
    calculate_confusion_matrix,
    generate_classification_report,
)
from src.evaluation.regression_metrics import (
    calculate_regression_metrics,
)
from src.models.core.model_config import (
    CLASSIFICATION_TARGETS,
    REGRESSION_TARGETS,
)


def evaluate_classification_model(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate a classification model.

    Returns the main metrics, confusion matrix,
    and detailed classification report.
    """

    metrics = calculate_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    confusion = calculate_confusion_matrix(
        y_true=y_true,
        y_pred=y_pred,
    )

    report = generate_classification_report(
        y_true=y_true,
        y_pred=y_pred,
    )

    return {
        "metrics": metrics,
        "confusion_matrix": confusion,
        "classification_report": report,
    }


def evaluate_regression_model(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate a regression model.
    """

    metrics = calculate_regression_metrics(
        y_true=y_true,
        y_pred=y_pred,
    )

    return {
        "metrics": metrics,
    }


def evaluate_model(
    target: str,
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, Any]:
    """
    Evaluate a model according to its target type.

    Classification targets use classification metrics.

    Regression targets use regression metrics.
    """

    if target in CLASSIFICATION_TARGETS:

        return evaluate_classification_model(
            y_true=y_true,
            y_pred=y_pred,
        )

    if target in REGRESSION_TARGETS:

        return evaluate_regression_model(
            y_true=y_true,
            y_pred=y_pred,
        )

    raise ValueError(
        f"Unsupported evaluation target: {target}"
    )