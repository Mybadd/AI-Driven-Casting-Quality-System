"""
Regression evaluation metrics for the
AI-Driven Casting Quality System.
"""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def calculate_regression_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """
    Calculate the main regression performance metrics.

    Parameters
    ----------
    y_true:
        Actual target values.

    y_pred:
        Predicted target values.

    Returns
    -------
    dict[str, float]
        Dictionary containing MAE, RMSE, and R² score.
    """

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = root_mean_squared_error(
        y_true,
        y_pred,
    )

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2_score": float(r2),
    }