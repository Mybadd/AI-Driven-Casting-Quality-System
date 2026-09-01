"""
Classification evaluation metrics for the
AI-Driven Casting Quality System.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def calculate_classification_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """
    Calculate the main classification performance metrics.

    Parameters
    ----------
    y_true:
        Actual target values.
    y_pred:
        Predicted target values.

    Returns
    -------
    dict[str, float]
        Dictionary containing accuracy, precision, recall,
        and F1-score.
    """

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label="Yes",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                pos_label="Yes",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                pos_label="Yes",
                zero_division=0,
            )
        ),
    }


def calculate_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> pd.DataFrame:
    """
    Calculate the confusion matrix.

    Rows represent actual classes.
    Columns represent predicted classes.

    Returns
    -------
    pd.DataFrame
        Labelled confusion matrix.
    """

    labels = ["No", "Yes"]

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    return pd.DataFrame(
        matrix,
        index=[
            "Actual_No",
            "Actual_Yes",
        ],
        columns=[
            "Predicted_No",
            "Predicted_Yes",
        ],
    )


def generate_classification_report(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, Any]:
    """
    Generate the detailed sklearn classification report.

    Returns
    -------
    dict
        Detailed classification metrics.
    """

    return classification_report(
        y_true,
        y_pred,
        labels=["No", "Yes"],
        output_dict=True,
        zero_division=0,
    )