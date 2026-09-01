"""
Utilities for saving machine learning evaluation results.

The module stores metrics, confusion matrices, and detailed
classification reports in a consistent project structure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

EVALUATION_REPORT_DIRECTORY = (
    PROJECT_ROOT
    / "reports"
    / "evaluation"
)

METRICS_DIRECTORY = (
    EVALUATION_REPORT_DIRECTORY
    / "metrics"
)

CONFUSION_MATRIX_DIRECTORY = (
    EVALUATION_REPORT_DIRECTORY
    / "confusion_matrices"
)

DETAILED_REPORT_DIRECTORY = (
    EVALUATION_REPORT_DIRECTORY
    / "detailed_reports"
)


def create_report_directories() -> None:
    """
    Create all directories required for evaluation reports.
    """

    directories = [
        METRICS_DIRECTORY,
        CONFUSION_MATRIX_DIRECTORY,
        DETAILED_REPORT_DIRECTORY,
    ]

    for directory in directories:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


def save_metrics(
    target: str,
    model_name: str,
    metrics: dict[str, float],
) -> Path:
    """
    Save model metrics as a JSON file.
    """

    create_report_directories()

    file_name = (
        f"{target.lower()}__"
        f"{model_name}.json"
    )

    file_path = (
        METRICS_DIRECTORY
        / file_name
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=4,
        )

    return file_path


def save_confusion_matrix(
    target: str,
    model_name: str,
    confusion: pd.DataFrame,
) -> Path:
    """
    Save a labelled confusion matrix as CSV.
    """

    create_report_directories()

    file_name = (
        f"{target.lower()}__"
        f"{model_name}.csv"
    )

    file_path = (
        CONFUSION_MATRIX_DIRECTORY
        / file_name
    )

    confusion.to_csv(
        file_path,
        index=True,
    )

    return file_path


def save_classification_report(
    target: str,
    model_name: str,
    report: dict[str, Any],
) -> Path:
    """
    Save the detailed classification report as JSON.
    """

    create_report_directories()

    file_name = (
        f"{target.lower()}__"
        f"{model_name}.json"
    )

    file_path = (
        DETAILED_REPORT_DIRECTORY
        / file_name
    )

    with file_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            report,
            file,
            indent=4,
        )

    return file_path


def save_evaluation_result(
    target: str,
    model_name: str,
    evaluation_result: dict[str, Any],
) -> dict[str, Path]:
    """
    Save all available evaluation artifacts.

    Classification results save:
    - metrics
    - confusion matrix
    - detailed classification report

    Regression results save:
    - metrics only
    """

    saved_paths: dict[str, Path] = {}

    metrics_path = save_metrics(
        target=target,
        model_name=model_name,
        metrics=evaluation_result["metrics"],
    )

    saved_paths["metrics"] = metrics_path

    if "confusion_matrix" in evaluation_result:

        confusion_path = save_confusion_matrix(
            target=target,
            model_name=model_name,
            confusion=evaluation_result[
                "confusion_matrix"
            ],
        )

        saved_paths[
            "confusion_matrix"
        ] = confusion_path

    if "classification_report" in evaluation_result:

        report_path = save_classification_report(
            target=target,
            model_name=model_name,
            report=evaluation_result[
                "classification_report"
            ],
        )

        saved_paths[
            "classification_report"
        ] = report_path

    return saved_paths