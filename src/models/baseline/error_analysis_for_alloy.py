"""
Normalized error-rate analysis by Alloy.

This experiment complements the raw error counts by calculating
false-positive and false-negative rates within each Alloy group.

The existing error-analysis script is not modified.
"""

from pathlib import Path

import pandas as pd

from src.data.load_data import load_csv
from src.data.split_data import split_dataset
from src.models.baseline.classification import (
    predict_classification,
    train_classification_model,
)
from src.models.core.model_config import CLASSIFICATION_TARGETS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"

MODEL_NAME = "balanced_logistic_regression"


def get_dataset_path() -> Path:
    csv_files = list(DATA_DIRECTORY.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset found in {DATA_DIRECTORY}"
        )

    if len(csv_files) > 1:
        raise ValueError(
            f"Expected exactly one CSV dataset in {DATA_DIRECTORY}, "
            f"found {len(csv_files)}"
        )

    return csv_files[0]


def calculate_alloy_error_rates(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: pd.Series,
) -> pd.DataFrame:
    dataframe = X_test.copy()

    dataframe["actual"] = y_test
    dataframe["prediction"] = predictions

    rows = []

    for alloy, group in dataframe.groupby("Alloy"):
        true_negative = (
            (group["actual"] == "No")
            & (group["prediction"] == "No")
        ).sum()

        false_positive = (
            (group["actual"] == "No")
            & (group["prediction"] == "Yes")
        ).sum()

        false_negative = (
            (group["actual"] == "Yes")
            & (group["prediction"] == "No")
        ).sum()

        true_positive = (
            (group["actual"] == "Yes")
            & (group["prediction"] == "Yes")
        ).sum()

        actual_negative = true_negative + false_positive
        actual_positive = false_negative + true_positive

        false_positive_rate = (
            false_positive / actual_negative
            if actual_negative > 0
            else 0.0
        )

        false_negative_rate = (
            false_negative / actual_positive
            if actual_positive > 0
            else 0.0
        )

        rows.append(
            {
                "Alloy": alloy,
                "samples": len(group),
                "actual_yes": int(actual_positive),
                "actual_no": int(actual_negative),
                "false_positive": int(false_positive),
                "false_negative": int(false_negative),
                "true_positive": int(true_positive),
                "true_negative": int(true_negative),
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
            }
        )

    return pd.DataFrame(rows).sort_values("Alloy")


def analyze_target(
    dataframe: pd.DataFrame,
    target: str,
    output_directory: Path,
) -> None:
    print("\n" + "=" * 60)
    print(f"NORMALIZED ALLOY ERROR ANALYSIS: {target}")
    print("=" * 60)

    X_train, X_test, y_train, y_test = split_dataset(
        dataframe=dataframe,
        target=target,
    )

    pipeline = train_classification_model(
        model_name=MODEL_NAME,
        X_train=X_train,
        y_train=y_train,
    )

    predictions = predict_classification(
        pipeline=pipeline,
        X_test=X_test,
    )

    results = calculate_alloy_error_rates(
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
    )

    print("\nNormalized error rates by Alloy:")
    print(results.to_string(index=False))

    target_directory = output_directory / target.lower()
    target_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = target_directory / "normalized_errors_by_alloy.csv"

    results.to_csv(
        output_path,
        index=False,
    )

    print("\nReport saved:")
    print(f"  {output_path}")


def main() -> None:
    dataset_path = get_dataset_path()

    print(f"Loading dataset: {dataset_path.name}")

    dataframe = load_csv(dataset_path)

    print(
        f"Dataset loaded successfully: "
        f"{len(dataframe)} rows"
    )

    output_directory = (
        PROJECT_ROOT
        / "reports"
        / "evaluation"
        / "error_analysis"
    )

    for target in CLASSIFICATION_TARGETS:
        analyze_target(
            dataframe=dataframe,
            target=target,
            output_directory=output_directory,
        )

    print("\n" + "=" * 60)
    print("NORMALIZED ALLOY ERROR ANALYSIS COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()