"""
Error analysis for the class-balanced Logistic Regression classifiers.

This experiment analyzes false positives and false negatives on the fixed
test split and examines whether errors are concentrated by alloy or process
conditions.

The original baseline and class-balanced training code are not modified.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import confusion_matrix

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

NUMERICAL_FEATURES = [
    "Pour_Temp",
    "Mold_Moisture",
    "Cooling_Time",
    "Riser",
]


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


def calculate_error_summary(
    y_true: pd.Series,
    predictions: pd.Series,
) -> dict:
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=["No", "Yes"],
    )

    true_negative, false_positive, false_negative, true_positive = (
        matrix.ravel()
    )

    total = matrix.sum()

    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if (false_positive + true_negative) > 0
        else 0.0
    )

    false_negative_rate = (
        false_negative / (false_negative + true_positive)
        if (false_negative + true_positive) > 0
        else 0.0
    )

    return {
        "true_negative": int(true_negative),
        "false_positive": int(false_positive),
        "false_negative": int(false_negative),
        "true_positive": int(true_positive),
        "total": int(total),
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
    }


def create_prediction_dataframe(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    predictions: pd.Series,
) -> pd.DataFrame:
    result = X_test.copy()

    result["actual"] = y_test
    result["prediction"] = predictions

    result["error_type"] = "Correct"

    false_positive_mask = (
        (result["actual"] == "No")
        & (result["prediction"] == "Yes")
    )

    false_negative_mask = (
        (result["actual"] == "Yes")
        & (result["prediction"] == "No")
    )

    true_positive_mask = (
        (result["actual"] == "Yes")
        & (result["prediction"] == "Yes")
    )

    true_negative_mask = (
        (result["actual"] == "No")
        & (result["prediction"] == "No")
    )

    result.loc[false_positive_mask, "error_type"] = "False Positive"
    result.loc[false_negative_mask, "error_type"] = "False Negative"
    result.loc[true_positive_mask, "error_type"] = "True Positive"
    result.loc[true_negative_mask, "error_type"] = "True Negative"

    return result


def analyze_alloy_errors(
    prediction_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    grouped = (
        prediction_dataframe
        .groupby(["Alloy", "error_type"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    return grouped


def analyze_process_values(
    prediction_dataframe: pd.DataFrame,
) -> pd.DataFrame:
    groups = []

    for error_type in [
        "True Negative",
        "False Positive",
        "False Negative",
        "True Positive",
    ]:
        subset = prediction_dataframe[
            prediction_dataframe["error_type"] == error_type
        ]

        if subset.empty:
            continue

        means = subset[NUMERICAL_FEATURES].mean()

        row = {
            "error_type": error_type,
            "count": len(subset),
        }

        for feature in NUMERICAL_FEATURES:
            row[f"{feature}_mean"] = means[feature]

        groups.append(row)

    return pd.DataFrame(groups)


def analyze_target(
    dataframe: pd.DataFrame,
    target: str,
    output_directory: Path,
) -> None:
    print("\n" + "=" * 60)
    print(f"ERROR ANALYSIS TARGET: {target}")
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

    summary = calculate_error_summary(
        y_true=y_test,
        predictions=predictions,
    )

    print("\nConfusion matrix:")
    print(
        pd.DataFrame(
            [
                [summary["true_negative"], summary["false_positive"]],
                [summary["false_negative"], summary["true_positive"]],
            ],
            index=["Actual No", "Actual Yes"],
            columns=["Predicted No", "Predicted Yes"],
        )
    )

    print("\nError summary:")
    print(f"  False positives: {summary['false_positive']}")
    print(f"  False negatives: {summary['false_negative']}")
    print(
        f"  False positive rate: "
        f"{summary['false_positive_rate']:.4f}"
    )
    print(
        f"  False negative rate: "
        f"{summary['false_negative_rate']:.4f}"
    )

    prediction_dataframe = create_prediction_dataframe(
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
    )

    alloy_results = analyze_alloy_errors(
        prediction_dataframe=prediction_dataframe,
    )

    process_results = analyze_process_values(
        prediction_dataframe=prediction_dataframe,
    )

    print("\nError counts by Alloy:")
    print(alloy_results.to_string(index=False))

    print("\nMean process values by prediction outcome:")
    print(process_results.to_string(index=False))

    target_directory = output_directory / target.lower()
    target_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = target_directory / "summary.csv"
    alloy_path = target_directory / "errors_by_alloy.csv"
    process_path = target_directory / "process_value_summary.csv"
    predictions_path = target_directory / "test_predictions_with_error_type.csv"

    pd.DataFrame([summary]).to_csv(
        summary_path,
        index=False,
    )

    alloy_results.to_csv(
        alloy_path,
        index=False,
    )

    process_results.to_csv(
        process_path,
        index=False,
    )

    prediction_dataframe.to_csv(
        predictions_path,
        index=False,
    )

    print("\nReports saved:")
    print(f"  summary: {summary_path}")
    print(f"  errors by alloy: {alloy_path}")
    print(f"  process values: {process_path}")
    print(f"  predictions: {predictions_path}")


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
    print("ERROR ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()  