"""
Tune classification probability thresholds for class-balanced models.

The original baseline and class-balanced training scripts are not modified.
This experiment evaluates different decision thresholds using predicted
probabilities and compares precision, recall, and F1-score for the
minority "Yes" class.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.data.load_data import load_csv
from src.data.split_data import split_dataset
from src.models.baseline.classification import train_classification_model
from src.models.core.model_config import CLASSIFICATION_TARGETS


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"

MODEL_NAME = "balanced_logistic_regression"

THRESHOLDS = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
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


def calculate_threshold_metrics(
    y_true: pd.Series,
    probabilities,
    threshold: float,
) -> dict:
    predictions = pd.Series(
        ["Yes" if probability >= threshold else "No"
         for probability in probabilities],
        index=y_true.index,
    )

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true,
            predictions,
            pos_label="Yes",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            pos_label="Yes",
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            predictions,
            pos_label="Yes",
            zero_division=0,
        ),
    }


def tune_target(dataframe: pd.DataFrame, target: str) -> None:
    print("=" * 60)
    print(f"THRESHOLD TUNING TARGET: {target}")
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

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    results = []

    for threshold in THRESHOLDS:
        metrics = calculate_threshold_metrics(
            y_true=y_test,
            probabilities=probabilities,
            threshold=threshold,
        )
        results.append(metrics)

        print(
            f"Threshold: {threshold:.2f} | "
            f"Accuracy: {metrics['accuracy']:.4f} | "
            f"Precision: {metrics['precision']:.4f} | "
            f"Recall: {metrics['recall']:.4f} | "
            f"F1: {metrics['f1_score']:.4f}"
        )

    results_dataframe = pd.DataFrame(results)

    best_row = results_dataframe.loc[
        results_dataframe["f1_score"].idxmax()
    ]

    print("\nBest threshold by F1-score:")
    print(f"  threshold: {best_row['threshold']:.2f}")
    print(f"  accuracy:  {best_row['accuracy']:.4f}")
    print(f"  precision: {best_row['precision']:.4f}")
    print(f"  recall:    {best_row['recall']:.4f}")
    print(f"  f1_score:  {best_row['f1_score']:.4f}")

    output_directory = (
        PROJECT_ROOT
        / "reports"
        / "evaluation"
        / "threshold_tuning"
        / target.lower()
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory
        / f"{target.lower()}__{MODEL_NAME}__thresholds.csv"
    )

    results_dataframe.to_csv(output_path, index=False)

    print(f"\nThreshold results saved to:")
    print(f"  {output_path}")


def main() -> None:
    dataset_path = get_dataset_path()

    print(f"Loading dataset: {dataset_path.name}")

    dataframe = load_csv(dataset_path)

    print(f"Dataset loaded successfully: {len(dataframe)} rows\n")

    for target in CLASSIFICATION_TARGETS:
        tune_target(
            dataframe=dataframe,
            target=target,
        )

    print("\n" + "=" * 60)
    print("THRESHOLD TUNING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()