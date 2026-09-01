"""
Train baseline machine learning models for the
AI-Driven Casting Quality System.

This module coordinates dataset loading, train/test splitting,
pipeline training, and prediction generation.
"""

from pathlib import Path

import pandas as pd

from src.data.load_data import load_csv
from src.data.split_data import split_dataset
from src.models.baseline.classification import (
    predict_classification,
    train_classification_model,
)
from src.models.baseline.regression import (
    predict_regression,
    train_regression_model,
)
from src.models.core.model_config import (
    BASELINE_CLASSIFICATION_MODELS,
    BASELINE_REGRESSION_MODELS,
    CLASSIFICATION_TARGETS,
    REGRESSION_TARGETS,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"


def get_dataset_path() -> Path:
    """
    Find the single CSV dataset inside the raw data directory.
    """

    csv_files = list(DATA_DIRECTORY.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {DATA_DIRECTORY}"
        )

    if len(csv_files) > 1:
        raise ValueError(
            "Multiple CSV files found. "
            f"Expected exactly one dataset, found: {len(csv_files)}"
        )

    return csv_files[0]


def train_classification_baselines(
    dataframe: pd.DataFrame,
) -> None:
    """
    Train all baseline classification models for every
    classification target.
    """

    for target in CLASSIFICATION_TARGETS:

        print("\n" + "=" * 60)
        print(f"CLASSIFICATION TARGET: {target}")
        print("=" * 60)

        X_train, X_test, y_train, y_test = split_dataset(
            dataframe=dataframe,
            target=target,
        )

        for model_name in BASELINE_CLASSIFICATION_MODELS:

            print(f"\nTraining model: {model_name}")

            pipeline = train_classification_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
            )

            predictions = predict_classification(
                pipeline=pipeline,
                X_test=X_test,
            )

            print(
                f"Training completed: {model_name}"
            )

            print(
                f"Test predictions generated: "
                f"{len(predictions)}"
            )


def train_regression_baselines(
    dataframe: pd.DataFrame,
) -> None:
    """
    Train all baseline regression models for every
    regression target.
    """

    for target in REGRESSION_TARGETS:

        print("\n" + "=" * 60)
        print(f"REGRESSION TARGET: {target}")
        print("=" * 60)

        X_train, X_test, y_train, y_test = split_dataset(
            dataframe=dataframe,
            target=target,
        )

        for model_name in BASELINE_REGRESSION_MODELS:

            print(f"\nTraining model: {model_name}")

            pipeline = train_regression_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
            )

            predictions = predict_regression(
                pipeline=pipeline,
                X_test=X_test,
            )

            print(
                f"Training completed: {model_name}"
            )

            print(
                f"Test predictions generated: "
                f"{len(predictions)}"
            )


def main() -> None:
    """
    Run baseline model training.
    """

    dataset_path = get_dataset_path()

    print(
        f"\nLoading dataset: {dataset_path.name}"
    )

    dataframe = load_csv(dataset_path)

    print(
        f"Dataset loaded successfully: "
        f"{len(dataframe)} rows"
    )

    train_classification_baselines(
        dataframe=dataframe,
    )

    train_regression_baselines(
        dataframe=dataframe,
    )

    print("\n" + "=" * 60)
    print("BASELINE MODEL TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()