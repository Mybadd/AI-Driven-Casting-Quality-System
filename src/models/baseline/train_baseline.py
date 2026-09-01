"""
Train and evaluate baseline machine learning models for the
AI-Driven Casting Quality System.

This module coordinates:

- Dataset loading
- Train/test splitting
- Model training
- Prediction generation
- Model evaluation
- Evaluation report saving
"""

from pathlib import Path

import pandas as pd

from src.data.load_data import load_csv
from src.data.split_data import split_dataset
from src.evaluation.evaluator import evaluate_model
from src.evaluation.generate_reports import (
    save_evaluation_result,
)
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

DATA_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
)


def get_dataset_path() -> Path:
    """
    Find the single CSV dataset inside the raw data directory.
    """

    csv_files = list(
        DATA_DIRECTORY.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {DATA_DIRECTORY}"
        )

    if len(csv_files) > 1:
        raise ValueError(
            "Multiple CSV files found. "
            f"Expected exactly one dataset, found: "
            f"{len(csv_files)}"
        )

    return csv_files[0]


def train_classification_baselines(
    dataframe: pd.DataFrame,
) -> None:
    """
    Train and evaluate all baseline classification models.
    """

    for target in CLASSIFICATION_TARGETS:

        print("\n" + "=" * 60)
        print(
            f"CLASSIFICATION TARGET: {target}"
        )
        print("=" * 60)

        X_train, X_test, y_train, y_test = (
            split_dataset(
                dataframe=dataframe,
                target=target,
            )
        )

        for model_name in (
            BASELINE_CLASSIFICATION_MODELS
        ):

            print(
                f"\nTraining model: {model_name}"
            )

            pipeline = (
                train_classification_model(
                    model_name=model_name,
                    X_train=X_train,
                    y_train=y_train,
                )
            )

            predictions = (
                predict_classification(
                    pipeline=pipeline,
                    X_test=X_test,
                )
            )

            evaluation_result = (
                evaluate_model(
                    target=target,
                    y_true=y_test,
                    y_pred=predictions,
                )
            )

            saved_paths = (
                save_evaluation_result(
                    target=target,
                    model_name=model_name,
                    evaluation_result=evaluation_result,
                )
            )

            print(
                f"Training completed: "
                f"{model_name}"
            )

            print(
                f"Test predictions generated: "
                f"{len(predictions)}"
            )

            print(
                "Evaluation metrics:"
            )

            for metric_name, metric_value in (
                evaluation_result[
                    "metrics"
                ].items()
            ):

                print(
                    f"  {metric_name}: "
                    f"{metric_value:.4f}"
                )

            print(
                "Reports saved:"
            )

            for report_name, report_path in (
                saved_paths.items()
            ):

                print(
                    f"  {report_name}: "
                    f"{report_path}"
                )


def train_regression_baselines(
    dataframe: pd.DataFrame,
) -> None:
    """
    Train and evaluate all baseline regression models.
    """

    for target in REGRESSION_TARGETS:

        print("\n" + "=" * 60)
        print(
            f"REGRESSION TARGET: {target}"
        )
        print("=" * 60)

        X_train, X_test, y_train, y_test = (
            split_dataset(
                dataframe=dataframe,
                target=target,
            )
        )

        for model_name in (
            BASELINE_REGRESSION_MODELS
        ):

            print(
                f"\nTraining model: {model_name}"
            )

            pipeline = train_regression_model(
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
            )

            predictions = predict_regression(
                pipeline=pipeline,
                X_test=X_test,
            )

            evaluation_result = (
                evaluate_model(
                    target=target,
                    y_true=y_test,
                    y_pred=predictions,
                )
            )

            saved_paths = (
                save_evaluation_result(
                    target=target,
                    model_name=model_name,
                    evaluation_result=evaluation_result,
                )
            )

            print(
                f"Training completed: "
                f"{model_name}"
            )

            print(
                f"Test predictions generated: "
                f"{len(predictions)}"
            )

            print(
                "Evaluation metrics:"
            )

            for metric_name, metric_value in (
                evaluation_result[
                    "metrics"
                ].items()
            ):

                print(
                    f"  {metric_name}: "
                    f"{metric_value:.4f}"
                )

            print(
                "Reports saved:"
            )

            for report_name, report_path in (
                saved_paths.items()
            ):

                print(
                    f"  {report_name}: "
                    f"{report_path}"
                )


def main() -> None:
    """
    Run baseline model training and evaluation.
    """

    dataset_path = get_dataset_path()

    print(
        f"\nLoading dataset: "
        f"{dataset_path.name}"
    )

    dataframe = load_csv(
        dataset_path
    )

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
    print(
        "BASELINE TRAINING AND "
        "EVALUATION COMPLETED SUCCESSFULLY"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()