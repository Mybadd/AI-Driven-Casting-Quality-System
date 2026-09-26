"""
Cross-validate class-balanced classification models.

This experiment evaluates the stability of the class-balanced Logistic
Regression model across multiple stratified folds.

The original baseline and class-balanced training scripts are not modified.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data.load_data import load_csv
from src.features.schema import INPUT_FEATURES
from src.models.baseline.classification import create_classification_pipeline
from src.models.core.model_config import CLASSIFICATION_TARGETS, RANDOM_STATE
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
MODEL_NAME = "balanced_logistic_regression"

CV_SPLITS = 5

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(
        precision_score,
        pos_label="Yes",
        zero_division=0,
    ),
    "recall": make_scorer(
        recall_score,
        pos_label="Yes",
        zero_division=0,
    ),
    "f1": make_scorer(
        f1_score,
        pos_label="Yes",
        zero_division=0,
    ),
}


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


def cross_validate_target(
    dataframe: pd.DataFrame,
    target: str,
) -> dict:
    X = dataframe[INPUT_FEATURES]
    y = dataframe[target]

    pipeline = create_classification_pipeline(
        model_name=MODEL_NAME
    )

    cross_validator = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = cross_validate(
        estimator=pipeline,
        X=X,
        y=y,
        cv=cross_validator,
        scoring=SCORING,
        return_train_score=False,
        n_jobs=-1,
    )

    summary = {}

    for metric in SCORING:
        scores = results[f"test_{metric}"]

        summary[metric] = {
            "mean": scores.mean(),
            "std": scores.std(),
            "fold_scores": scores.tolist(),
        }

    return summary


def print_results(target: str, results: dict) -> None:
    print("\n" + "=" * 60)
    print(f"CROSS-VALIDATION TARGET: {target}")
    print("=" * 60)

    print(f"Model: {MODEL_NAME}")
    print(f"Folds: {CV_SPLITS}")

    for metric, values in results.items():
        print(
            f"{metric}: "
            f"{values['mean']:.4f} "
            f"+/- "
            f"{values['std']:.4f}"
        )

    print("\nFold-level F1 scores:")

    for fold_number, score in enumerate(
        results["f1"]["fold_scores"],
        start=1,
    ):
        print(f"  Fold {fold_number}: {score:.4f}")


def save_results(
    all_results: dict,
) -> None:
    output_directory = (
        PROJECT_ROOT
        / "reports"
        / "evaluation"
        / "cross_validation"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for target, metrics in all_results.items():
        for metric, values in metrics.items():
            rows.append(
                {
                    "target": target,
                    "model": MODEL_NAME,
                    "metric": metric,
                    "mean": values["mean"],
                    "std": values["std"],
                }
            )

    results_dataframe = pd.DataFrame(rows)

    output_path = (
        output_directory
        / f"{MODEL_NAME}__5fold_summary.csv"
    )

    results_dataframe.to_csv(
        output_path,
        index=False,
    )

    print("\nCross-validation summary saved to:")
    print(f"  {output_path}")


def main() -> None:
    dataset_path = get_dataset_path()

    print(f"Loading dataset: {dataset_path.name}")

    dataframe = load_csv(dataset_path)

    print(
        f"Dataset loaded successfully: "
        f"{len(dataframe)} rows"
    )

    all_results = {}

    for target in CLASSIFICATION_TARGETS:
        results = cross_validate_target(
            dataframe=dataframe,
            target=target,
        )

        all_results[target] = results

        print_results(
            target=target,
            results=results,
        )

    save_results(all_results)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()