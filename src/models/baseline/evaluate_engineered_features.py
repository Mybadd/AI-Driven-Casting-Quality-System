"""
Compare original and engineered feature sets using stratified cross-validation.

The original baseline, class-balanced training, and threshold-tuning scripts
are not modified.

This experiment determines whether the engineered features provide a stable
improvement over the original approved input features.
"""

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.data.load_data import load_csv
from src.features.engineering import add_engineered_features
from src.features.schema import INPUT_FEATURES
from src.models.baseline.classification import create_classification_pipeline
from src.models.core.model_config import CLASSIFICATION_TARGETS, RANDOM_STATE
from sklearn.pipeline import Pipeline

from src.features.preprocessing import (
    create_scaled_preprocessor_for_features,
)
from src.models.core.model_factory import create_classification_model

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


def evaluate_feature_set(
    dataframe: pd.DataFrame,
    target: str,
    feature_set_name: str,
    feature_columns: list[str],
) -> dict:
    X = dataframe[feature_columns]
    y = dataframe[target]

    if feature_set_name == "original":
        pipeline = create_classification_pipeline(
            model_name=MODEL_NAME
        )

    elif feature_set_name == "engineered":
        numerical_features = [
            column
            for column in feature_columns
            if column != "Alloy"
        ]

        preprocessor = create_scaled_preprocessor_for_features(
            numerical_features=numerical_features,
        )

        model = create_classification_model(
            model_name=MODEL_NAME
        )

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

    else:
        raise ValueError(
            f"Unsupported feature set: {feature_set_name}"
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

    return {
        "target": target,
        "feature_set": feature_set_name,
        "model": MODEL_NAME,
        "metrics": summary,
    }


def print_results(result: dict) -> None:
    print("\n" + "-" * 60)
    print(f"Target: {result['target']}")
    print(f"Feature set: {result['feature_set']}")
    print(f"Model: {result['model']}")

    for metric, values in result["metrics"].items():
        print(
            f"{metric}: "
            f"{values['mean']:.4f} "
            f"+/- "
            f"{values['std']:.4f}"
        )


def print_comparison(
    original_result: dict,
    engineered_result: dict,
) -> None:
    print("\nComparison:")

    for metric in SCORING:
        original_mean = original_result["metrics"][metric]["mean"]
        engineered_mean = engineered_result["metrics"][metric]["mean"]
        difference = engineered_mean - original_mean

        print(
            f"  {metric}: "
            f"{original_mean:.4f} -> "
            f"{engineered_mean:.4f} "
            f"(change {difference:+.4f})"
        )


def save_results(results: list[dict]) -> None:
    output_directory = (
        PROJECT_ROOT
        / "reports"
        / "evaluation"
        / "feature_engineering"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for result in results:
        for metric, values in result["metrics"].items():
            rows.append(
                {
                    "target": result["target"],
                    "feature_set": result["feature_set"],
                    "model": result["model"],
                    "metric": metric,
                    "mean": values["mean"],
                    "std": values["std"],
                }
            )

    output_dataframe = pd.DataFrame(rows)

    output_path = (
        output_directory
        / "balanced_logistic_regression_feature_comparison.csv"
    )

    output_dataframe.to_csv(
        output_path,
        index=False,
    )

    print("\nFeature-engineering comparison saved to:")
    print(f"  {output_path}")


def main() -> None:
    dataset_path = get_dataset_path()

    print(f"Loading dataset: {dataset_path.name}")

    dataframe = load_csv(dataset_path)

    print(
        f"Dataset loaded successfully: "
        f"{len(dataframe)} rows"
    )

    engineered_dataframe = add_engineered_features(
        dataframe
    )

    engineered_features = (
        INPUT_FEATURES
        + [
            column
            for column in engineered_dataframe.columns
            if column not in dataframe.columns
        ]
    )

    print("\nOriginal feature count:")
    print(f"  {len(INPUT_FEATURES)}")

    print("Engineered feature count:")
    print(f"  {len(engineered_features)}")

    all_results = []

    for target in CLASSIFICATION_TARGETS:
        original_result = evaluate_feature_set(
            dataframe=dataframe,
            target=target,
            feature_set_name="original",
            feature_columns=INPUT_FEATURES,
        )

        engineered_result = evaluate_feature_set(
            dataframe=engineered_dataframe,
            target=target,
            feature_set_name="engineered",
            feature_columns=engineered_features,
        )

        all_results.extend(
            [
                original_result,
                engineered_result,
            ]
        )

        print("\n" + "=" * 60)
        print(f"FEATURE ENGINEERING TARGET: {target}")
        print("=" * 60)

        print_results(original_result)
        print_results(engineered_result)

        print_comparison(
            original_result=original_result,
            engineered_result=engineered_result,
        )

    save_results(all_results)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING EVALUATION COMPLETED SUCCESSFULLY")
    print("=" * 60)


if __name__ == "__main__":
    main()