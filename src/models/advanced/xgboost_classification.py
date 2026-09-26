"""
XGBoost classification experiment.

This module compares an XGBoost classifier against the established
class-balanced Logistic Regression reference using the original five
approved ML features.

No engineered features are used in this experiment.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data.load_data import load_csv
from src.features.preprocessing import create_tree_preprocessor
from src.features.schema import INPUT_FEATURES
from src.models.baseline.classification import (
    create_classification_pipeline,
)
from src.models.core.model_config import (
    CLASSIFICATION_TARGETS,
    RANDOM_STATE,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"

CV_SPLITS = 5


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


def encode_xgboost_target(y: pd.Series) -> pd.Series:
    """
    Encode the project labels for XGBoost.

    No -> 0
    Yes -> 1
    """

    mapping = {
        "No": 0,
        "Yes": 1,
    }

    encoded = y.map(mapping)

    if encoded.isna().any():
        invalid_values = sorted(
            y[encoded.isna()].unique()
        )

        raise ValueError(
            f"Unexpected target labels: {invalid_values}"
        )

    return encoded.astype(int)


def decode_xgboost_predictions(
    predictions: pd.Series,
) -> pd.Series:
    """
    Convert XGBoost predictions back to project labels.

    0 -> No
    1 -> Yes
    """

    mapping = {
        0: "No",
        1: "Yes",
    }

    decoded = predictions.map(mapping)

    if decoded.isna().any():
        invalid_values = sorted(
            predictions[decoded.isna()].unique()
        )

        raise ValueError(
            f"Unexpected XGBoost prediction labels: {invalid_values}"
        )

    return decoded


def create_xgboost_model(
    y_train: pd.Series,
) -> XGBClassifier:
    """
    Create a class-balanced XGBoost classifier.

    scale_pos_weight is calculated from the current training fold only.
    """

    positive_count = int(
        (y_train == 1).sum()
    )

    negative_count = int(
        (y_train == 0).sum()
    )

    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1.0
    )

    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def create_xgboost_pipeline(
    y_train: pd.Series,
) -> Pipeline:
    """
    Create the preprocessing + XGBoost pipeline.
    """

    preprocessor = create_tree_preprocessor()

    model = create_xgboost_model(
        y_train=y_train,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def evaluate_xgboost(
    dataframe: pd.DataFrame,
    target: str,
) -> dict:
    """
    Evaluate XGBoost using stratified cross-validation.
    """

    X = dataframe[INPUT_FEATURES]
    y = dataframe[target]

    y_encoded = encode_xgboost_target(y)

    cross_validator = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
    }

    for fold_number, (
        train_index,
        test_index,
    ) in enumerate(
        cross_validator.split(X, y_encoded),
        start=1,
    ):
        X_train = X.iloc[train_index]
        X_test = X.iloc[test_index]

        y_train = y_encoded.iloc[train_index]
        y_test = y.iloc[test_index]

        pipeline = create_xgboost_pipeline(
            y_train=y_train,
        )

        pipeline.fit(
            X_train,
            y_train,
        )

        predictions_encoded = pipeline.predict(
            X_test,
        )

        predictions = decode_xgboost_predictions(
            pd.Series(
                predictions_encoded,
                index=X_test.index,
            )
        )

        fold_results["accuracy"].append(
            (predictions == y_test).mean()
        )

        fold_results["precision"].append(
            precision_score(
                y_test,
                predictions,
                pos_label="Yes",
                zero_division=0,
            )
        )

        fold_results["recall"].append(
            recall_score(
                y_test,
                predictions,
                pos_label="Yes",
                zero_division=0,
            )
        )

        fold_results["f1"].append(
            f1_score(
                y_test,
                predictions,
                pos_label="Yes",
                zero_division=0,
            )
        )

        print(
            f"  {target} | Fold {fold_number} complete"
        )

    summary = {}

    for metric, scores in fold_results.items():
        score_series = pd.Series(scores)

        summary[metric] = {
            "mean": score_series.mean(),
            "std": score_series.std(ddof=0),
            "fold_scores": scores,
        }

    return summary


def evaluate_reference_model(
    dataframe: pd.DataFrame,
    target: str,
) -> dict:
    """
    Evaluate the established balanced Logistic Regression reference.
    """

    from sklearn.metrics import make_scorer
    from sklearn.model_selection import cross_validate

    X = dataframe[INPUT_FEATURES]
    y = dataframe[target]

    pipeline = create_classification_pipeline(
        model_name="balanced_logistic_regression",
    )

    scoring = {
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
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1,
    )

    summary = {}

    for metric in scoring:
        scores = results[f"test_{metric}"]

        summary[metric] = {
            "mean": scores.mean(),
            "std": scores.std(),
            "fold_scores": scores.tolist(),
        }

    return summary


def print_model_results(
    model_name: str,
    target: str,
    results: dict,
) -> None:
    print("\n" + "-" * 60)
    print(f"Target: {target}")
    print(f"Model: {model_name}")

    for metric, values in results.items():
        print(
            f"{metric}: "
            f"{values['mean']:.4f} "
            f"+/- "
            f"{values['std']:.4f}"
        )


def print_comparison(
    reference_results: dict,
    xgboost_results: dict,
) -> None:
    print(
        "\nComparison against balanced Logistic Regression:"
    )

    for metric in (
        "accuracy",
        "precision",
        "recall",
        "f1",
    ):
        reference_mean = (
            reference_results[metric]["mean"]
        )

        xgboost_mean = (
            xgboost_results[metric]["mean"]
        )

        difference = (
            xgboost_mean - reference_mean
        )

        print(
            f"  {metric}: "
            f"{reference_mean:.4f} -> "
            f"{xgboost_mean:.4f} "
            f"(change {difference:+.4f})"
        )


def save_results(
    all_results: list[dict],
) -> None:
    output_directory = (
        PROJECT_ROOT
        / "reports"
        / "evaluation"
        / "advanced_models"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    for result in all_results:
        for metric, values in result["metrics"].items():
            rows.append(
                {
                    "target": result["target"],
                    "model": result["model"],
                    "metric": metric,
                    "mean": values["mean"],
                    "std": values["std"],
                }
            )

    output_dataframe = pd.DataFrame(rows)

    output_path = (
        output_directory
        / "xgboost_vs_balanced_logistic_regression.csv"
    )

    output_dataframe.to_csv(
        output_path,
        index=False,
    )

    print(
        "\nAdvanced-model comparison saved to:"
    )
    print(f"  {output_path}")


def main() -> None:
    dataset_path = get_dataset_path()

    print(
        f"Loading dataset: {dataset_path.name}"
    )

    dataframe = load_csv(dataset_path)

    print(
        f"Dataset loaded successfully: "
        f"{len(dataframe)} rows"
    )

    all_results = []

    for target in CLASSIFICATION_TARGETS:
        print("\n" + "=" * 60)
        print(
            f"ADVANCED MODEL TARGET: {target}"
        )
        print("=" * 60)

        print(
            "\nEvaluating balanced Logistic Regression..."
        )

        reference_results = evaluate_reference_model(
            dataframe=dataframe,
            target=target,
        )

        print_model_results(
            model_name="balanced_logistic_regression",
            target=target,
            results=reference_results,
        )

        print("\nEvaluating XGBoost...")

        xgboost_results = evaluate_xgboost(
            dataframe=dataframe,
            target=target,
        )

        print_model_results(
            model_name="xgboost",
            target=target,
            results=xgboost_results,
        )

        print_comparison(
            reference_results=reference_results,
            xgboost_results=xgboost_results,
        )

        all_results.extend(
            [
                {
                    "target": target,
                    "model": "balanced_logistic_regression",
                    "metrics": reference_results,
                },
                {
                    "target": target,
                    "model": "xgboost",
                    "metrics": xgboost_results,
                },
            ]
        )

    save_results(all_results)

    print("\n" + "=" * 60)
    print(
        "XGBOOST ADVANCED MODEL EVALUATION COMPLETED"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()