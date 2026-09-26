from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from src.features.schema import INPUT_FEATURES
from src.models.baseline.classification import create_classification_pipeline
from src.models.advanced.xgboost_classification import (
    create_xgboost_pipeline,
    encode_xgboost_target,
)


POSITIVE_LABEL = "Yes"
NEGATIVE_LABEL = "No"


@dataclass(frozen=True)
class DecisionPolicy:
    min_recall: float | None = None
    min_precision: float | None = None
    max_alert_rate: float | None = None


def _positive_probability(
    pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """Return probability of the positive class."""

    probabilities = pipeline.predict_proba(X)
    classes = list(pipeline.named_steps["model"].classes_)

    if POSITIVE_LABEL in classes:
        positive_index = classes.index(POSITIVE_LABEL)
    elif 1 in classes:
        positive_index = classes.index(1)
    else:
        raise ValueError(
            f"Positive class not found in model classes: {classes}"
        )

    return probabilities[:, positive_index]


def generate_oof_probabilities(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
    n_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate out-of-fold positive-class probabilities.

    Threshold selection is performed only using the training data.
    The final test set is never used here.
    """

    X_train = X_train[INPUT_FEATURES].reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    oof_probabilities = np.zeros(
        len(X_train),
        dtype=float,
    )

    for fold, (fit_index, validation_index) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):
        X_fit = X_train.iloc[fit_index]
        X_validation = X_train.iloc[validation_index]
        y_fit = y_train.iloc[fit_index]

        if model_name == "balanced_logistic_regression":

            pipeline = create_classification_pipeline(
                "balanced_logistic_regression"
            )

            pipeline.fit(
                X_fit,
                y_fit,
            )

        elif model_name == "xgboost":

            y_fit_encoded = encode_xgboost_target(y_fit)

            pipeline = create_xgboost_pipeline(
                y_fit_encoded
            )

            pipeline.fit(
                X_fit,
                y_fit_encoded,
            )

        else:
            raise ValueError(
                "model_name must be "
                "'balanced_logistic_regression' or 'xgboost'"
            )

        oof_probabilities[validation_index] = (
            _positive_probability(
                pipeline,
                X_validation,
            )
        )

        print(
            f"{model_name}: "
            f"fold {fold}/{n_splits} complete"
        )

    return oof_probabilities


def evaluate_thresholds(
    y_true: pd.Series,
    probabilities: np.ndarray,
    thresholds: Iterable[float] | None = None,
) -> pd.DataFrame:
    """
    Evaluate model performance across different thresholds.
    """

    if thresholds is None:
        thresholds = np.round(
            np.arange(
                0.10,
                0.91,
                0.01,
            ),
            2,
        )

    y_binary = y_true.map(
        {
            NEGATIVE_LABEL: 0,
            POSITIVE_LABEL: 1,
        }
    )

    if y_binary.isna().any():

        unexpected_labels = sorted(
            y_true[
                y_binary.isna()
            ].astype(str).unique()
        )

        raise ValueError(
            f"Unexpected target labels: "
            f"{unexpected_labels}"
        )

    rows = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_binary,
            predictions,
            labels=[0, 1],
        ).ravel()

        alert_rate = float(
            predictions.mean()
        )

        false_positive_rate = (
            fp / (fp + tn)
            if (fp + tn)
            else 0.0
        )

        false_negative_rate = (
            fn / (fn + tp)
            if (fn + tp)
            else 0.0
        )

        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": accuracy_score(
                    y_binary,
                    predictions,
                ),
                "precision": precision_score(
                    y_binary,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_binary,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_binary,
                    predictions,
                    zero_division=0,
                ),
                "alert_rate": alert_rate,
                "false_positive_rate": false_positive_rate,
                "false_negative_rate": false_negative_rate,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    return pd.DataFrame(rows)


def select_threshold_by_policy(
    results: pd.DataFrame,
    policy: DecisionPolicy,
) -> pd.Series | None:
    """
    Select a threshold using explicit engineering constraints.

    Ranking among feasible thresholds:

    1. Highest recall
    2. Highest precision
    3. Lowest alert rate
    4. Highest threshold

    If no engineering constraints are supplied,
    no threshold is selected.
    """

    if (
        policy.min_recall is None
        and policy.min_precision is None
        and policy.max_alert_rate is None
    ):
        return None

    feasible = results.copy()

    if policy.min_recall is not None:

        feasible = feasible[
            feasible["recall"]
            >= policy.min_recall
        ]

    if policy.min_precision is not None:

        feasible = feasible[
            feasible["precision"]
            >= policy.min_precision
        ]

    if policy.max_alert_rate is not None:

        feasible = feasible[
            feasible["alert_rate"]
            <= policy.max_alert_rate
        ]

    if feasible.empty:
        return None

    feasible = feasible.sort_values(
        by=[
            "recall",
            "precision",
            "alert_rate",
            "threshold",
        ],
        ascending=[
            False,
            False,
            True,
            False,
        ],
    )

    return feasible.iloc[0]


def run_oof_threshold_analysis(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    target_name: str,
    model_name: str,
    output_dir,
    policy: DecisionPolicy,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """
    Run OOF probability generation and threshold evaluation.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    probabilities = generate_oof_probabilities(
        X_train=X_train,
        y_train=y_train,
        model_name=model_name,
    )

    oof_predictions = pd.DataFrame(
        {
            "target": target_name,
            "actual": y_train.reset_index(
                drop=True
            ),
            "oof_probability_yes": probabilities,
        }
    )

    oof_predictions.to_csv(
        output_dir
        / (
            f"{target_name.lower()}"
            f"__{model_name}"
            f"__oof_predictions.csv"
        ),
        index=False,
    )

    threshold_results = evaluate_thresholds(
        y_true=y_train.reset_index(
            drop=True
        ),
        probabilities=probabilities,
    )

    threshold_results.to_csv(
        output_dir
        / (
            f"{target_name.lower()}"
            f"__{model_name}"
            f"__threshold_curve.csv"
        ),
        index=False,
    )

    selected_threshold = select_threshold_by_policy(
        threshold_results,
        policy,
    )

    return (
        threshold_results,
        selected_threshold,
    )