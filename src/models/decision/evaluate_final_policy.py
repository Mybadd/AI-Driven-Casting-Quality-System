from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

from src.data.load_data import load_csv
from src.features.schema import CLASSIFICATION_TARGETS, INPUT_FEATURES
from src.models.baseline.classification import (
    predict_classification_probabilities,
    train_classification_model,
)
from src.models.advanced.xgboost_classification import (
    create_xgboost_pipeline,
    decode_xgboost_predictions,
    encode_xgboost_target,
)


RANDOM_STATE = 42
TEST_SIZE = 0.20

SELECTED_MODELS = {
    "Defect": ("xgboost", 0.48),
    "Porosity": ("xgboost", 0.47),
    "Scrap": ("balanced_logistic_regression", 0.50),
}


def evaluate_predictions(
    y_true: pd.Series,
    probabilities,
    threshold: float,
) -> dict:
    predictions = (probabilities >= threshold).astype(int)

    accuracy = accuracy_score(y_true, predictions)
    precision = precision_score(y_true, predictions, zero_division=0)
    recall = recall_score(y_true, predictions, zero_division=0)
    f1 = f1_score(y_true, predictions, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    total = len(y_true)
    actual_negative = tn + fp
    actual_positive = fn + tp

    fpr = fp / actual_negative if actual_negative else 0.0
    fnr = fn / actual_positive if actual_positive else 0.0
    alert_rate = (fp + tp) / total if total else 0.0

    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "alert_rate": alert_rate,
        "fpr": fpr,
        "fnr": fnr,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    y_train_encoded = encode_xgboost_target(y_train)
    y_test_encoded = encode_xgboost_target(y_test)

    pipeline = create_xgboost_pipeline(y_train_encoded)
    pipeline.fit(X_train, y_train_encoded)

    probabilities = pipeline.predict_proba(X_test)[:, 1]

    return evaluate_predictions(
        y_test_encoded,
        probabilities,
        threshold,
    )


def evaluate_balanced_logistic(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict:
    pipeline = train_classification_model(
        "balanced_logistic_regression",
        X_train,
        y_train,
    )

    probabilities = predict_classification_probabilities(
        pipeline,
        X_test,
    )[:, 1]

    y_test_encoded = y_test.map({"No": 0, "Yes": 1}).astype(int)

    return evaluate_predictions(
        y_test_encoded,
        probabilities,
        threshold,
    )


def main() -> None:
    global TEST_Y

    project_root = Path(__file__).resolve().parents[3]
    data_path = project_root / "data" / "raw" / "foundry_quality_dataset_13548.csv"
    output_dir = project_root / "reports" / "evaluation" / "final_policy"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_csv(data_path)

    X = dataframe[INPUT_FEATURES]

    results = []

    for target in CLASSIFICATION_TARGETS:
        y = dataframe[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        TEST_Y = y_test

        model_name, threshold = SELECTED_MODELS[target]

        if model_name == "xgboost":
            metrics = evaluate_xgboost(
                X_train,
                y_train,
                X_test,
                y_test,
                threshold,
            )
        elif model_name == "balanced_logistic_regression":
            metrics = evaluate_balanced_logistic(
                X_train,
                y_train,
                X_test,
                y_test,
                threshold,
            )
        else:
            raise ValueError(f"Unsupported model: {model_name}")

        metrics.update(
            {
                "target": target,
                "model": model_name,
            }
        )

        results.append(metrics)

    results_df = pd.DataFrame(results)

    column_order = [
        "target",
        "model",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "alert_rate",
        "fpr",
        "fnr",
        "tn",
        "fp",
        "fn",
        "tp",
    ]

    results_df = results_df[column_order]

    output_path = output_dir / "final_policy_test_results.csv"
    results_df.to_csv(output_path, index=False)

    print(results_df.to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()