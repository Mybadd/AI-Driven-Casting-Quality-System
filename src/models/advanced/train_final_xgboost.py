"""Train and save final XGBoost classification pipelines."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data.load_data import load_csv
from src.features.schema import CLASSIFICATION_TARGETS, INPUT_FEATURES
from src.models.advanced.xgboost_classification import (
    create_xgboost_pipeline,
    encode_xgboost_target,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
MODEL_DIRECTORY = PROJECT_ROOT / "models"


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


def train_and_save_model(
    dataframe: pd.DataFrame,
    target: str,
) -> Path:
    """Train XGBoost on the complete dataset and save the pipeline."""

    X = dataframe[INPUT_FEATURES]
    y = encode_xgboost_target(dataframe[target])

    pipeline = create_xgboost_pipeline(
        y_train=y,
    )

    pipeline.fit(X, y)

    target_directory = MODEL_DIRECTORY / target.lower()
    target_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        target_directory
        / "xgboost_pipeline.joblib"
    )

    joblib.dump(
        pipeline,
        model_path,
    )

    return model_path


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

    print(
        "\nTraining final XGBoost models..."
    )

    for target in CLASSIFICATION_TARGETS:
        print(
            f"\nTraining target: {target}"
        )

        model_path = train_and_save_model(
            dataframe=dataframe,
            target=target,
        )

        print(
            f"Saved model: {model_path}"
        )

    print(
        "\nFinal XGBoost model training complete."
    )


if __name__ == "__main__":
    main()