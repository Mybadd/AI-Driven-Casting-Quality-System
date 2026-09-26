"""Train and save the final Isolation Forest anomaly pipeline."""

from __future__ import annotations

from pathlib import Path

from src.data.load_data import load_csv
from src.features.schema import INPUT_FEATURES
from src.models.anomaly.isolation_forest import (
    save_anomaly_model,
    train_anomaly_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "anomaly"
    / "isolation_forest_pipeline.joblib"
)


def get_dataset_path() -> Path:
    """Locate the single raw CSV dataset."""

    csv_files = list(DATA_DIRECTORY.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV dataset found in {DATA_DIRECTORY}"
        )

    if len(csv_files) > 1:
        raise ValueError(
            f"Expected one CSV dataset, found {len(csv_files)}: "
            f"{csv_files}"
        )

    return csv_files[0]


def main() -> None:
    """Train and save the anomaly detection pipeline."""

    dataset_path = get_dataset_path()

    print(f"Loading dataset: {dataset_path.name}")

    dataframe = load_csv(dataset_path)

    print(
        f"Dataset loaded successfully: "
        f"{len(dataframe)} rows"
    )

    X = dataframe[INPUT_FEATURES]

    print("\nTraining Isolation Forest anomaly model...")

    pipeline = train_anomaly_model(X)

    save_anomaly_model(
        pipeline=pipeline,
        model_path=MODEL_PATH,
    )

    print(f"Saved anomaly model: {MODEL_PATH}")
    print("\nAnomaly model training complete.")


if __name__ == "__main__":
    main()  