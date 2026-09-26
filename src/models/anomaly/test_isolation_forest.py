from pathlib import Path

from src.data.load_data import load_csv
from src.features.schema import INPUT_FEATURES
from src.models.anomaly.isolation_forest import (
    predict_anomalies,
    train_anomaly_model,
)


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]

    data_path = (
        project_root
        / "data"
        / "raw"
        / "foundry_quality_dataset_13548.csv"
    )

    dataframe = load_csv(data_path)

    X = dataframe[INPUT_FEATURES]

    model = train_anomaly_model(X)

    results = predict_anomalies(model, X)

    print("\nAnomaly detection results")
    print("=" * 40)

    print(
        results["anomaly_status"]
        .value_counts()
    )

    print("\nAnomaly percentage:")
    anomaly_percentage = (
        (results["anomaly_status"] == "Anomaly").mean() * 100
    )
    print(f"{anomaly_percentage:.2f}%")

    print("\nFirst 10 results:")
    print(results.head(10).to_string())


if __name__ == "__main__":
    main()