from pathlib import Path

import pandas as pd

from src.data.load_data import load_csv
from src.features.schema import (
    CLASSIFICATION_TARGETS,
    INPUT_FEATURES,
    REGRESSION_TARGET,
)
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

    output_dir = (
        project_root
        / "reports"
        / "evaluation"
        / "anomaly"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataframe = load_csv(data_path)

    X = dataframe[INPUT_FEATURES]

    model = train_anomaly_model(X)

    anomaly_results = predict_anomalies(model, X)

    analysis = dataframe.copy()

    analysis["anomaly_status"] = anomaly_results["anomaly_status"]
    analysis["anomaly_score"] = anomaly_results["anomaly_score"]

    print("\nAnomaly vs Quality Analysis")
    print("=" * 50)

    for target in CLASSIFICATION_TARGETS:
        print(f"\n--- {target} ---")

        comparison = pd.crosstab(
            analysis["anomaly_status"],
            analysis[target],
            normalize="index",
        ) * 100

        print(comparison.round(2).to_string())

    print(f"\n--- {REGRESSION_TARGET} ---")

    yield_comparison = (
        analysis
        .groupby("anomaly_status")[REGRESSION_TARGET]
        .agg(["count", "mean", "median", "std"])
    )

    print(yield_comparison.round(3).to_string())

    output_path = output_dir / "anomaly_quality_analysis.csv"

    summary_rows = []

    for status, group in analysis.groupby("anomaly_status"):
        row = {
            "anomaly_status": status,
            "count": len(group),
            "yield_mean": group[REGRESSION_TARGET].mean(),
            "yield_median": group[REGRESSION_TARGET].median(),
        }

        for target in CLASSIFICATION_TARGETS:
            row[f"{target}_yes_rate"] = (
                group[target].eq("Yes").mean()
            )

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(output_path, index=False)

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()