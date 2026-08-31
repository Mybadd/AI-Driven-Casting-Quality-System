from pathlib import Path
import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data.load_data import load_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"
EDA_REPORT_DIRECTORY = PROJECT_ROOT / "reports" / "eda"
FIGURE_DIRECTORY = PROJECT_ROOT / "reports" / "figures"


INPUT_FEATURES = [
    "Alloy",
    "Pour_Temp",
    "Mold_Moisture",
    "Cooling_Time",
    "Riser",
]

CLASSIFICATION_TARGETS = [
    "Defect",
    "Porosity",
    "Scrap",
]

REGRESSION_TARGET = "Yield"


def find_dataset() -> Path:
    """Return the first CSV dataset found in data/raw."""

    csv_files = list(RAW_DATA_DIRECTORY.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/raw."
        )

    if len(csv_files) > 1:
        print(
            "Warning: Multiple CSV files found. "
            f"Using: {csv_files[0].name}"
        )

    return csv_files[0]


def create_directories() -> None:
    """Create report and figure directories if needed."""

    EDA_REPORT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


def analyse_target_distribution(
    dataframe: pd.DataFrame,
) -> dict:
    """Calculate target distributions."""

    results = {}

    for target in CLASSIFICATION_TARGETS:
        counts = dataframe[target].value_counts(
            dropna=False
        )

        percentages = (
            dataframe[target]
            .value_counts(normalize=True, dropna=False)
            .mul(100)
            .round(2)
        )

        results[target] = {
            str(value): {
                "count": int(counts[value]),
                "percentage": float(percentages[value]),
            }
            for value in counts.index
        }

    yield_series = dataframe[REGRESSION_TARGET]

    results[REGRESSION_TARGET] = {
        "minimum": float(yield_series.min()),
        "maximum": float(yield_series.max()),
        "mean": float(yield_series.mean()),
        "median": float(yield_series.median()),
        "standard_deviation": float(
            yield_series.std()
        ),
    }

    return results


def create_target_plots(
    dataframe: pd.DataFrame,
) -> None:
    """Create target distribution plots."""

    for target in CLASSIFICATION_TARGETS:

        plt.figure(figsize=(7, 5))

        sns.countplot(
            data=dataframe,
            x=target,
        )

        plt.title(
            f"{target} Distribution"
        )

        plt.xlabel(target)
        plt.ylabel("Count")

        plt.tight_layout()

        plt.savefig(
            FIGURE_DIRECTORY
            / f"{target.lower()}_distribution.png"
        )

        plt.close()

    plt.figure(figsize=(8, 5))

    sns.histplot(
        dataframe[REGRESSION_TARGET],
        kde=True,
    )

    plt.title("Yield Distribution")

    plt.xlabel("Yield")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "yield_distribution.png"
    )

    plt.close()


def create_numeric_correlation(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Create numeric feature correlation matrix."""

    numeric_columns = [
        "Pour_Temp",
        "Mold_Moisture",
        "Cooling_Time",
        "Riser",
        "Yield",
    ]

    correlation = dataframe[
        numeric_columns
    ].corr()

    correlation.to_csv(
        EDA_REPORT_DIRECTORY
        / "correlations.csv"
    )

    return correlation


def create_correlation_plot(
    correlation: pd.DataFrame,
) -> None:
    """Create correlation heatmap."""

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
    )

    plt.title(
        "Numeric Feature Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIRECTORY
        / "correlation_heatmap.png"
    )

    plt.close()


def create_feature_summary(
    dataframe: pd.DataFrame,
) -> dict:
    """Create summary statistics for input features."""

    summary = {}

    for feature in INPUT_FEATURES:

        if feature == "Alloy":

            summary[feature] = (
                dataframe[feature]
                .value_counts()
                .to_dict()
            )

        else:

            series = dataframe[feature]

            summary[feature] = {
                "minimum": float(series.min()),
                "maximum": float(series.max()),
                "mean": float(series.mean()),
                "median": float(series.median()),
                "standard_deviation": float(
                    series.std()
                ),
            }

    return summary


def check_ml_readiness(
    dataframe: pd.DataFrame,
) -> dict:
    """Check basic ML readiness conditions."""

    expected_columns = (
        INPUT_FEATURES
        + CLASSIFICATION_TARGETS
        + [REGRESSION_TARGET]
    )

    missing_columns = [
        column
        for column in expected_columns
        if column not in dataframe.columns
    ]

    return {
        "dataset_rows": int(dataframe.shape[0]),
        "dataset_columns": int(
            dataframe.shape[1]
        ),
        "missing_required_columns": (
            missing_columns
        ),
        "missing_values": int(
            dataframe[
                expected_columns
            ].isnull().sum().sum()
        ),
        "duplicate_rows": int(
            dataframe.duplicated().sum()
        ),
        "status": (
            "READY_FOR_PREPROCESSING"
            if not missing_columns
            else "NOT_READY"
        ),
    }


def write_markdown_report(
    target_distribution: dict,
    feature_summary: dict,
    readiness: dict,
) -> None:
    """Write EDA summary in Markdown."""

    report_path = (
        EDA_REPORT_DIRECTORY
        / "ml_readiness_report.md"
    )

    lines = [
        "# ML Readiness Report",
        "",
        "## Dataset Readiness",
        "",
    ]

    for key, value in readiness.items():
        lines.append(
            f"- **{key}**: {value}"
        )

    lines.extend([
        "",
        "## Classification Target Distribution",
        "",
    ])

    for target in CLASSIFICATION_TARGETS:

        lines.append(
            f"### {target}"
        )
        lines.append("")

        for value, details in (
            target_distribution[target]
            .items()
        ):

            lines.append(
                f"- **{value}**: "
                f"{details['count']} rows "
                f"({details['percentage']}%)"
            )

        lines.append("")

    lines.extend([
        "## Yield Statistics",
        "",
    ])

    for metric, value in (
        target_distribution[
            REGRESSION_TARGET
        ].items()
    ):

        lines.append(
            f"- **{metric}**: {value}"
        )

    lines.extend([
        "",
        "## Input Feature Summary",
        "",
    ])

    for feature, values in (
        feature_summary.items()
    ):

        lines.append(
            f"### {feature}"
        )
        lines.append("")

        for key, value in values.items():
            lines.append(
                f"- **{key}**: {value}"
            )

        lines.append("")

    lines.extend([
        "## Initial Conclusion",
        "",
        "The dataset has been analysed for "
        "machine-learning readiness. Final "
        "preprocessing decisions must be based "
        "on target balance, feature behaviour, "
        "and relationship analysis.",
    ])

    report_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    """Run exploratory data analysis."""

    create_directories()

    dataset_path = find_dataset()

    print(
        f"Loading dataset: {dataset_path.name}"
    )

    dataframe = load_csv(dataset_path)

    target_distribution = (
        analyse_target_distribution(
            dataframe
        )
    )

    feature_summary = (
        create_feature_summary(
            dataframe
        )
    )

    readiness = check_ml_readiness(
        dataframe
    )

    correlation = (
        create_numeric_correlation(
            dataframe
        )
    )

    create_target_plots(dataframe)

    create_correlation_plot(correlation)

    write_markdown_report(
        target_distribution,
        feature_summary,
        readiness,
    )

    summary_path = (
        EDA_REPORT_DIRECTORY
        / "target_distribution.json"
    )

    summary_path.write_text(
        json.dumps(
            target_distribution,
            indent=4,
        ),
        encoding="utf-8",
    )

    print(
        "EDA completed successfully."
    )

    print(
        f"Reports: {EDA_REPORT_DIRECTORY}"
    )

    print(
        f"Figures: {FIGURE_DIRECTORY}"
    )


if __name__ == "__main__":
    main()