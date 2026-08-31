from pathlib import Path

import pandas as pd

from src.data.validate_data import (
    get_categorical_statistics,
    get_dataset_summary,
    get_missing_values,
    get_numeric_statistics,
    get_unique_value_counts,
)


def generate_data_quality_report(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Generate a Markdown data quality report.
    """

    summary = get_dataset_summary(dataframe)
    missing_values = get_missing_values(dataframe)
    unique_values = get_unique_value_counts(dataframe)
    numeric_statistics = get_numeric_statistics(
        dataframe
    )
    categorical_statistics = (
        get_categorical_statistics(dataframe)
    )

    report_lines = []

    report_lines.append("# Data Quality Report\n")

    report_lines.append("## Dataset Summary\n")

    report_lines.append(
        f"- Total rows: {summary['rows']}"
    )

    report_lines.append(
        f"- Total columns: {summary['columns']}"
    )

    report_lines.append(
        f"- Duplicate rows: {summary['duplicate_rows']}\n"
    )

    report_lines.append("## Columns\n")

    for column, dtype in summary["data_types"].items():
        report_lines.append(
            f"- `{column}`: {dtype}"
        )

    report_lines.append("\n## Missing Values\n")

    for column, count in missing_values.items():
        report_lines.append(
            f"- `{column}`: {count}"
        )

    report_lines.append("\n## Unique Values\n")

    for column, count in unique_values.items():
        report_lines.append(
            f"- `{column}`: {count}"
        )

    report_lines.append(
        "\n## Numeric Statistics\n"
    )

    for column, statistics in (
        numeric_statistics.items()
    ):

        report_lines.append(
            f"### {column}\n"
        )

        for metric, value in statistics.items():
            report_lines.append(
                f"- {metric}: {value}"
            )

        report_lines.append("")

    report_lines.append(
        "\n## Categorical Value Distribution\n"
    )

    for column, values in (
        categorical_statistics.items()
    ):

        report_lines.append(
            f"### {column}\n"
        )

        for value, count in values.items():
            report_lines.append(
                f"- `{value}`: {count}"
            )

        report_lines.append("")

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )