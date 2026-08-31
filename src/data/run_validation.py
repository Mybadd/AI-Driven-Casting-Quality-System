from pathlib import Path

from src.data.generate_report import (
    generate_data_quality_report,
)
from src.data.load_data import load_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIRECTORY = (
    PROJECT_ROOT / "data" / "raw"
)

REPORT_DIRECTORY = (
    PROJECT_ROOT / "reports"
)


def main() -> None:
    """
    Run validation for all CSV files
    in the raw dataset directory.
    """

    csv_files = list(
        RAW_DATA_DIRECTORY.glob("*.csv")
    )

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files found in data/raw."
        )

    for csv_file in csv_files:

        print(
            f"\nValidating dataset: {csv_file.name}"
        )

        dataframe = load_csv(csv_file)

        report_path = (
            REPORT_DIRECTORY
            / f"{csv_file.stem}_data_quality_report.md"
        )

        generate_data_quality_report(
            dataframe=dataframe,
            output_path=report_path,
        )

        print(
            f"Report generated: {report_path}"
        )


if __name__ == "__main__":
    main()