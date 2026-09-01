"""
Run and verify the data splitting module.
"""

from pathlib import Path

from src.data.load_data import load_csv
from src.data.split_data import SUPPORTED_TARGETS, split_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"


def get_dataset_path() -> Path:
    """
    Find the CSV dataset inside the raw data directory.

    Returns
    -------
    Path
        Path to the dataset CSV file.
    """

    csv_files = list(DATA_DIRECTORY.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {DATA_DIRECTORY}"
        )

    if len(csv_files) > 1:
        raise ValueError(
            "Multiple CSV files found. "
            f"Expected exactly one dataset, found: {len(csv_files)}"
        )

    return csv_files[0]


def main() -> None:
    """
    Load the dataset and verify splitting for every supported target.
    """

    dataset_path = get_dataset_path()

    print(f"\nLoading dataset: {dataset_path.name}")

    dataframe = load_csv(dataset_path)

    print("\nDATA SPLITTING VERIFICATION")
    print("=" * 50)

    for target in SUPPORTED_TARGETS:

        X_train, X_test, y_train, y_test = split_dataset(
            dataframe=dataframe,
            target=target,
        )

        print(f"\nTarget: {target}")
        print("-" * 50)

        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Features used: {list(X_train.columns)}")

        if target != "Yield":

            print("\nTraining class distribution:")
            print(
                y_train.value_counts(
                    normalize=True
                ).round(3)
            )

            print("\nTesting class distribution:")
            print(
                y_test.value_counts(
                    normalize=True
                ).round(3)
            )

    print("\nDATA SPLITTING COMPLETED SUCCESSFULLY")


if __name__ == "__main__":
    main()