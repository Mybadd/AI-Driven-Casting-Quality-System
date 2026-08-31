from pathlib import Path

import pandas as pd


PERCENTAGE_COLUMNS = [
    "Yield",
]


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV dataset safely.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file was not found: {file_path}"
        )

    if file_path.suffix.lower() != ".csv":
        raise ValueError(
            f"Expected a CSV file, received: {file_path.suffix}"
        )

    try:
        dataframe = pd.read_csv(file_path)
    except Exception as error:
        raise RuntimeError(
            f"Failed to load dataset: {file_path}"
        ) from error

    if dataframe.empty:
        raise ValueError(
            f"Dataset is empty: {file_path}"
        )

    dataframe = convert_percentage_columns(dataframe)

    return dataframe


def convert_percentage_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert percentage-formatted columns such as
    '60.90%' into numeric values such as 60.90.

    The raw CSV file is never modified.
    """

    dataframe = dataframe.copy()

    for column in PERCENTAGE_COLUMNS:

        if column not in dataframe.columns:
            continue

        dataframe[column] = (
            dataframe[column]
            .astype(str)
            .str.strip()
            .str.rstrip("%")
        )

        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    return dataframe