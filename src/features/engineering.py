"""
Engineering-motivated feature transformations.

The engineered features are derived only from the approved raw ML inputs.
No external process variables or identifier columns are introduced.
"""

from __future__ import annotations

import pandas as pd


ENGINEERED_FEATURE_NAMES = [
    "Pour_Temp_sq",
    "Mold_Moisture_sq",
    "Cooling_Time_sq",
    "Riser_sq",
    "Pour_Temp_x_Mold_Moisture",
    "Pour_Temp_x_Cooling_Time",
    "Mold_Moisture_x_Cooling_Time",
    "Riser_div_Cooling_Time",
]


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a copy of the input dataframe.

    The transformation uses only the approved process variables:
    Pour_Temp, Mold_Moisture, Cooling_Time, and Riser.

    Parameters
    ----------
    dataframe:
        Input dataframe containing the approved numerical features.

    Returns
    -------
    pd.DataFrame
        Copy of the dataframe with engineered features appended.

    Raises
    ------
    KeyError
        If a required numerical feature is missing.
    """

    required_columns = [
        "Pour_Temp",
        "Mold_Moisture",
        "Cooling_Time",
        "Riser",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise KeyError(
            f"Missing required columns: {missing_columns}"
        )

    result = dataframe.copy()

    result["Pour_Temp_sq"] = result["Pour_Temp"] ** 2
    result["Mold_Moisture_sq"] = result["Mold_Moisture"] ** 2
    result["Cooling_Time_sq"] = result["Cooling_Time"] ** 2
    result["Riser_sq"] = result["Riser"] ** 2

    result["Pour_Temp_x_Mold_Moisture"] = (
        result["Pour_Temp"] * result["Mold_Moisture"]
    )

    result["Pour_Temp_x_Cooling_Time"] = (
        result["Pour_Temp"] * result["Cooling_Time"]
    )

    result["Mold_Moisture_x_Cooling_Time"] = (
        result["Mold_Moisture"] * result["Cooling_Time"]
    )

    result["Riser_div_Cooling_Time"] = (
        result["Riser"] / result["Cooling_Time"]
    )

    return result