INPUT_FEATURES = [
    "Alloy",
    "Pour_Temp",
    "Mold_Moisture",
    "Cooling_Time",
    "Riser",
]

CATEGORICAL_FEATURES = [
    "Alloy",
]

NUMERICAL_FEATURES = [
    "Pour_Temp",
    "Mold_Moisture",
    "Cooling_Time",
    "Riser",
]

IDENTIFIER_COLUMNS = [
    "Batch",
]

CLASSIFICATION_TARGETS = [
    "Defect",
    "Porosity",
    "Scrap",
]

REGRESSION_TARGET = "Yield"

ALL_TARGETS = (
    CLASSIFICATION_TARGETS
    + [REGRESSION_TARGET]
)