"""Reusable anomaly detection inference utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from src.features.schema import INPUT_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "anomaly"

MODEL_PATH = MODEL_DIRECTORY / "isolation_forest_pipeline.joblib"


def load_anomaly_model():
    """Load the trained Isolation Forest pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Anomaly model artifact not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def validate_input(casting_input: Dict) -> None:
    """Validate that all approved model inputs are present."""

    missing = [
        feature
        for feature in INPUT_FEATURES
        if feature not in casting_input
    ]

    if missing:
        raise ValueError(
            f"Missing required input features: {missing}"
        )


def predict_anomaly(casting_input: Dict) -> Dict[str, object]:
    """Return anomaly status and score for one casting input."""

    validate_input(casting_input)

    model = load_anomaly_model()

    input_dataframe = pd.DataFrame(
        [casting_input],
        columns=INPUT_FEATURES,
    )

    prediction = int(model.predict(input_dataframe)[0])
    score = float(model.decision_function(input_dataframe)[0])

    status = "Anomaly" if prediction == -1 else "Normal"

    return {
        "status": status,
        "score": round(score, 6),
    }