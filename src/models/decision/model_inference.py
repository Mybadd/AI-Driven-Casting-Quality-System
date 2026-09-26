"""Reusable inference utilities for trained casting-quality models."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

from src.features.schema import INPUT_FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_DIRECTORY = PROJECT_ROOT / "models"


SUPPORTED_TARGETS = {
    "Defect": "defect",
    "Porosity": "porosity",
    "Scrap": "scrap",
}


def load_model(target: str):
    """Load the trained XGBoost pipeline for a quality target."""

    if target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Unsupported target: {target}. "
            f"Supported targets: {list(SUPPORTED_TARGETS)}"
        )

    model_path = (
        MODEL_DIRECTORY
        / SUPPORTED_TARGETS[target]
        / "xgboost_pipeline.joblib"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {model_path}"
        )

    return joblib.load(model_path)


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


def predict_probability(
    target: str,
    casting_input: Dict,
) -> float:
    """Return the predicted Yes probability for one casting input."""

    validate_input(casting_input)

    model = load_model(target)

    input_dataframe = pd.DataFrame(
        [casting_input],
        columns=INPUT_FEATURES,
    )

    probability = model.predict_proba(
        input_dataframe
    )[0, 1]

    return float(probability)


def predict_all_targets(
    casting_input: Dict,
) -> Dict[str, float]:
    """Return predicted Yes probabilities for all classification targets."""

    validate_input(casting_input)

    return {
        target: predict_probability(
            target=target,
            casting_input=casting_input,
        )
        for target in SUPPORTED_TARGETS
    }