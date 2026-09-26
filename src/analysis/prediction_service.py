"""High-level casting quality prediction service."""

from __future__ import annotations

from typing import Dict

from src.models.anomaly.anomaly_inference import predict_anomaly
from src.models.decision.decision_policy import classify_all_predictions
from src.models.decision.model_inference import predict_all_targets


def analyze_casting_input(casting_input: Dict) -> Dict:
    """Generate ML predictions, decision states, and anomaly analysis."""

    probabilities = predict_all_targets(casting_input)
    decisions = classify_all_predictions(probabilities)
    anomaly = predict_anomaly(casting_input)

    return {
        "input": casting_input,
        "predictions": decisions,
        "anomaly": anomaly,
    }