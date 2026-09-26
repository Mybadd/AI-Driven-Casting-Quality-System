"""Combine ML, anomaly, and metallurgical signals into engineering quality analysis."""

from __future__ import annotations

from typing import Dict, List


def classify_prediction_confidence(
    probability: float,
    threshold: float,
    review_margin: float = 0.10,
) -> str:
    """
    Classify a prediction into a practical review state.

    High-risk:
        Probability is clearly above the decision threshold.

    Low-risk:
        Probability is clearly below the decision threshold.

    Review:
        Probability is close to the threshold and should not be treated
        as a confident decision.
    """
    lower_bound = threshold - review_margin
    upper_bound = threshold + review_margin

    if probability >= upper_bound:
        return "High-risk"
    if probability <= lower_bound:
        return "Low-risk"

    return "Review"


def build_quality_analysis(
    target: str,
    probability: float,
    threshold: float,
    anomaly_status: str,
    engineering_factors: List[Dict],
) -> Dict:
    """
    Combine model probability, decision policy, anomaly status,
    and engineering factors into a structured quality assessment.
    """

    confidence = classify_prediction_confidence(
        probability=probability,
        threshold=threshold,
    )

    if confidence == "High-risk":
        decision = "Further engineering investigation recommended."
    elif confidence == "Review":
        decision = "Prediction is close to the decision boundary; engineering review recommended."
    else:
        decision = "No high-risk signal from the classification model."

    if anomaly_status == "Anomaly":
        anomaly_interpretation = (
            "Process-condition pattern is outside the learned normal region "
            "and should be reviewed."
        )
    else:
        anomaly_interpretation = (
            "Process-condition pattern is within the learned normal region."
        )

    return {
        "target": target,
        "predicted_probability": round(float(probability), 4),
        "decision_threshold": round(float(threshold), 4),
        "prediction_state": confidence,
        "decision": decision,
        "anomaly_status": anomaly_status,
        "anomaly_interpretation": anomaly_interpretation,
        "engineering_factors": engineering_factors,
    }