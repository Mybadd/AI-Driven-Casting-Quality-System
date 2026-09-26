"""Decision policy for casting-quality classification outputs."""

from __future__ import annotations

from typing import Dict


DECISION_THRESHOLDS = {
    "Defect": 0.48,
    "Porosity": 0.47,
    "Scrap": 0.50,
}

REVIEW_MARGIN = 0.10


def classify_prediction(
    target: str,
    probability: float,
) -> Dict[str, object]:
    """Classify a predicted probability using the selected operating threshold."""

    if target not in DECISION_THRESHOLDS:
        raise ValueError(
            f"Unsupported target: {target}. "
            f"Supported targets: {list(DECISION_THRESHOLDS)}"
        )

    threshold = DECISION_THRESHOLDS[target]

    if probability >= threshold + REVIEW_MARGIN:
        state = "High-risk"
        decision = "Further engineering investigation recommended."
    elif probability <= threshold - REVIEW_MARGIN:
        state = "Low-risk"
        decision = "No high-risk signal from the classification model."
    else:
        state = "Review"
        decision = (
            "Prediction is close to the decision boundary; "
            "engineering review recommended."
        )

    return {
        "target": target,
        "probability": round(float(probability), 4),
        "threshold": threshold,
        "state": state,
        "decision": decision,
    }


def classify_all_predictions(
    probabilities: Dict[str, float],
) -> Dict[str, Dict[str, object]]:
    """Apply the decision policy to all target probabilities."""

    return {
        target: classify_prediction(
            target=target,
            probability=probability,
        )
        for target, probability in probabilities.items()
    }   