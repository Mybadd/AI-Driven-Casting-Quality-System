"""What-if scenario analysis for casting process conditions."""

from __future__ import annotations

from typing import Dict, List


SUPPORTED_FEATURES = [
    "Alloy",
    "Pour_Temp",
    "Mold_Moisture",
    "Cooling_Time",
    "Riser",
]


def validate_scenario(scenario: Dict) -> None:
    """Validate that a what-if scenario contains only supported inputs."""

    missing = [
        feature
        for feature in SUPPORTED_FEATURES
        if feature not in scenario
    ]

    if missing:
        raise ValueError(
            f"Scenario is missing required features: {missing}"
        )


def create_scenario(
    base_input: Dict,
    changes: Dict,
) -> Dict:
    """
    Create a what-if scenario by applying controlled changes
    to an existing casting input.
    """

    validate_scenario(base_input)

    unknown_features = [
        feature
        for feature in changes
        if feature not in SUPPORTED_FEATURES
    ]

    if unknown_features:
        raise ValueError(
            f"Unsupported scenario features: {unknown_features}"
        )

    scenario = base_input.copy()
    scenario.update(changes)

    validate_scenario(scenario)

    return scenario


def compare_probabilities(
    base_probability: float,
    scenario_probability: float,
) -> Dict:
    """Compare model probabilities between baseline and scenario."""

    change = scenario_probability - base_probability

    if change < 0:
        direction = "Lower predicted risk"
    elif change > 0:
        direction = "Higher predicted risk"
    else:
        direction = "No predicted change"

    return {
        "base_probability": round(float(base_probability), 4),
        "scenario_probability": round(float(scenario_probability), 4),
        "probability_change": round(float(change), 4),
        "direction": direction,
    }


def build_scenario_summary(
    target: str,
    base_input: Dict,
    scenario_input: Dict,
    base_probability: float,
    scenario_probability: float,
) -> Dict:
    """Build a structured what-if analysis result."""

    comparison = compare_probabilities(
        base_probability=base_probability,
        scenario_probability=scenario_probability,
    )

    changed_features = {
        feature: {
            "base": base_input[feature],
            "scenario": scenario_input[feature],
        }
        for feature in SUPPORTED_FEATURES
        if base_input[feature] != scenario_input[feature]
    }

    return {
        "target": target,
        "changed_features": changed_features,
        "comparison": comparison,
        "interpretation": (
            "This is a model-based scenario comparison. "
            "It does not establish causal effects or guarantee "
            "the resulting casting quality."
        ),
    }
from src.models.decision.model_inference import predict_probability


def analyze_what_if(
    target: str,
    base_input: Dict,
    changes: Dict,
) -> Dict:
    """
    Run a model-based what-if analysis for one quality target.

    The result compares the baseline input with a scenario created by
    applying the requested parameter changes.
    """

    base_probability = predict_probability(
        target=target,
        casting_input=base_input,
    )

    scenario_input = create_scenario(
        base_input=base_input,
        changes=changes,
    )

    scenario_probability = predict_probability(
        target=target,
        casting_input=scenario_input,
    )

    return build_scenario_summary(
        target=target,
        base_input=base_input,
        scenario_input=scenario_input,
        base_probability=base_probability,
        scenario_probability=scenario_probability,
    )