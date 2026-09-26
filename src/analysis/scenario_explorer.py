"""Constrained scenario exploration for casting quality decision support."""

from __future__ import annotations

from itertools import product
from typing import Dict, List

from src.analysis.scenario_constraints import validate_scenario_constraints
from src.models.decision.model_inference import predict_probability


def generate_scenarios(
    base_input: Dict,
    parameter_grid: Dict[str, List],
) -> List[Dict]:
    """Generate feasible scenarios from a baseline input and parameter grid."""

    scenarios = []

    features = list(parameter_grid.keys())
    values = [parameter_grid[feature] for feature in features]

    for combination in product(*values):
        scenario = dict(base_input)

        for feature, value in zip(features, combination):
            scenario[feature] = value

        validate_scenario_constraints(scenario)
        scenarios.append(scenario)

    return scenarios


def evaluate_scenarios(
    target: str,
    scenarios: List[Dict],
) -> List[Dict]:
    """Evaluate predicted risk for each feasible scenario."""

    results = []

    for scenario in scenarios:
        probability = predict_probability(
            target=target,
            casting_input=scenario,
        )

        results.append(
            {
                "scenario": scenario,
                "predicted_probability": probability,
            }
        )

    return results

def compare_scenarios_to_baseline(
    baseline_probability: float,
    evaluated_scenarios: List[Dict],
) -> List[Dict]:
    """Add baseline comparison information to evaluated scenarios."""

    results = []

    for item in evaluated_scenarios:
        scenario_probability = item["predicted_probability"]
        change = scenario_probability - baseline_probability

        if change < 0:
            interpretation = "Lower predicted risk"
        elif change > 0:
            interpretation = "Higher predicted risk"
        else:
            interpretation = "No predicted risk change"

        results.append(
            {
                **item,
                "baseline_probability": baseline_probability,
                "probability_change": change,
                "interpretation": interpretation,
            }
        )

    return results

def rank_scenarios(
    evaluated_scenarios: List[Dict],
    top_n: int = 5,
) -> List[Dict]:
    """Rank scenarios from lowest to highest predicted risk."""

    ranked = sorted(
        evaluated_scenarios,
        key=lambda item: item["predicted_probability"],
    )

    return ranked[:top_n]


def explore_scenarios(
    target: str,
    base_input: Dict,
    parameter_grid: Dict[str, List],
    top_n: int = 5,
) -> Dict:
    """Run constrained scenario exploration for one quality target."""

    validate_scenario_constraints(base_input)

    baseline_probability = predict_probability(
        target=target,
        casting_input=base_input,
    )

    scenarios = generate_scenarios(
        base_input=base_input,
        parameter_grid=parameter_grid,
    )

    evaluated = evaluate_scenarios(
        target=target,
        scenarios=scenarios,
    )

    compared = compare_scenarios_to_baseline(
        baseline_probability=baseline_probability,
        evaluated_scenarios=evaluated,
    )

    ranked = rank_scenarios(
        evaluated_scenarios=compared,
        top_n=top_n,
    )

    return {
        "target": target,
        "baseline": base_input,
        "baseline_probability": baseline_probability,
        "scenarios_evaluated": len(scenarios),
        "top_scenarios": ranked,
        "decision_note": (
            "Scenarios are ranked by model-predicted risk within "
            "dataset-derived prototype bounds. Results are intended "
            "for engineering decision support and scenario exploration, "
            "not as guaranteed optimal or causal process settings."
        ),
    }