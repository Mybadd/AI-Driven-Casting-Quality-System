"""Unified engineering decision service for casting quality analysis."""

from __future__ import annotations

from typing import Dict, List, Optional

from src.analysis.engineering_review_service import get_engineering_review
from src.analysis.prediction_service import analyze_casting_input
from src.analysis.scenario_explorer import explore_scenarios


QUALITY_TARGETS = ("Defect", "Porosity", "Scrap")


def build_engineering_decision(
    casting_input: Dict,
    scenario_grids: Optional[Dict[str, Dict[str, List]]] = None,
) -> Dict:
    """
    Combine ML decisions, anomaly detection, engineering review,
    and optional constrained scenario exploration.

    The result is intended for engineer-facing analysis and dashboard use.
    It does not claim that ML explanations are proven causal relationships.
    Scenario exploration is model-based decision support, not guaranteed
    process optimization.
    """

    analysis = analyze_casting_input(casting_input)

    engineering_reviews = {}

    for target in QUALITY_TARGETS:
        engineering_reviews[target] = get_engineering_review(
            target=target,
            top_n=4,
        )

    scenario_exploration = {}

    if scenario_grids:
        for target, parameter_grid in scenario_grids.items():
            if target not in QUALITY_TARGETS:
                raise ValueError(
                    f"Unsupported scenario target: {target}. "
                    f"Supported targets: {QUALITY_TARGETS}"
                )

            scenario_exploration[target] = explore_scenarios(
                target=target,
                base_input=casting_input,
                parameter_grid=parameter_grid,
                top_n=5,
            )

    return {
        "input": casting_input,
        "quality_predictions": analysis["predictions"],
        "anomaly": analysis["anomaly"],
        "engineering_reviews": engineering_reviews,
        "scenario_exploration": scenario_exploration,
        "decision_note": (
            "Predictions and anomaly results are model-based signals. "
            "Engineering interpretation should be treated as review support, "
            "not as proof of causal relationships. "
            "Scenario exploration ranks feasible model-based scenarios "
            "within prototype bounds and does not represent guaranteed "
            "optimal process settings."
        ),
    }