"""Tests for constrained casting scenario exploration."""

import pytest

from src.analysis.scenario_explorer import (
    generate_scenarios,
    rank_scenarios,
    explore_scenarios,
)


BASE_INPUT = {
    "Alloy": "Al-Si",
    "Pour_Temp": 710,
    "Mold_Moisture": 3.0,
    "Cooling_Time": 330,
    "Riser": 1.25,
}


def test_generate_scenarios():
    scenarios = generate_scenarios(
        base_input=BASE_INPUT,
        parameter_grid={
            "Pour_Temp": [700, 710],
            "Mold_Moisture": [2.8, 3.0],
        },
    )

    assert len(scenarios) == 4

    for scenario in scenarios:
        assert scenario["Alloy"] == "Al-Si"
        assert scenario["Cooling_Time"] == 330
        assert scenario["Riser"] == 1.25


def test_generate_scenarios_rejects_invalid_values():
    with pytest.raises(ValueError):
        generate_scenarios(
            base_input=BASE_INPUT,
            parameter_grid={
                "Pour_Temp": [700, 750],
            },
        )


def test_rank_scenarios():
    evaluated = [
        {
            "scenario": {"Pour_Temp": 720},
            "predicted_probability": 0.60,
        },
        {
            "scenario": {"Pour_Temp": 700},
            "predicted_probability": 0.40,
        },
        {
            "scenario": {"Pour_Temp": 710},
            "predicted_probability": 0.50,
        },
    ]

    ranked = rank_scenarios(evaluated, top_n=2)

    assert len(ranked) == 2
    assert ranked[0]["predicted_probability"] == 0.40
    assert ranked[1]["predicted_probability"] == 0.50


def test_rank_scenarios_returns_all_when_top_n_is_large():
    evaluated = [
        {
            "scenario": {"Pour_Temp": 700},
            "predicted_probability": 0.40,
        },
        {
            "scenario": {"Pour_Temp": 710},
            "predicted_probability": 0.50,
        },
    ]

    ranked = rank_scenarios(evaluated, top_n=10)

    assert len(ranked) == 2


def test_explore_scenarios():
    result = explore_scenarios(
        target="Defect",
        base_input=BASE_INPUT,
        parameter_grid={
            "Pour_Temp": [700, 710],
            "Mold_Moisture": [2.8, 3.0],
        },
        top_n=3,
    )

    assert result["target"] == "Defect"
    assert result["scenarios_evaluated"] == 4
    assert len(result["top_scenarios"]) == 3
    assert "decision_note" in result
    assert "baseline_probability" in result
    assert "probability_change" in result["top_scenarios"][0]
    assert "interpretation" in result["top_scenarios"][0]

def test_compare_scenarios_to_baseline():
    from src.analysis.scenario_explorer import compare_scenarios_to_baseline

    evaluated = [
        {
            "scenario": {"Pour_Temp": 700},
            "predicted_probability": 0.40,
        },
        {
            "scenario": {"Pour_Temp": 710},
            "predicted_probability": 0.50,
        },
        {
            "scenario": {"Pour_Temp": 720},
            "predicted_probability": 0.45,
        },
    ]

    compared = compare_scenarios_to_baseline(
        baseline_probability=0.45,
        evaluated_scenarios=evaluated,
    )

    assert compared[0]["probability_change"] == pytest.approx(-0.05)
    assert compared[0]["interpretation"] == "Lower predicted risk"

    assert compared[1]["probability_change"] == pytest.approx(0.05)
    assert compared[1]["interpretation"] == "Higher predicted risk"

    assert compared[2]["probability_change"] == pytest.approx(0.0)
    assert compared[2]["interpretation"] == "No predicted risk change"