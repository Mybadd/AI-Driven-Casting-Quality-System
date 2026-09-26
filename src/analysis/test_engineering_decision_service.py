"""Tests for the unified engineering decision service."""

import pytest

from src.analysis.engineering_decision_service import (
    build_engineering_decision,
)


BASE_INPUT = {
    "Alloy": "Al-Si",
    "Pour_Temp": 710,
    "Mold_Moisture": 3.0,
    "Cooling_Time": 330,
    "Riser": 1.25,
}


def test_build_engineering_decision():
    result = build_engineering_decision(BASE_INPUT)

    assert result["input"] == BASE_INPUT
    assert set(result["quality_predictions"]) == {
        "Defect",
        "Porosity",
        "Scrap",
    }
    assert "anomaly" in result
    assert set(result["engineering_reviews"]) == {
        "Defect",
        "Porosity",
        "Scrap",
    }
    assert result["scenario_exploration"] == {}
    assert "decision_note" in result


def test_build_engineering_decision_with_scenario_exploration():
    result = build_engineering_decision(
        BASE_INPUT,
        scenario_grids={
            "Defect": {
                "Pour_Temp": [700, 710],
                "Mold_Moisture": [2.8, 3.0],
            }
        },
    )

    assert "Defect" in result["scenario_exploration"]

    exploration = result["scenario_exploration"]["Defect"]

    assert exploration["target"] == "Defect"
    assert exploration["scenarios_evaluated"] == 4
    assert len(exploration["top_scenarios"]) == 4
    assert "baseline_probability" in exploration


def test_scenario_exploration_supports_multiple_targets():
    result = build_engineering_decision(
        BASE_INPUT,
        scenario_grids={
            "Defect": {
                "Pour_Temp": [710, 720],
            },
            "Porosity": {
                "Mold_Moisture": [2.8, 3.0],
            },
        },
    )

    assert set(result["scenario_exploration"]) == {
        "Defect",
        "Porosity",
    }

    assert (
        result["scenario_exploration"]["Defect"]["scenarios_evaluated"]
        == 2
    )

    assert (
        result["scenario_exploration"]["Porosity"]["scenarios_evaluated"]
        == 2
    )


def test_invalid_scenario_target_is_rejected():
    with pytest.raises(ValueError):
        build_engineering_decision(
            BASE_INPUT,
            scenario_grids={
                "UnknownTarget": {
                    "Pour_Temp": [700, 710],
                }
            },
        )