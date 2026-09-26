"""Tests for the what-if scenario analysis layer."""

from src.analysis.what_if import (
    analyze_what_if,
    build_scenario_summary,
    compare_probabilities,
    create_scenario,
    validate_scenario,
)


BASE_INPUT = {
    "Alloy": "Al-Si",
    "Pour_Temp": 710,
    "Mold_Moisture": 3.0,
    "Cooling_Time": 330,
    "Riser": 1.25,
}


def test_valid_scenario():
    validate_scenario(BASE_INPUT)


def test_create_scenario():
    scenario = create_scenario(
        base_input=BASE_INPUT,
        changes={
            "Pour_Temp": 700,
            "Cooling_Time": 350,
        },
    )

    assert scenario["Pour_Temp"] == 700
    assert scenario["Cooling_Time"] == 350
    assert scenario["Mold_Moisture"] == 3.0
    assert scenario["Riser"] == 1.25


def test_reject_unknown_change():
    try:
        create_scenario(
            base_input=BASE_INPUT,
            changes={"Unknown_Feature": 10},
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for unknown feature.")


def test_probability_comparison():
    result = compare_probabilities(
        base_probability=0.70,
        scenario_probability=0.55,
    )

    assert result["probability_change"] == -0.15
    assert result["direction"] == "Lower predicted risk"


def test_scenario_summary():
    scenario = create_scenario(
        base_input=BASE_INPUT,
        changes={"Cooling_Time": 350},
    )

    result = build_scenario_summary(
        target="Defect",
        base_input=BASE_INPUT,
        scenario_input=scenario,
        base_probability=0.70,
        scenario_probability=0.55,
    )

    assert result["target"] == "Defect"
    assert "Cooling_Time" in result["changed_features"]
    assert result["comparison"]["direction"] == "Lower predicted risk"
    assert "causal effects" in result["interpretation"]


def test_analyze_what_if_integration():
    result = analyze_what_if(
        target="Porosity",
        base_input=BASE_INPUT,
        changes={
            "Pour_Temp": 720,
            "Mold_Moisture": 3.4,
            "Cooling_Time": 350,
        },
    )

    assert result["target"] == "Porosity"

    assert result["changed_features"]["Pour_Temp"] == {
        "base": 710,
        "scenario": 720,
    }

    assert result["changed_features"]["Mold_Moisture"] == {
        "base": 3.0,
        "scenario": 3.4,
    }

    assert result["changed_features"]["Cooling_Time"] == {
        "base": 330,
        "scenario": 350,
    }

    assert 0.0 <= result["comparison"]["base_probability"] <= 1.0
    assert 0.0 <= result["comparison"]["scenario_probability"] <= 1.0

    assert result["comparison"]["direction"] in {
        "Lower predicted risk",
        "Higher predicted risk",
        "No change",
    }

    assert "does not establish causal effects" in result["interpretation"]


if __name__ == "__main__":
    test_valid_scenario()
    test_create_scenario()
    test_reject_unknown_change()
    test_probability_comparison()
    test_scenario_summary()
    test_analyze_what_if_integration()

    print("All what-if tests passed.")