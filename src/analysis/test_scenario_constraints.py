"""Tests for casting scenario feasibility constraints."""

import pytest

from src.analysis.scenario_constraints import (
    validate_numeric_bounds,
    validate_scenario_constraints,
)


def test_valid_numeric_bounds():
    validate_numeric_bounds("Pour_Temp", 710)
    validate_numeric_bounds("Mold_Moisture", 3.0)
    validate_numeric_bounds("Cooling_Time", 330)
    validate_numeric_bounds("Riser", 1.25)


def test_boundary_values_are_valid():
    validate_numeric_bounds("Pour_Temp", 680)
    validate_numeric_bounds("Pour_Temp", 740)
    validate_numeric_bounds("Mold_Moisture", 2.2)
    validate_numeric_bounds("Mold_Moisture", 4.0)
    validate_numeric_bounds("Cooling_Time", 250)
    validate_numeric_bounds("Cooling_Time", 450)
    validate_numeric_bounds("Riser", 1.05)
    validate_numeric_bounds("Riser", 1.55)


def test_out_of_range_value_is_rejected():
    with pytest.raises(ValueError):
        validate_numeric_bounds("Pour_Temp", 750)


def test_unknown_feature_is_rejected():
    with pytest.raises(ValueError):
        validate_numeric_bounds("Unknown_Feature", 10)


def test_valid_scenario_constraints():
    validate_scenario_constraints(
        {
            "Alloy": "Al-Si",
            "Pour_Temp": 710,
            "Mold_Moisture": 3.0,
            "Cooling_Time": 330,
            "Riser": 1.25,
        }
    )


def test_out_of_range_scenario_is_rejected():
    with pytest.raises(ValueError):
        validate_scenario_constraints(
            {
                "Pour_Temp": 750,
                "Mold_Moisture": 3.0,
                "Cooling_Time": 330,
                "Riser": 1.25,
            }
        )