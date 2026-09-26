"""Prototype feasibility constraints for casting scenario exploration."""

from __future__ import annotations


NUMERIC_BOUNDS = {
    "Pour_Temp": (680, 740),
    "Mold_Moisture": (2.2, 4.0),
    "Cooling_Time": (250, 450),
    "Riser": (1.05, 1.55),
}


def validate_numeric_bounds(
    feature: str,
    value: float,
) -> None:
    """Validate a numeric scenario value against dataset-derived bounds."""

    if feature not in NUMERIC_BOUNDS:
        raise ValueError(
            f"Unsupported constrained feature: {feature}"
        )

    minimum, maximum = NUMERIC_BOUNDS[feature]

    if not minimum <= value <= maximum:
        raise ValueError(
            f"{feature}={value} is outside the prototype feasible range "
            f"[{minimum}, {maximum}]."
        )


def validate_scenario_constraints(
    scenario_input: dict,
) -> None:
    """Validate all constrained numeric inputs in a scenario."""

    for feature, value in scenario_input.items():
        if feature in NUMERIC_BOUNDS:
            validate_numeric_bounds(feature, value)