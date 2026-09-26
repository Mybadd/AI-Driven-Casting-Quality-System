"""
Metallurgical interpretation rules for the AI-Driven Casting Quality System.

This module does not establish physical causality.
It converts model findings into engineering-oriented review guidance.
"""

from __future__ import annotations


def interpret_feature(
    feature_name: str,
    value: float,
    target: str,
) -> dict:
    """
    Convert an influential model feature into engineering review guidance.

    The output is advisory and should not be interpreted as proof of causality.
    """

    if feature_name == "Mold_Moisture":
        return {
            "factor": "Mold Moisture",
            "interpretation": (
                "Mold moisture is an important process condition to review. "
                "Changes in moisture may influence mold condition, gas-related "
                "behavior, and the environment surrounding the casting."
            ),
            "review": (
                "Review mold moisture against the established process range "
                "and inspect related molding conditions."
            ),
        }

    if feature_name == "Pour_Temp":
        return {
            "factor": "Pour Temperature",
            "interpretation": (
                "Pour temperature is an important thermal process condition. "
                "Changes in pouring temperature can affect filling and solidification "
                "conditions."
            ),
            "review": (
                "Review the pouring temperature and its consistency relative "
                "to the intended process window."
            ),
        }

    if feature_name == "Cooling_Time":
        return {
            "factor": "Cooling Time",
            "interpretation": (
                "Cooling time is related to the thermal history of the casting. "
                "Differences in cooling conditions may influence solidification "
                "behavior and resulting quality."
            ),
            "review": (
                "Review cooling-time consistency and compare the batch with "
                "the normal process window."
            ),
        }

    if feature_name == "Riser":
        return {
            "factor": "Riser",
            "interpretation": (
                "Riser condition is relevant to feeding and solidification "
                "behavior. Its setting should therefore be reviewed when the "
                "model identifies it as an influential factor."
            ),
            "review": (
                "Review the riser setting and its suitability for the current "
                "alloy and casting process."
            ),
        }

    if feature_name == "Alloy":
        return {
            "factor": "Alloy",
            "interpretation": (
                "Alloy selection changes the material and solidification context "
                "of the casting process."
            ),
            "review": (
                "Review whether the observed behavior is consistent with the "
                "expected behavior for the selected alloy."
            ),
        }

    return {
        "factor": feature_name,
        "interpretation": (
            "This feature was identified by the model as influential and "
            "should be considered during engineering review."
        ),
        "review": (
            "Review this process condition together with other available "
            "quality information."
        ),
    }


def build_quality_review(
    target: str,
    influential_features: list[tuple[str, float]],
) -> dict:
    """
    Build an engineering-oriented review summary from influential features.

    Parameters
    ----------
    target:
        Quality target such as Defect, Porosity, or Scrap.

    influential_features:
        List of (feature_name, importance) pairs ordered by importance.

    Returns
    -------
    dict
        Structured engineering review information.
    """

    factors = []

    for feature_name, importance in influential_features:
        interpretation = interpret_feature(
            feature_name=feature_name,
            value=importance,
            target=target,
        )

        factors.append(
            {
                "factor": interpretation["factor"],
                "importance": float(importance),
                "interpretation": interpretation["interpretation"],
                "review": interpretation["review"],
            }
        )

    return {
        "target": target,
        "statement": (
            "The following factors were identified as influential by the "
            "trained model. They are presented as likely contributing factors "
            "for engineering review, not as proven causes."
        ),
        "factors": factors,
    }