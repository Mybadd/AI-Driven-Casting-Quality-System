"""Integration test for ML quality analysis and engineering interpretation."""

from src.analysis.metallurgical_rules import build_quality_review
from src.analysis.quality_analysis import build_quality_analysis


def test_quality_analysis_with_engineering_review():
    engineering_review = build_quality_review(
        target="Defect",
        influential_features=[
            ("Mold_Moisture", 0.2092),
            ("Riser", 0.1899),
            ("Pour_Temp", 0.1628),
        ],
    )

    result = build_quality_analysis(
        target="Defect",
        probability=0.72,
        threshold=0.48,
        anomaly_status="Anomaly",
        engineering_factors=engineering_review["factors"],
    )

    assert result["target"] == "Defect"
    assert result["prediction_state"] == "High-risk"
    assert result["anomaly_status"] == "Anomaly"

    assert len(result["engineering_factors"]) == 3

    factor_names = [
        factor["factor"]
        for factor in result["engineering_factors"]
    ]

    assert "Mold Moisture" in factor_names
    assert "Riser" in factor_names
    assert "Pour Temperature" in factor_names

    assert result["decision"] == (
        "Further engineering investigation recommended."
    )


if __name__ == "__main__":
    test_quality_analysis_with_engineering_review()

    print("Quality analysis integration test passed.")