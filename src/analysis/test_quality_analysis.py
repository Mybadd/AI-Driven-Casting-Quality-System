"""Tests for the engineering quality analysis layer."""

from src.analysis.quality_analysis import (
    build_quality_analysis,
    classify_prediction_confidence,
)


def test_high_risk_prediction():
    state = classify_prediction_confidence(
        probability=0.80,
        threshold=0.50,
    )

    assert state == "High-risk"


def test_low_risk_prediction():
    state = classify_prediction_confidence(
        probability=0.20,
        threshold=0.50,
    )

    assert state == "Low-risk"


def test_boundary_prediction_requires_review():
    state = classify_prediction_confidence(
        probability=0.55,
        threshold=0.50,
    )

    assert state == "Review"


def test_quality_analysis_structure():
    result = build_quality_analysis(
        target="Defect",
        probability=0.72,
        threshold=0.48,
        anomaly_status="Anomaly",
        engineering_factors=[
            {
                "feature": "Mold_Moisture",
                "interpretation": "Review moisture-related process conditions.",
            }
        ],
    )

    assert result["target"] == "Defect"
    assert result["prediction_state"] == "High-risk"
    assert result["anomaly_status"] == "Anomaly"
    assert len(result["engineering_factors"]) == 1
    assert "investigation" in result["decision"].lower()


if __name__ == "__main__":
    test_high_risk_prediction()
    test_low_risk_prediction()
    test_boundary_prediction_requires_review()
    test_quality_analysis_structure()

    print("All quality analysis tests passed.")