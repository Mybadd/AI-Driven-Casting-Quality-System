
"""Automated API tests for the casting quality system."""

from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)

VALID_CASTING_INPUT = {
    "Alloy": "Al-Si",
    "Pour_Temp": 710,
    "Mold_Moisture": 3.0,
    "Cooling_Time": 330,
    "Riser": 1.25,
}


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint():
    response = client.post("/api/predict", json=VALID_CASTING_INPUT)

    assert response.status_code == 200

    result = response.json()
    assert "predictions" in result
    assert "anomaly" in result

    for target in ("Defect", "Porosity", "Scrap"):
        assert target in result["predictions"]


def test_analyze_endpoint():
    response = client.post("/api/analyze", json=VALID_CASTING_INPUT)

    assert response.status_code == 200

    result = response.json()
    assert "quality_predictions" in result
    assert "engineering_reviews" in result
    assert "anomaly" in result
    assert "scenario_exploration" in result
    assert "decision_note" in result



def test_scenarios_endpoint():
    response = client.post(
        "/api/scenarios",
        json={
            "target": "Defect",
            "base_input": VALID_CASTING_INPUT,
            "parameter_grid": {
                "Pour_Temp": [700, 710, 720],
                "Mold_Moisture": [2.8, 3.0, 3.2],
            },
            "top_n": 5,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["target"] == "Defect"
    assert result["scenarios_evaluated"] == 9
    assert len(result["top_scenarios"]) <= 5
    assert "baseline_probability" in result
    assert "decision_note" in result

    for scenario in result["top_scenarios"]:
        assert "scenario" in scenario
        assert "predicted_probability" in scenario
        assert "probability_change" in scenario
        assert "interpretation" in scenario


def test_predict_rejects_missing_required_input():
    incomplete_input = {
        "Alloy": "Al-Si",
        "Pour_Temp": 710,
        "Mold_Moisture": 3.0,
        "Cooling_Time": 330,
        # Riser intentionally omitted
    }

    response = client.post("/api/predict", json=incomplete_input)

    assert response.status_code == 422


def test_scenarios_rejects_unsupported_target():
    response = client.post(
        "/api/scenarios",
        json={
            "target": "UnsupportedTarget",
            "base_input": VALID_CASTING_INPUT,
            "parameter_grid": {"Pour_Temp": [700, 710]},
            "top_n": 5,
        },
    )

    assert response.status_code == 422


def test_scenarios_rejects_out_of_bounds_parameter():
    response = client.post(
        "/api/scenarios",
        json={
            "target": "Defect",
            "base_input": VALID_CASTING_INPUT,
            "parameter_grid": {"Pour_Temp": [650, 710]},
            "top_n": 5,
        },
    )

    assert response.status_code == 422
