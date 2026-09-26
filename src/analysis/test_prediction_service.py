from src.analysis.prediction_service import analyze_casting_input


def test_prediction_service():
    casting_input = {
        "Alloy": "Al-Si",
        "Pour_Temp": 710,
        "Mold_Moisture": 3.0,
        "Cooling_Time": 330,
        "Riser": 1.25,
    }

    result = analyze_casting_input(casting_input)

    assert "input" in result
    assert "predictions" in result

    assert set(result["predictions"]) == {
        "Defect",
        "Porosity",
        "Scrap",
    }

    for prediction in result["predictions"].values():
        assert 0 <= prediction["probability"] <= 1
        assert 0 <= prediction["threshold"] <= 1
        assert prediction["state"] in {
            "High-risk",
            "Review",
            "Low-risk",
        }


if __name__ == "__main__":
    test_prediction_service()
    print("Prediction service test passed.")