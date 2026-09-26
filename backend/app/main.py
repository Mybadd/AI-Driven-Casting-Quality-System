from typing import Dict

from fastapi import FastAPI
from pydantic import BaseModel

from src.analysis.prediction_service import analyze_casting_input

from src.analysis.engineering_decision_service import build_engineering_decision
from src.analysis.scenario_explorer import explore_scenarios
app = FastAPI(
    title="AI-Driven Casting Quality System",
    description="Backend API for casting quality analysis and engineering decision support.",
    version="1.0.0",
)


class CastingInput(BaseModel):
    Alloy: str
    Pour_Temp: float
    Mold_Moisture: float
    Cooling_Time: float
    Riser: float

class ScenarioRequest(BaseModel):
    target: str
    base_input: CastingInput
    parameter_grid: Dict[str, list]
    top_n: int = 5



@app.get("/")
def root():
    return {
        "message": "AI-Driven Casting Quality System API",
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.post("/api/predict")
def predict_casting(casting_input: CastingInput) -> Dict:
    return analyze_casting_input(casting_input.model_dump())

@app.post("/api/analyze")
def analyze_casting(casting_input: CastingInput) -> Dict:
    return build_engineering_decision(casting_input.model_dump())
j
@app.post("/api/scenarios")
def explore_casting_scenarios(request: ScenarioRequest) -> Dict:
    return explore_scenarios(
        target=request.target,
        base_input=request.base_input.model_dump(),
        parameter_grid=request.parameter_grid,
        top_n=request.top_n,
    )