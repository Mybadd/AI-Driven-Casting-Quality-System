from typing import Dict

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from src.analysis.prediction_service import analyze_casting_input
from fastapi.middleware.cors import CORSMiddleware
from src.analysis.engineering_decision_service import build_engineering_decision
from src.analysis.scenario_explorer import explore_scenarios
from src.analysis.history_service import save_analysis_record
from src.analysis.history_service import list_analysis_records, get_analysis_record
from src.analysis.history_service import (
    save_analysis_record,
    list_analysis_records,
    get_analysis_record,
)
app = FastAPI(
    title="AI-Driven Casting Quality System",
    description="Backend API for casting quality analysis and engineering decision support.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
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
    casting_input_data = casting_input.model_dump()
    result = build_engineering_decision(casting_input_data)

    save_analysis_record(casting_input_data, result)

    return result

@app.post("/api/scenarios")
def explore_casting_scenarios(request: ScenarioRequest) -> Dict:
    try:
        return explore_scenarios(
            target=request.target,
            base_input=request.base_input.model_dump(),
            parameter_grid=request.parameter_grid,
            top_n=request.top_n,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
@app.post("/api/history")
def save_history(payload: dict) -> Dict:
    casting_input = payload.get("input")
    result = payload.get("result")

    if not isinstance(casting_input, dict) or not isinstance(result, dict):
        raise HTTPException(
            status_code=422,
            detail="Request must contain dictionary fields 'input' and 'result'.",
        )

    return save_analysis_record(casting_input, result)


@app.get("/api/history")
def get_history(limit: int = 50, offset: int = 0) -> Dict:
    if limit < 1 or offset < 0:
        raise HTTPException(
            status_code=422,
            detail="limit must be positive and offset must be non-negative.",
        )

    return list_analysis_records(limit=limit, offset=offset)


@app.get("/api/history/{record_id}")
def get_history_item(record_id: int) -> Dict:
    record = get_analysis_record(record_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Analysis history record not found.",
        )

    return record