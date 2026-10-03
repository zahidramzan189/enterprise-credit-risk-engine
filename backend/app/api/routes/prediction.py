from fastapi import APIRouter, HTTPException
from backend.app.schemas.prediction import PredictionRequest
from backend.app.schemas.responses import PredictionResponse, ExplanationResponse
from backend.app.services.prediction_service import predict
from backend.app.services.explanation_service import explain

router = APIRouter(prefix="/api/v1", tags=["risk"])

@router.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    try:
        return predict(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

@router.post("/explain", response_model=ExplanationResponse)
def explain_endpoint(request: PredictionRequest):
    try:
        return explain(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
