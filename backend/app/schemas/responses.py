from typing import Any
from pydantic import BaseModel, Field

class PredictionResponse(BaseModel):
    default_probability: float = Field(ge=0, le=1)
    default_probability_percent: float = Field(ge=0, le=100)
    risk_score: int = Field(ge=0, le=100)
    risk_level: str
    prediction: int
    prediction_label: str
    model_type: str

class ExplanationResponse(BaseModel):
    prediction: PredictionResponse
    base_value: float
    key_factors: list[dict[str, Any]]
    positive_risk_contributors: list[dict[str, Any]]
    negative_risk_contributors: list[dict[str, Any]]
    human_readable: str
