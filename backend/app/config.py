import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

_allowed = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:8080,"
    "http://127.0.0.1:8080,"
    "http://localhost:8000,"
    "http://127.0.0.1:8000"
)

ALLOWED_ORIGINS = [
    x.strip()
    for x in _allowed.split(",")
    if x.strip()
]

MODEL_PATH = os.getenv(
    "MODEL_PATH",
    str(ROOT / "models/model/credit_risk_model.joblib")
)

PREPROCESSOR_PATH = os.getenv(
    "PREPROCESSOR_PATH",
    str(ROOT / "models/preprocessor/preprocessor.joblib")
)

CALIBRATED_MODEL_PATH = os.getenv(
    "CALIBRATED_MODEL_PATH",
    str(ROOT / "models/calibration/calibrated_model.joblib")
)