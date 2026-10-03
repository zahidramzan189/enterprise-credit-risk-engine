from pathlib import Path
from fastapi import HTTPException
from backend.app.config import MODEL_PATH, PREPROCESSOR_PATH

def ensure_model_available():
    missing = [p for p in [MODEL_PATH, PREPROCESSOR_PATH] if not Path(p).exists()]
    if missing:
        raise HTTPException(503, "ML artifacts unavailable. Run python -m ml.train first.")
