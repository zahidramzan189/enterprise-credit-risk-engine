import json
from fastapi import APIRouter, HTTPException
from ml.config import METADATA_PATH
router = APIRouter(prefix="/api/v1", tags=["model"])

@router.get("/model-info")
def model_info():
    if not METADATA_PATH.exists(): raise HTTPException(503, "Model metadata unavailable")
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
