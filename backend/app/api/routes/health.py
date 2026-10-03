from fastapi import APIRouter
from backend.app.config import ENVIRONMENT
router = APIRouter(tags=["health"])

@router.get("/health")
def health():
    return {"status":"ok","service":"enterprise-credit-risk-engine","environment":ENVIRONMENT}
