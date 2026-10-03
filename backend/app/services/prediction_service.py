import logging
from ml.predict import predict_credit_risk
logger = logging.getLogger(__name__)

def predict(request):
    try: return predict_credit_risk(request.model_dump())
    except Exception:
        logger.exception("Prediction failed")
        raise
