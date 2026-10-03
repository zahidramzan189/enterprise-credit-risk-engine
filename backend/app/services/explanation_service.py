import logging
from ml.explain import explain_credit_risk
logger = logging.getLogger(__name__)

def explain(request):
    try: return explain_credit_risk(request.model_dump())
    except Exception:
        logger.exception("Explanation failed")
        raise
