import joblib
import numpy as np
import pandas as pd
import shap
from .config import *
from .features import engineer_features
from .predict import predict_credit_risk

def explain_credit_risk(payload, background_rows=20):
    pre = joblib.load(PREPROCESSOR_PATH)
    path = CALIBRATED_MODEL_PATH if CALIBRATED_MODEL_PATH.exists() else MODEL_PATH
    model = joblib.load(path)

    frame = pd.DataFrame([{f: payload[f] for f in INPUT_FEATURES}])
    x = pre.transform(engineer_features(frame))
    dense = x.toarray() if hasattr(x, "toarray") else np.asarray(x)
    background = np.repeat(dense, background_rows, axis=0)
    names = list(pre.get_feature_names_out())

    def fn(values):
        return model.predict_proba(values)[:, 1]

    explainer = shap.Explainer(fn, background, feature_names=names)
    exp = explainer(dense)
    values = np.asarray(exp.values[0])
    if values.ndim > 1:
        values = values[:, 0]

    ranked = sorted(zip(names, values), key=lambda x: abs(float(x[1])), reverse=True)[:10]
    factors = [
        {"feature": n, "impact": round(float(v), 6),
         "direction": "increases_default_risk" if v > 0 else "decreases_default_risk"}
        for n, v in ranked
    ]
    positive = [{"feature": n, "impact": round(float(v), 6)} for n, v in ranked if v > 0]
    negative = [{"feature": n, "impact": round(float(v), 6)} for n, v in ranked if v < 0]

    return {
        "prediction": predict_credit_risk(payload),
        "base_value": float(np.asarray(exp.base_values).reshape(-1)[0]),
        "key_factors": factors,
        "positive_risk_contributors": positive,
        "negative_risk_contributors": negative,
        "human_readable": "Positive SHAP values push the calibrated model toward default; negative values push it away from default.",
    }
