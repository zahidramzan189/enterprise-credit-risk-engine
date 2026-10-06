import joblib
import numpy as np
import pandas as pd
from functools import lru_cache

from .config import *
from .features import engineer_features


@lru_cache(maxsize=1)
def _artifacts():
    if not PREPROCESSOR_PATH.exists() or not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model artifacts are missing. Run `python -m ml.train`."
        )

    pre = joblib.load(PREPROCESSOR_PATH)

    path = (
        CALIBRATED_MODEL_PATH
        if CALIBRATED_MODEL_PATH.exists()
        else MODEL_PATH
    )

    model = joblib.load(path)

    return pre, model, path.name


def risk_from_probability(p):
    p = float(np.clip(p, 0, 1))

    score = int(round((1 - p) * 100))

    if p < RISK_THRESHOLDS["low_max"]:
        level = "LOW"
    elif p < RISK_THRESHOLDS["medium_max"]:
        level = "MEDIUM"
    else:
        level = "HIGH"

    return score, level


def predict_credit_risk(payload):
    missing = [f for f in INPUT_FEATURES if f not in payload]

    if missing:
        raise ValueError(f"Missing fields: {missing}")

    frame = pd.DataFrame([
        {f: payload[f] for f in INPUT_FEATURES}
    ])

    pre, model, artifact = _artifacts()

    features = engineer_features(frame)
    transformed = pre.transform(features)

    probability = float(
        model.predict_proba(transformed)[0, 1]
    )

    prediction = int(probability >= 0.5)

    score, level = risk_from_probability(probability)

    return {
        "default_probability": probability,
        "default_probability_percent": round(probability * 100, 2),
        "risk_score": score,
        "risk_level": level,
        "prediction": prediction,
        "prediction_label": (
            "DEFAULT" if prediction else "NON_DEFAULT"
        ),
        "model_type": artifact.replace(".joblib", ""),
    }