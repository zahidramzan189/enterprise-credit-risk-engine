import joblib
import numpy as np
import pandas as pd
import shap

from .config import (
    CALIBRATED_MODEL_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    INPUT_FEATURES,
)
from .features import engineer_features
from .predict import _artifacts, predict_credit_risk


def _get_base_estimator(model):
    """
    Extract the underlying estimator from CalibratedClassifierCV.

    The production prediction continues to use the calibrated model.
    SHAP explains the underlying tree model because TreeExplainer
    can explain the XGBoost estimator directly.
    """
    if hasattr(model, "calibrated_classifiers_"):
        calibrated = model.calibrated_classifiers_

        if not calibrated:
            raise ValueError("Calibrated model contains no calibrated classifiers.")

        calibrated_classifier = calibrated[0]

        if hasattr(calibrated_classifier, "estimator"):
            return calibrated_classifier.estimator

        if hasattr(calibrated_classifier, "base_estimator"):
            return calibrated_classifier.base_estimator

    if hasattr(model, "estimator"):
        return model.estimator

    return model


def _shap_values(explainer, dense):
    """
    Handle SHAP output differences between SHAP versions.
    """
    explanation = explainer(dense)

    values = np.asarray(explanation.values)

    if values.ndim == 3:
        values = values[0, :, 0]
    elif values.ndim == 2:
        values = values[0]
    elif values.ndim == 1:
        values = values
    else:
        values = values.reshape(-1)

    base_values = np.asarray(explanation.base_values).reshape(-1)

    base_value = float(base_values[0])

    return values, base_value


def explain_credit_risk(payload, background_rows=20):
    """
    Generate SHAP-based explanations for one applicant.

    Prediction remains based on the calibrated model.
    SHAP explains the underlying XGBoost estimator.
    """

    missing = [f for f in INPUT_FEATURES if f not in payload]

    if missing:
        raise ValueError(f"Missing fields: {missing}")

    # Use the cached production artifacts.
    pre, model, _ = _artifacts()

    frame = pd.DataFrame([
        {f: payload[f] for f in INPUT_FEATURES}
    ])

    engineered = engineer_features(frame)

    transformed = pre.transform(engineered)

    dense = (
        transformed.toarray()
        if hasattr(transformed, "toarray")
        else np.asarray(transformed)
    )

    names = list(pre.get_feature_names_out())

    base_model = _get_base_estimator(model)

    # TreeExplainer is considerably lighter and more appropriate
    # for the underlying XGBoost tree model than the previous
    # generic model.predict_proba wrapper.
    explainer = shap.TreeExplainer(
        base_model,
        feature_perturbation="tree_path_dependent",
    )

    values, base_value = _shap_values(explainer, dense)

    if len(values) != len(names):
        raise ValueError(
            f"SHAP feature mismatch: {len(values)} values for {len(names)} features."
        )

    ranked = sorted(
        zip(names, values),
        key=lambda item: abs(float(item[1])),
        reverse=True,
    )[:10]

    factors = [
        {
            "feature": name,
            "impact": round(float(value), 6),
            "direction": (
                "increases_default_risk"
                if float(value) > 0
                else "decreases_default_risk"
            ),
        }
        for name, value in ranked
    ]

    positive = [
        {
            "feature": name,
            "impact": round(float(value), 6),
        }
        for name, value in ranked
        if float(value) > 0
    ]

    negative = [
        {
            "feature": name,
            "impact": round(float(value), 6),
        }
        for name, value in ranked
        if float(value) < 0
    ]

    # IMPORTANT:
    # This remains the calibrated production prediction.
    prediction = predict_credit_risk(payload)

    return {
        "prediction": prediction,
        "base_value": base_value,
        "key_factors": factors,
        "positive_risk_contributors": positive,
        "negative_risk_contributors": negative,
        "human_readable": (
            "Positive SHAP values push the underlying XGBoost model "
            "toward default; negative values push it away from default. "
            "The displayed risk probability and decision are produced "
            "by the calibrated production model."
        ),
    }