import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .config import *
from .data_loader import load_dataset, validate_dataset
from .features import engineer_features
from .preprocessing import build_preprocessor
from .evaluate import evaluate_predictions, select_best

def train():
    df = load_dataset()
    validate_dataset(df)
    X = engineer_features(df[INPUT_FEATURES])
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    preprocessor = build_preprocessor()
    Xtr = preprocessor.fit_transform(X_train)
    Xte = preprocessor.transform(X_test)

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            objective="binary:logistic", eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=4
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(Xtr, y_train)
        results[name] = evaluate_predictions(y_test, model.predict_proba(Xte)[:, 1])

    best = select_best(results)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSOR_PATH.parent.mkdir(parents=True, exist_ok=True)
    METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(models[best], MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)

    metadata = {
        "project": "Enterprise Credit Risk Engine",
        "target": TARGET,
        "best_model": best,
        "best_roc_auc": results[best]["roc_auc"],
        "models": results,
        "random_state": RANDOM_STATE,
        "training_samples": int(len(X_train)),
        "testing_samples": int(len(X_test)),
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "input_features": INPUT_FEATURES,
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata

if __name__ == "__main__":
    train()
