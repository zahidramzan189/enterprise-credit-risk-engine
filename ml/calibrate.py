import json
import joblib
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from .config import *
from .data_loader import load_dataset, validate_dataset
from .features import engineer_features

def calibrate():
    df = load_dataset()
    validate_dataset(df)
    X = engineer_features(df[INPUT_FEATURES])
    y = df[TARGET].astype(int)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(MODEL_PATH)
    Xt = preprocessor.transform(X_train)

    calibrated = CalibratedClassifierCV(
        estimator=model, method="sigmoid", cv=CALIBRATION_CV
    )
    calibrated.fit(Xt, y_train)
    CALIBRATED_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(calibrated, CALIBRATED_MODEL_PATH)

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metadata["calibration"] = {"method": "sigmoid", "cv": CALIBRATION_CV}
    METADATA_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

if __name__ == "__main__":
    calibrate()
