from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/raw/credit_risk_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "model/credit_risk_model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor/preprocessor.joblib"
CALIBRATED_MODEL_PATH = MODEL_DIR / "calibration/calibrated_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))
TEST_SIZE = 0.20
CALIBRATION_CV = 5
TARGET = "loan_status"

INPUT_FEATURES = [
    "person_age", "person_income", "person_home_ownership",
    "person_emp_length", "loan_intent", "loan_grade", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_default_on_file",
    "cb_person_cred_hist_length"
]
NUMERICAL_FEATURES = [
    "person_age", "person_income", "person_emp_length", "loan_amnt",
    "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"
]
CATEGORICAL_FEATURES = [
    "person_home_ownership", "loan_intent", "loan_grade",
    "cb_person_default_on_file"
]
REQUIRED_COLUMNS = INPUT_FEATURES + [TARGET]
RISK_THRESHOLDS = {"low_max": 0.10, "medium_max": 0.25}
