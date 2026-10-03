# Enterprise Credit Risk Engine

Production-style credit default assessment platform using classical ML, XGBoost, probability calibration, SHAP, FastAPI, Pydantic validation, vanilla HTML/CSS/JavaScript, tests, Docker and CI.

## Dataset
Place the real dataset at `data/raw/credit_risk_dataset.csv`.

Exact columns:
`person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_status, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length`

Target: `loan_status` (`0=non-default`, `1=default`).

No fabricated dataset rows or metrics are included.

## Install — Windows PowerShell
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train
```powershell
python -m ml.train
python -m ml.calibrate
```

The training command measures Logistic Regression, Random Forest and XGBoost using actual data and writes `models/metadata.json`. The calibrated artifact is created by the second command.

## Run
```powershell
python -m uvicorn backend.app.main:app --reload
```
Open `http://127.0.0.1:8000/`.

API docs: `http://127.0.0.1:8000/docs`

## Endpoints
- GET `/health`
- POST `/api/v1/predict`
- POST `/api/v1/explain`
- GET `/api/v1/model-info`

## Canonical request
```json
{
  "person_age": 30,
  "person_income": 60000,
  "person_home_ownership": "RENT",
  "person_emp_length": 5,
  "loan_intent": "PERSONAL",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.17,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 8
}
```

## Tests
```powershell
pytest -q
```

## Docker
```powershell
docker compose up --build
```
Dashboard: `http://localhost:8080/`
API docs: `http://localhost:8000/docs`

## Architecture
Real CSV → validation → deterministic feature engineering → sklearn preprocessing → Logistic Regression / Random Forest / XGBoost → ROC-AUC model selection → calibrated selected model → FastAPI → SHAP → enterprise dashboard.

## Limitations
This is a portfolio/engineering system, not a regulated lending decisioning product. Real deployment needs security, governance, fairness analysis, drift monitoring, auditability, model approval and regulatory review.
