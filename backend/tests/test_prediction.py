from fastapi.testclient import TestClient
from backend.app.main import app
client=TestClient(app)
VALID={"person_age":30,"person_income":60000,"person_home_ownership":"RENT","person_emp_length":5,"loan_intent":"PERSONAL","loan_grade":"B","loan_amnt":10000,"loan_int_rate":11.5,"loan_percent_income":.17,"cb_person_default_on_file":"N","cb_person_cred_hist_length":8}
def test_prediction_is_artifact_dependent():
    assert client.post("/api/v1/predict",json=VALID).status_code in (200,503)
