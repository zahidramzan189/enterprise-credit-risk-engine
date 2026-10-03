from fastapi.testclient import TestClient
from backend.app.main import app
client=TestClient(app)
VALID={"person_age":30,"person_income":60000,"person_home_ownership":"RENT","person_emp_length":5,"loan_intent":"PERSONAL","loan_grade":"B","loan_amnt":10000,"loan_int_rate":11.5,"loan_percent_income":.17,"cb_person_default_on_file":"N","cb_person_cred_hist_length":8}
def test_unknown_field_rejected(): assert client.post("/api/v1/predict",json={**VALID,"bad":1}).status_code==422
def test_negative_income_rejected(): assert client.post("/api/v1/predict",json={**VALID,"person_income":-1}).status_code==422
def test_missing_field_rejected():
    x=dict(VALID);x.pop("loan_amnt");assert client.post("/api/v1/predict",json=x).status_code==422
def test_malformed_json():
    r=client.post("/api/v1/predict",data='{"person_age":30,',headers={"Content-Type":"application/json"});assert r.status_code==422
