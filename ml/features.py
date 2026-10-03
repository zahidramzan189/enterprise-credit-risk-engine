import numpy as np
import pandas as pd

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["income_to_loan"] = out["person_income"] / out["loan_amnt"].replace(0, np.nan)
    out["monthly_income_proxy"] = out["person_income"] / 12.0
    out["loan_to_income"] = out["loan_amnt"] / out["person_income"].replace(0, np.nan)
    out["employment_age_ratio"] = out["person_emp_length"] / out["person_age"].replace(0, np.nan)
    out["credit_history_age_ratio"] = out["cb_person_cred_hist_length"] / out["person_age"].replace(0, np.nan)
    out["interest_burden_proxy"] = (
        out["loan_amnt"] * out["loan_int_rate"] / 100.0
    ) / out["person_income"].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)
