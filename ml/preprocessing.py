from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import NUMERICAL_FEATURES, CATEGORICAL_FEATURES

ENGINEERED_NUMERICAL_FEATURES = NUMERICAL_FEATURES + [
    "income_to_loan", "monthly_income_proxy", "loan_to_income",
    "employment_age_ratio", "credit_history_age_ratio", "interest_burden_proxy"
]

def build_preprocessor():
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, ENGINEERED_NUMERICAL_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES),
    ])
