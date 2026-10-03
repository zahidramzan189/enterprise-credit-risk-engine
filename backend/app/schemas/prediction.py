from pydantic import BaseModel, ConfigDict, Field, field_validator

class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    person_age: int = Field(ge=18, le=100)
    person_income: float = Field(ge=0)
    person_home_ownership: str = Field(min_length=1, max_length=30)
    person_emp_length: float = Field(ge=0, le=100)
    loan_intent: str = Field(min_length=1, max_length=50)
    loan_grade: str = Field(min_length=1, max_length=10)
    loan_amnt: float = Field(ge=0)
    loan_int_rate: float = Field(ge=0, le=100)
    loan_percent_income: float = Field(ge=0, le=10)
    cb_person_default_on_file: str = Field(min_length=1, max_length=5)
    cb_person_cred_hist_length: float = Field(ge=0, le=100)

    @field_validator("person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file")
    @classmethod
    def normalize(cls, value):
        value = value.strip().upper()
        if not value: raise ValueError("Value cannot be empty")
        return value
