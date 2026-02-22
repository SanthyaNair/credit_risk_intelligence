# app/schemas.py
from pydantic import BaseModel

class LoanApplication(BaseModel):
    AMT_CREDIT: float
    AMT_INCOME_TOTAL: float
    AMT_ANNUITY: float
    DAYS_EMPLOYED: int
    DAYS_BIRTH: int
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    CNT_FAM_MEMBERS: float

class PredictionResponse(BaseModel):
    default_probability: float
    decision: str
    risk_score: int
    top_risk_factors: dict