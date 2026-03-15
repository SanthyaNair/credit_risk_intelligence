import pytest
import pandas as pd
import numpy as np
from src.preprocessing import handle_missing_values, engineer_features

def test_no_nulls_after_fill():
    df = pd.DataFrame({
        'AMT_CREDIT': [100, np.nan, 300],
        'AMT_INCOME_TOTAL': [50, 60, np.nan],
        'AMT_ANNUITY': [10, np.nan, 30],
        'DAYS_EMPLOYED': [-1000, -2000, np.nan],
        'DAYS_BIRTH': [-10000, -12000, -11000]
    })
    df = handle_missing_values(df)
    assert df.isnull().sum().sum() == 0

def test_feature_engineering_creates_columns():
    df = pd.DataFrame({
        'AMT_CREDIT': [100], 'AMT_INCOME_TOTAL': [50],
        'AMT_ANNUITY': [10], 'DAYS_EMPLOYED': [-500],
        'DAYS_BIRTH': [-10000]
    })
    df = engineer_features(df)
    assert 'CREDIT_INCOME_RATIO' in df.columns
    assert 'ANNUITY_INCOME_RATIO' in df.columns