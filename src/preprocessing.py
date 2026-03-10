import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Fill numeric with median, categorical with mode
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(include=['object']).columns
    
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())
    df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col].astype(str))
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Credit utilization ratio
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['CREDIT_TERM'] = df['AMT_ANNUITY'] / (df['AMT_CREDIT'] + 1)
    df['DAYS_EMPLOYED_RATIO'] = df['DAYS_EMPLOYED'] / (df['DAYS_BIRTH'] + 1)
    return df

def preprocess_pipeline(path: str) -> pd.DataFrame:
    df = load_data(path)
    df = handle_missing_values(df)
    df = engineer_features(df)
    df = encode_categoricals(df)
    return df