# src/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

def load_data(filepath: str) -> pd.DataFrame:
    return pd.read_csv(filepath)

def drop_high_missing(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    return df[df.columns[df.isnull().mean() < threshold]]

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    for col in df.select_dtypes('object').columns:
        if df[col].nunique() <= 2:
            df[col] = le.fit_transform(df[col].astype(str))
        else:
            df = pd.get_dummies(df, columns=[col], drop_first=True)
    return df

def impute_nulls(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = df.select_dtypes('number').columns
    imputer  = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    return df

def preprocess(filepath: str) -> pd.DataFrame:
    df = load_data(filepath)
    df = drop_high_missing(df)
    df = encode_categoricals(df)
    df = impute_nulls(df)
    return df