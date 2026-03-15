import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def load_raw_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found: {filepath}")
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Loaded {df.shape[0]} rows and {df.shape[1]} columns")
    return df

def load_processed_data(filepath: str) -> pd.DataFrame:
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Processed file not found: {filepath}")
    logger.info(f"Loading processed data from {filepath}")
    return pd.read_csv(filepath)
