import logging
import logging.config
import yaml
import pandas as pd
import joblib

with open('config/logging.yaml') as f:
    logging.config.dictConfig(yaml.safe_load(f))
with open('config/config.yaml') as f:
    config = yaml.safe_load(f)

logger = logging.getLogger(__name__)

from src.preprocessing import preprocess_pipeline
from src.train import train_model
from src.evaluate import evaluate_model
from src.predict import predict

def run_pipeline():
    # 1. Load + Preprocess
    logger.info("=== STAGE 1 & 2: Loading and Preprocessing ===")
    processed = preprocess_pipeline(config['data']['raw_path'])
    processed.to_csv(config['data']['processed_path'], index=False)
    logger.info(f"Processed shape: {processed.shape}")

    # 2. Train
    logger.info("=== STAGE 3: Training XGBoost ===")
    X = processed.drop('TARGET', axis=1)
    y = processed['TARGET']
    model = train_model(X, y)
    joblib.dump(model, config['model']['save_path'])
    logger.info("Model saved to models/xgboost_model.pkl")

    # 3. Evaluate
    logger.info("=== STAGE 4: Evaluation + SHAP ===")
    evaluate_model(model, X, y, config)

    # 4. Predict sample
    logger.info("=== STAGE 5: Sample Predictions ===")
    preds = predict(model, X.head(5))
    logger.info(f"Sample predictions: {preds}")

    logger.info("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    run_pipeline()
