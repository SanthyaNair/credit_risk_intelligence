import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import yaml
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.preprocessing import preprocess_pipeline

def load_config(path="config.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def train_model():
    cfg = load_config()
    
    # Load & preprocess
    df = preprocess_pipeline(cfg['paths']['train_data'])
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg['training']['test_size'],
        stratify=y,
        random_state=cfg['model']['random_state']
    )
    
    # XGBoost model
    model = xgb.XGBClassifier(
        n_estimators=cfg['model']['n_estimators'],
        max_depth=cfg['model']['max_depth'],
        learning_rate=cfg['model']['learning_rate'],
        subsample=cfg['model']['subsample'],
        colsample_bytree=cfg['model']['colsample_bytree'],
        scale_pos_weight=cfg['model']['scale_pos_weight'],
        eval_metric=cfg['model']['eval_metric'],
        early_stopping_rounds=cfg['model']['early_stopping_rounds'],
        random_state=cfg['model']['random_state'],
        use_label_encoder=False
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50
    )
    
    # Evaluate
    val_preds = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_preds)
    print(f"\n✅ Validation ROC-AUC: {auc:.4f}")
    
    # Save model
    joblib.dump(model, cfg['paths']['model_output'])
    print(f"✅ Model saved to {cfg['paths']['model_output']}")
    
    return model, X_val, y_val

if __name__ == "__main__":
    train_model()