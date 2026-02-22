# src/validate_metrics.py
import argparse, joblib, pandas as pd
from src.preprocess import preprocess
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def validate(min_auc: float = 0.80):
    df     = preprocess('data/application_train.csv')
    X      = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y      = df['TARGET']
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    model  = joblib.load('models/xgb_best.pkl')
    y_prob = model.predict_proba(X_test)[:, 1]
    auc    = roc_auc_score(y_test, y_prob)

    print(f"Validation ROC-AUC: {auc:.4f} | Threshold: {min_auc}")
    if auc < min_auc:
        raise ValueError(f"FAIL — AUC {auc:.4f} is below minimum {min_auc}")
    print("PASS — Model meets quality threshold.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-auc', type=float, default=0.80)
    args = parser.parse_args()
    validate(args.min_auc)