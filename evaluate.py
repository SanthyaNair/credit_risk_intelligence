import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import joblib
import yaml
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split

def plot_roc_curve(model, X_val, y_val):
    preds = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, preds)
    auc = roc_auc_score(y_val, preds)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.4f})", color="steelblue")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Credit Risk Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=150)
    plt.close()
    print("ROC curve saved to outputs/roc_curve.png")

def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[idx], color="steelblue")
    plt.yticks(range(top_n), [feature_names[i] for i in idx])
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()
    print("Feature importance saved to outputs/feature_importance.png")

def shap_analysis(model, X_val):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val[:500])
    shap.summary_plot(shap_values, X_val[:500], show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150)
    plt.close()
    print("SHAP summary saved to outputs/shap_summary.png")

def run_evaluation():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = joblib.load(cfg['paths']['model_output'])
    print("Model loaded from", cfg['paths']['model_output'])

    from src.preprocessing import preprocess_pipeline
    df = preprocess_pipeline(cfg['paths']['train_data'])
    X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
    y = df['TARGET']

    _, X_val, _, y_val = train_test_split(
        X, y,
        test_size=cfg['training']['test_size'],
        stratify=y,
        random_state=cfg['model']['random_state']
    )

    print("Running ROC curve...")
    plot_roc_curve(model, X_val, y_val)

    print("Running feature importance...")
    plot_feature_importance(model, list(X_val.columns))

    print("Running SHAP analysis...")
    shap_analysis(model, X_val)

    preds = model.predict(X_val)
    report = classification_report(y_val, preds)
    print("\nClassification Report:")
    print(report)

    with open("outputs/reports/metrics.txt", "w") as f:
        f.write(report)
    print("Report saved to outputs/reports/metrics.txt")

if __name__ == "__main__":
    run_evaluation()
