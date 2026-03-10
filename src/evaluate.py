import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

def plot_roc_curve(model, X_val, y_val):
    preds = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, preds)
    auc = roc_auc_score(y_val, preds)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"XGBoost (AUC = {auc:.4f})", color="steelblue")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Credit Risk Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig("outputs/roc_curve.png", dpi=150)
    plt.show()

def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[idx], color="steelblue")
    plt.yticks(range(top_n), [feature_names[i] for i in idx])
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.show()

def shap_analysis(model, X_val):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val[:500])
    shap.summary_plot(shap_values, X_val[:500], show=False)
    plt.tight_layout()
    plt.savefig("outputs/shap_summary.png", dpi=150)
    plt.show()