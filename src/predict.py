import pandas as pd
import joblib
import yaml
import os
from src.preprocessing import preprocess_pipeline

def predict(test_path=None):
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    model = joblib.load(cfg["paths"]["model_output"])
    print("Model loaded from", cfg["paths"]["model_output"])

    test_path = test_path or cfg["paths"]["test_data"]

    if not os.path.exists(test_path):
        print("Test file not found:", test_path)
        print("Running predictions on training data instead...")
        test_path = cfg["paths"]["train_data"]

    print("Running predictions on:", test_path)
    df_test = preprocess_pipeline(test_path)

    sk_ids = df_test["SK_ID_CURR"] if "SK_ID_CURR" in df_test.columns else None
    X_test = df_test.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")

    if os.path.exists(cfg["paths"]["processed_train"]):
        train_df = pd.read_csv(cfg["paths"]["processed_train"])
        train_cols = [c for c in train_df.columns if c not in ["TARGET", "SK_ID_CURR"]]
        X_test = X_test.reindex(columns=train_cols, fill_value=0)
        print("Aligned to", len(train_cols), "training features")

    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    if sk_ids is not None:
        submission = pd.DataFrame({"SK_ID_CURR": sk_ids, "TARGET": probs})
    else:
        submission = pd.DataFrame({"TARGET": probs})

    submission.to_csv("outputs/submission.csv", index=False)
    print("Predictions saved to outputs/submission.csv")
    print("Total predictions:", len(probs))
    print("Default rate:", round(preds.mean(), 4))
    return submission

if __name__ == "__main__":
    predict()
