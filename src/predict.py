import pandas as pd
import joblib
import yaml
from src.preprocessing import preprocess_pipeline

def predict(test_path=None):
    cfg = yaml.safe_load(open("config.yaml"))
    model = joblib.load(cfg['paths']['model_output'])
    
    test_path = test_path or cfg['paths']['test_data']
    df_test = preprocess_pipeline(test_path)
    
    sk_ids = df_test['SK_ID_CURR']
    X_test = df_test.drop(columns=['SK_ID_CURR'], errors='ignore')
    
    # Align columns with training features
    train_df = pd.read_csv(cfg['paths']['processed_train'])
    X_test = X_test.reindex(columns=train_df.columns, fill_value=0)
    
    probs = model.predict_proba(X_test)[:, 1]
    
    submission = pd.DataFrame({'SK_ID_CURR': sk_ids, 'TARGET': probs})
    submission.to_csv("outputs/submission.csv", index=False)
    print("✅ Predictions saved to outputs/submission.csv")
    return submission

if __name__ == "__main__":
    predict()