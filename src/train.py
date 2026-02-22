# src/train.py
import mlflow
import mlflow.sklearn
import joblib
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from src.preprocess import preprocess
from src.evaluate import evaluate_model

def train(data_path: str = 'data/application_train.csv'):
    df = preprocess(data_path)
    X  = df.drop(['TARGET', 'SK_ID_CURR'], axis=1)
    y  = df['TARGET']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    with mlflow.start_run(run_name="xgb_champion"):
        params = {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': 11,
            'eval_metric': 'auc',
            'random_state': 42
        }
        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, 'models/xgb_best.pkl')
        print("Model saved to models/xgb_best.pkl")
        print(f"Metrics: {metrics}")

if __name__ == "__main__":
    train()