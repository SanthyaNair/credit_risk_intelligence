# 🏦 Credit Risk Intelligence

> Predicting loan default probability using XGBoost on the Home Credit dataset.

## 📊 Results
| Metric | Score |
|--------|-------|
| ROC-AUC (validation) | ~0.76 |
| Algorithm | XGBoost |
| Features engineered | 4+ custom |

## 🚀 Quick Start
```bash
git clone https://github.com/YOUR_USERNAME/credit-risk-intelligence
cd credit-risk-intelligence
pip install -r requirements.txt

# Download data (requires Kaggle API key)
kaggle competitions download -c home-credit-default-risk

# Train
python -m src.train

# Evaluate & predict
python -m src.predict
```

## 🏗️ Architecture
- **Preprocessing:** Median imputation, label encoding, 4 domain-specific ratios
- **Model:** XGBoost with early stopping + class imbalance handling via `scale_pos_weight`
- **Explainability:** SHAP TreeExplainer summary plots
- **Optimization:** Optuna-based hyperparameter search (optional)

## 📁 Project Structure
```
src/          # Core pipeline modules
notebooks/    # EDA & experimentation
tests/        # Unit tests (pytest)
outputs/      # Plots & submission CSV
```
