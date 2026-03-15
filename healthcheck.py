# healthcheck.py - Windows Compatible
import sys
import os
import joblib
import pandas as pd
from pathlib import Path

# ── Windows-safe path handling ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.resolve()

TRAIN_CSV  = BASE_DIR / "data" / "raw" / "application_train.csv"
TEST_CSV   = BASE_DIR / "data" / "raw" / "application_test.csv"
CONFIG     = BASE_DIR / "config.yaml"
MODEL_FILE = BASE_DIR / "models" / "xgboost_model.pkl"
SUBMIT_CSV = BASE_DIR / "outputs" / "submission.csv"

# ── Check tracker ────────────────────────────────────────────────────────────
CHECKS = []

def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    icon   = "[OK]" if condition else "[!!]"
    CHECKS.append((name, status))
    detail_str = f"  --> {detail}" if detail else ""
    print(f"  {icon}  {name}{detail_str}")

# ── Section printer ──────────────────────────────────────────────────────────
def section(title):
    print()
    print("=" * 50)
    print(f"  {title}")
    print("=" * 50)

# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 50)
print("   CREDIT RISK INTELLIGENCE - HEALTH CHECK")
print("   Running on Windows")
print("=" * 50)

# ── 1. FILE STRUCTURE ────────────────────────────────────────────────────────
section("1. FILE STRUCTURE")

check("Train CSV exists",      TRAIN_CSV.exists(),  str(TRAIN_CSV))
check("Test CSV exists",       TEST_CSV.exists(),   str(TEST_CSV))
check("Config (yaml) exists",  CONFIG.exists(),     str(CONFIG))
check("Models folder exists",  (BASE_DIR / "models").exists())
check("Outputs folder exists", (BASE_DIR / "outputs").exists())
check("src folder exists",     (BASE_DIR / "src").exists())

# ── 2. PYTHON PACKAGES ───────────────────────────────────────────────────────
section("2. PYTHON PACKAGES")

packages = {
    "xgboost":      "xgboost",
    "scikit-learn": "sklearn",
    "pandas":       "pandas",
    "numpy":        "numpy",
    "shap":         "shap",
    "joblib":       "joblib",
    "pyyaml":       "yaml",
}

for display_name, import_name in packages.items():
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "unknown")
        check(f"{display_name} installed", True, f"version {ver}")
    except ImportError:
        check(f"{display_name} installed", False, "Run: pip install " + display_name)

# ── 3. DATA VALIDATION ───────────────────────────────────────────────────────
section("3. DATA VALIDATION")

if TRAIN_CSV.exists() and TEST_CSV.exists():
    try:
        train_df = pd.read_csv(TRAIN_CSV)
        test_df  = pd.read_csv(TEST_CSV)

        check("Train CSV loads without error",  True)
        check("Test CSV loads without error",   True)
        check("Train has correct rows",         len(train_df) == 307511,
              f"Found {len(train_df)} rows (expected 307511)")
        check("Test has correct rows",          len(test_df) == 48744,
              f"Found {len(test_df)} rows (expected 48744)")
        check("TARGET column exists in train",  "TARGET" in train_df.columns)
        check("SK_ID_CURR exists in train",     "SK_ID_CURR" in train_df.columns)
        check("SK_ID_CURR exists in test",      "SK_ID_CURR" in test_df.columns)

        null_pct = round(train_df.isnull().mean().mean() * 100, 2)
        check("Null % is within expected range (< 50%)",
              null_pct < 50, f"Current null %: {null_pct}%")

        class_counts = train_df["TARGET"].value_counts(normalize=True)
        imbalance_ok = class_counts.get(0, 0) > 0.8
        check("Class imbalance is expected (90/10 split)",
              imbalance_ok,
              f"Class 0: {class_counts.get(0,0):.1%}  Class 1: {class_counts.get(1,0):.1%}")

    except Exception as e:
        check("Data files readable", False, str(e))
else:
    check("Data files readable", False,
          "CSV files missing — download from Kaggle first")

# ── 4. PREPROCESSING PIPELINE ────────────────────────────────────────────────
section("4. PREPROCESSING PIPELINE")

try:
    # Add project root to path so src imports work
    sys.path.insert(0, str(BASE_DIR))
    from src.preprocessing import preprocess_pipeline

    df_proc = preprocess_pipeline(str(TRAIN_CSV))

    check("preprocess_pipeline() runs",         df_proc is not None)
    check("No nulls after preprocessing",       df_proc.isnull().sum().sum() == 0,
          f"Nulls found: {df_proc.isnull().sum().sum()}")
    check("CREDIT_INCOME_RATIO column exists",  "CREDIT_INCOME_RATIO" in df_proc.columns)
    check("ANNUITY_INCOME_RATIO column exists", "ANNUITY_INCOME_RATIO" in df_proc.columns)
    check("CREDIT_TERM column exists",          "CREDIT_TERM" in df_proc.columns)
    check("DAYS_EMPLOYED_RATIO column exists",  "DAYS_EMPLOYED_RATIO" in df_proc.columns)
    check("Row count unchanged after preprocess",
          len(df_proc) == 307511, f"Got {len(df_proc)} rows")

except Exception as e:
    check("Preprocessing pipeline runs", False, str(e))

# ── 5. MODEL FILE ────────────────────────────────────────────────────────────
section("5. MODEL FILE")

if MODEL_FILE.exists():
    try:
        model       = joblib.load(MODEL_FILE)
        model_size  = MODEL_FILE.stat().st_size / (1024 * 1024)

        check("Model file exists",             True, str(MODEL_FILE))
        check("Model loads without error",     True)
        check("Model size > 1 MB",             model_size > 1,
              f"Size: {model_size:.2f} MB")
        check("Model has feature_importances", hasattr(model, "feature_importances_"))
        check("Feature count > 100",           model.n_features_in_ > 100,
              f"Features: {model.n_features_in_}")
        check("best_iteration is set",         hasattr(model, "best_iteration"),
              f"Best iteration: {getattr(model, 'best_iteration', 'N/A')}")

    except Exception as e:
        check("Model loads without error", False, str(e))
else:
    check("Model file exists", False,
          "Run:  python -m src.train")

# ── 6. PREDICTIONS / SUBMISSION ──────────────────────────────────────────────
section("6. PREDICTIONS / SUBMISSION FILE")

if SUBMIT_CSV.exists():
    try:
        sub = pd.read_csv(SUBMIT_CSV)

        check("Submission CSV exists",          True, str(SUBMIT_CSV))
        check("Submission loads without error", True)
        check("Has SK_ID_CURR column",          "SK_ID_CURR" in sub.columns)
        check("Has TARGET column",              "TARGET" in sub.columns)
        check("Correct number of rows",         len(sub) == 48744,
              f"Found {len(sub)} rows (expected 48744)")
        check("No nulls in submission",         sub.isnull().sum().sum() == 0)
        check("Scores are valid probabilities", sub["TARGET"].between(0, 1).all(),
              f"Min: {sub['TARGET'].min():.4f}  Max: {sub['TARGET'].max():.4f}")
        check("Mean score is reasonable (0.05-0.20)",
              sub["TARGET"].mean().between(0.05, 0.20),
              f"Mean score: {sub['TARGET'].mean():.4f}")

    except Exception as e:
        check("Submission file readable", False, str(e))
else:
    check("Submission CSV exists", False,
          "Run:  python -m src.predict")

# ── 7. OUTPUT FILES ──────────────────────────────────────────────────────────
section("7. OUTPUT PLOTS")

plots = [
    "outputs/roc_curve.png",
    "outputs/feature_importance.png",
    "outputs/shap_summary.png",
]
for plot in plots:
    p = BASE_DIR / plot
    check(f"{Path(plot).name} exists", p.exists(), str(p))

# ════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 50)
print("   FINAL SUMMARY")
print("=" * 50)

passed = [c for c in CHECKS if c[1] == "PASS"]
failed = [c for c in CHECKS if c[1] == "FAIL"]
total  = len(CHECKS)

print(f"  Passed : {len(passed)}/{total}")
print(f"  Failed : {len(failed)}/{total}")

if failed:
    print()
    print("  Fix these issues:")
    for i, (name, _) in enumerate(failed, 1):
        print(f"    {i}. {name}")
    print()
    print("  STATUS: Project needs attention before use.")
    sys.exit(1)
else:
    print()
    print("  STATUS: Project is healthy and ready!")
    print("  You can now push to GitHub safely.")
    sys.exit(0)