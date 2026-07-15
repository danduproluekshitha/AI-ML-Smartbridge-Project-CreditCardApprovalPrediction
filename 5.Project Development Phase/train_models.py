"""
Trains and compares four classifiers for credit card approval screening:
Logistic Regression, Random Forest, XGBoost, Decision Tree.

Pipeline:
  1. Load applicant data (data/applicants.csv)
  2. Feature engineering:
       - collapse multi-class payment_status (0-5) into a binary risk label
         (0-1 = low risk / approve, 2-5 = high risk / decline) -> target
       - one-hot encode categorical fields
       - scale numeric fields with StandardScaler
  3. Hyperparameter tuning with GridSearchCV (5-fold, ROC-AUC scoring)
  4. Evaluate all four models, pick the best by ROC-AUC on the held-out test set
  5. Save: best model, scaler, feature column order, SHAP background sample,
     and a metrics.json used by the Flask dashboard.

Run: python train_models.py
"""

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    # Falls back to GradientBoosting so the pipeline still runs end-to-end
    # in environments where xgboost hasn't been installed yet. Add
    # `xgboost` to requirements.txt / pip install it for the real thing.
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    HAS_XGB = False

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "applicants.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)

CATEGORICAL = ["income_type", "education", "family_status", "housing_type", "occupation_type"]
NUMERIC = [
    "age_years", "annual_income", "employment_years", "num_children", "family_size",
    "owns_car", "owns_realty", "credit_history_months", "existing_loan_balance",
    "num_credit_lines",
]


def engineer_features(df: pd.DataFrame):
    """Binary risk label + one-hot encoding + column alignment."""
    df = df.copy()
    # Multi-class payment_status (0-5) -> binary: 0 = low risk (approve), 1 = high risk (decline)
    df["target_high_risk"] = (df["payment_status"] >= 2).astype(int)

    X_cat = pd.get_dummies(df[CATEGORICAL], prefix=CATEGORICAL)
    X_num = df[NUMERIC]
    X = pd.concat([X_num, X_cat], axis=1)
    y = df["target_high_risk"]
    return X, y


def build_grids():
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"], "solver": ["lbfgs"]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=42),
            {"max_depth": [3, 5, 8, 12], "min_samples_leaf": [1, 5, 10]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
            {"n_estimators": [100, 200], "max_depth": [6, 10, None], "min_samples_leaf": [1, 5]},
        ),
        "XGBoost": (
            XGBClassifier(eval_metric="logloss", random_state=42)
            if HAS_XGB else XGBClassifier(random_state=42),
            {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.05, 0.1]},
        ),
    }


def main():
    print(f"Loading data from {DATA_PATH} ...")
    df = pd.read_csv(DATA_PATH)
    X, y = engineer_features(df)
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    grids = build_grids()
    results = {}
    fitted_models = {}

    for name, (estimator, param_grid) in grids.items():
        print(f"\nTuning {name} ...")
        t0 = time.time()
        search = GridSearchCV(estimator, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
        search.fit(X_train_scaled, y_train)
        best = search.best_estimator_
        preds = best.predict(X_test_scaled)
        proba = best.predict_proba(X_test_scaled)[:, 1]

        metrics = {
            "best_params": search.best_params_,
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds), 4),
            "recall": round(recall_score(y_test, preds), 4),
            "f1": round(f1_score(y_test, preds), 4),
            "roc_auc": round(roc_auc_score(y_test, proba), 4),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "train_seconds": round(time.time() - t0, 1),
        }
        results[name] = metrics
        fitted_models[name] = best
        print(f"  {name}: ROC-AUC={metrics['roc_auc']} | Acc={metrics['accuracy']} | best_params={metrics['best_params']}")

    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    best_model = fitted_models[best_name]
    print(f"\nBest model: {best_name} (ROC-AUC={results[best_name]['roc_auc']})")

    # Feature importance for the best model (used by the dashboard / explainability panel)
    if hasattr(best_model, "feature_importances_"):
        importances = dict(zip(feature_columns, best_model.feature_importances_.tolist()))
    elif hasattr(best_model, "coef_"):
        importances = dict(zip(feature_columns, np.abs(best_model.coef_[0]).tolist()))
    else:
        importances = {}
    importances = dict(sorted(importances.items(), key=lambda kv: kv[1], reverse=True))

    # Persist everything the Flask app needs
    joblib.dump(best_model, MODEL_DIR / "best_model.joblib")
    joblib.dump(scaler, MODEL_DIR / "scaler.joblib")
    joblib.dump(feature_columns, MODEL_DIR / "feature_columns.joblib")
    # Small background sample for SHAP explainability at inference time
    joblib.dump(X_train_scaled[:200], MODEL_DIR / "shap_background.joblib")

    metadata = {
        "best_model_name": best_name,
        "used_real_xgboost": HAS_XGB,
        "results": results,
        "feature_importance": importances,
        "categorical_columns": CATEGORICAL,
        "numeric_columns": NUMERIC,
        "n_rows_trained_on": len(df),
    }
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nArtifacts saved to {MODEL_DIR}/")
    if not HAS_XGB:
        print("NOTE: xgboost was not installed in this environment — GradientBoostingClassifier "
              "was used as a stand-in. `pip install xgboost` and re-run for the real XGBoost model.")


if __name__ == "__main__":
    main()
