"""
Credit Card Approval Screening — Flask web app.

Routes:
  /                 Application form (analyst / compliance officer / customer)
  /predict          POST — returns approve/decline + probability + SHAP explanation
  /dashboard        Model comparison + income-vs-approval visualizations
  /sandbox          "What-if" scenario tester for customers, doesn't save anything
  /notifications    Borderline-case queue for analysts (in-memory demo store)
  /api/metrics      Raw metrics.json for the dashboard's charts
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

try:
    import shap
    HAS_SHAP = True
except ImportError:
    # Falls back to the model's global feature importance (see explain()) when
    # shap isn't installed. Add `shap` to requirements.txt for per-prediction
    # explanations in the real deployment.
    HAS_SHAP = False

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

app = Flask(__name__)

# ---- Load trained artifacts once at startup -------------------------------
model = joblib.load(MODEL_DIR / "best_model.joblib")
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
feature_columns = joblib.load(MODEL_DIR / "feature_columns.joblib")

shap_background = None
if HAS_SHAP:
    try:
        shap_background = joblib.load(MODEL_DIR / "shap_background.joblib")
    except Exception:
        shap_background = None

with open(MODEL_DIR / "metrics.json") as f:
    METRICS = json.load(f)

CATEGORICAL = METRICS["categorical_columns"]
NUMERIC = METRICS["numeric_columns"]

CATEGORY_OPTIONS = {
    "income_type": ["Working", "Commercial associate", "Pensioner", "State servant", "Student"],
    "education": ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary"],
    "family_status": ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
    "housing_type": ["House / apartment", "With parents", "Rented apartment", "Municipal apartment"],
    "occupation_type": ["Laborers", "Core staff", "Managers", "Sales staff", "Drivers", "Accountants", "IT staff", "Medicine staff"],
}

# Borderline probability band that triggers an analyst review flag.
BORDERLINE_LOW, BORDERLINE_HIGH = 0.40, 0.60

# In-memory notification queue for the demo. Swap for a DB table in production.
NOTIFICATIONS = []

_explainer = None
if HAS_SHAP:
    try:
        _explainer = shap.Explainer(model, shap_background)
    except Exception:
        _explainer = None


def build_feature_row(payload: dict) -> pd.DataFrame:
    """Turn a raw form/JSON payload into a one-hot-encoded row aligned to feature_columns."""
    row = {col: 0 for col in feature_columns}
    for col in NUMERIC:
        row[col] = float(payload.get(col, 0))
    for cat in CATEGORICAL:
        choice = payload.get(cat)
        one_hot_col = f"{cat}_{choice}"
        if one_hot_col in row:
            row[one_hot_col] = 1
    return pd.DataFrame([row], columns=feature_columns)


def explain(row_scaled: np.ndarray, top_n: int = 5):
    """Return top contributing features for this prediction (SHAP if available, else model importance)."""
    if _explainer is not None:
        try:
            sv = _explainer(row_scaled)
            values = sv.values[0]
            pairs = sorted(zip(feature_columns, values), key=lambda kv: abs(kv[1]), reverse=True)[:top_n]
            return [{"feature": f, "impact": round(float(v), 4)} for f, v in pairs]
        except Exception:
            pass
    # Fallback: global feature importance from training
    imp = METRICS.get("feature_importance", {})
    pairs = list(imp.items())[:top_n]
    return [{"feature": f, "impact": round(float(v), 4)} for f, v in pairs]


def score_application(payload: dict):
    row = build_feature_row(payload)
    row_scaled = scaler.transform(row)
    proba_high_risk = float(model.predict_proba(row_scaled)[0, 1])
    decision = "Decline" if proba_high_risk >= 0.5 else "Approve"
    borderline = BORDERLINE_LOW <= proba_high_risk <= BORDERLINE_HIGH
    result = {
        "decision": decision,
        "approve_probability": round(1 - proba_high_risk, 4),
        "high_risk_probability": round(proba_high_risk, 4),
        "borderline": borderline,
        "explanation": explain(row_scaled),
    }
    return result


@app.route("/")
def index():
    return render_template("index.html", categories=CATEGORY_OPTIONS, numeric=NUMERIC,
                            best_model=METRICS["best_model_name"])


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    result = score_application(payload)

    if result["borderline"]:
        NOTIFICATIONS.append({
            "applicant_summary": {k: payload.get(k) for k in ["income_type", "annual_income", "employment_years"]},
            "high_risk_probability": result["high_risk_probability"],
            "note": "Borderline case — probability lands in the manual-review band.",
        })

    return jsonify(result)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", metrics=METRICS)


@app.route("/api/metrics")
def api_metrics():
    df = pd.read_csv(BASE_DIR / "data" / "applicants.csv")
    bins = pd.cut(df["annual_income"], bins=[0, 25000, 40000, 55000, 70000, 90000, np.inf])
    income_vs_target = (
        df.assign(low_risk=(df["payment_status"] < 2).astype(int))
        .groupby(bins, observed=True)["low_risk"].mean()
        .reset_index()
    )
    income_vs_approval = {
        "labels": [str(b) for b in income_vs_target["annual_income"]],
        "approval_rate": (income_vs_target["low_risk"] * 100).round(1).tolist(),
    }
    return jsonify({
        "model_comparison": METRICS["results"],
        "best_model": METRICS["best_model_name"],
        "feature_importance": dict(list(METRICS["feature_importance"].items())[:10]),
        "income_vs_approval": income_vs_approval,
    })


@app.route("/sandbox")
def sandbox():
    return render_template("sandbox.html", categories=CATEGORY_OPTIONS, numeric=NUMERIC)


@app.route("/notifications")
def notifications():
    return render_template("notifications.html", notifications=list(reversed(NOTIFICATIONS)))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=True)
