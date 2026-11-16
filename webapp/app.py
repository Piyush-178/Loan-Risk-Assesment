#!/usr/bin/env python3
"""
Loan Risk Assessment — web backend (Flask).

Serves the single-page frontend and a small JSON API:

  GET  /                -> frontend (static/index.html)
  GET  /api/health      -> {"status": "ok", "model_loaded": true}
  GET  /api/model-info  -> model name, ROC-AUC, training size, feature list
  POST /api/predict     -> {"risk_probability": 0.074, "risk_percent": 7.4,
                            "category": "LOW", "recommendation": "...",
                            "factors": [{"type": "positive", "text": "..."}]}

Every input is validated (presence, type, range, allowed values). Bad input
returns HTTP 400 with a clear JSON error — never a stack trace.

Run:  python3 app.py     then open http://127.0.0.1:5000
"""
import json
import os
import site
import sys

# Filter out incompatible roaming user site-packages (e.g. NumPy 2.x conflicting with Anaconda C-extensions)
_user_site = site.getusersitepackages()
_user_sites = [_user_site] if isinstance(_user_site, str) else (_user_site or [])
sys.path = [p for p in sys.path if not any(p.lower().startswith(u.lower()) for u in _user_sites)]

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODEL_PATH = os.path.join(ROOT, "models", "loan_risk_model.pkl")
METRICS_PATH = os.path.join(ROOT, "outputs", "metrics.json")
STATIC_DIR = os.path.join(HERE, "static")

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="/static")

# ---------------------------------------------------------------- model ----
try:
    MODEL = joblib.load(MODEL_PATH)
    MODEL_ERROR = None
except Exception as exc:  # keep the server up so /api/health can report it
    MODEL = None
    MODEL_ERROR = str(exc)
    print(f"[ERROR] Failed to load model from {MODEL_PATH}: {exc}", file=sys.stderr)

try:
    with open(METRICS_PATH) as f:
        METRICS = json.load(f)
except Exception:
    METRICS = {}

FEATURES = [
    "age", "annual_income", "employment_years", "employment_type",
    "credit_score", "existing_loans", "loan_amount", "loan_tenure_months",
    "loan_purpose", "debt_to_income_ratio",
]
EMPLOYMENT_TYPES = ["Salaried", "Self-Employed", "Business"]
LOAN_PURPOSES = ["Home", "Education", "Personal", "Business", "Vehicle", "Medical"]

# field -> (kind, min, max)   kind: "int" | "float"
NUMERIC_RULES = {
    "age": ("int", 18, 100),
    "annual_income": ("float", 10_000, 10_000_000),
    "employment_years": ("float", 0, 50),
    "credit_score": ("float", 300, 900),
    "existing_loans": ("int", 0, 20),
    "loan_amount": ("float", 1_000, 10_000_000),
    "loan_tenure_months": ("int", 6, 120),
    "debt_to_income_ratio": ("float", 0.0, 1.0),
}


def validate(payload):
    """Return (clean_dict, error_message). Exactly one of them is None."""
    if not isinstance(payload, dict):
        return None, "Request body must be a JSON object."
    clean = {}
    for field, (kind, lo, hi) in NUMERIC_RULES.items():
        if field not in payload or payload[field] is None or payload[field] == "":
            return None, f"Missing field: {field}"
        value = payload[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None, f"Field '{field}' must be a number."
        if kind == "int":
            value = int(value)
        if not (lo <= value <= hi):
            return None, f"Field '{field}' must be between {lo} and {hi}."
        clean[field] = value
    for field, allowed in (("employment_type", EMPLOYMENT_TYPES),
                           ("loan_purpose", LOAN_PURPOSES)):
        if field not in payload or payload[field] is None or payload[field] == "":
            return None, f"Missing field: {field}"
        if payload[field] not in allowed:
            return None, f"Field '{field}' must be one of: {', '.join(allowed)}."
        clean[field] = payload[field]
    return clean, None


def risk_factors(a):
    """Transparent, rule-based highlights shown next to the model score."""
    factors = []
    cs = a["credit_score"]
    if cs >= 750:
        factors.append(("positive", f"Strong credit score ({cs:.0f})"))
    elif cs < 580:
        factors.append(("negative", f"Low credit score ({cs:.0f})"))
    dti = a["debt_to_income_ratio"]
    if dti > 0.45:
        factors.append(("negative", f"High debt-to-income ratio ({dti:.0%})"))
    elif dti <= 0.25:
        factors.append(("positive", f"Comfortable debt-to-income ratio ({dti:.0%})"))
    lti = a["loan_amount"] / max(a["annual_income"], 1)
    if lti > 1.5:
        factors.append(("negative", f"Loan is {lti:.1f}x the annual income"))
    elif lti <= 0.75:
        factors.append(("positive", f"Loan is only {lti:.1f}x the annual income"))
    if a["existing_loans"] >= 3:
        factors.append(("negative", f"{a['existing_loans']} loans already running"))
    elif a["existing_loans"] == 0:
        factors.append(("positive", "No existing loans"))
    if a["employment_years"] >= 5:
        factors.append(("positive", f"Stable employment ({a['employment_years']:.0f} years)"))
    elif a["employment_years"] < 2:
        factors.append(("negative", "Very short employment history"))
    return [{"type": t, "text": x} for t, x in factors]


# ---------------------------------------------------------------- routes ---
@app.route("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok" if MODEL is not None else "model_error",
        "model_loaded": MODEL is not None,
        "model_error": MODEL_ERROR,
    })


@app.route("/api/model-info")
def model_info():
    best = METRICS.get("best_model", "Logistic Regression")
    return jsonify({
        "model": best,
        "roc_auc": METRICS.get(best, {}).get("roc_auc"),
        "cv_roc_auc": METRICS.get("best_model_cv_roc_auc_mean"),
        "trained_on_records": 5000,
        "features": FEATURES,
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if MODEL is None:
        return jsonify({"error": "Model is not loaded on the server."}), 503
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Request body must be valid JSON."}), 400
    clean, error = validate(payload)
    if error:
        return jsonify({"error": error}), 400
    try:
        row = pd.DataFrame([{f: clean[f] for f in FEATURES}])
        risk = float(MODEL.predict_proba(row)[0, 1])
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Prediction failed for the given input."}), 500
    if risk < 0.35:
        category, recommendation = "LOW", "APPROVE — low default risk"
    elif risk < 0.60:
        category, recommendation = ("MODERATE",
                                    "MANUAL REVIEW — verify income & documents")
    else:
        category, recommendation = ("HIGH",
                                    "REJECT — or ask for collateral / a co-applicant")
    return jsonify({
        "risk_probability": round(risk, 4),
        "risk_percent": round(risk * 100, 1),
        "category": category,
        "recommendation": recommendation,
        "factors": risk_factors(clean),
    })


@app.errorhandler(404)
def not_found(_e):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Unknown API endpoint."}), 404
    return send_from_directory(STATIC_DIR, "index.html")


@app.errorhandler(405)
def method_not_allowed(_e):
    return jsonify({"error": "Method not allowed for this endpoint."}), 405


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
