#!/usr/bin/env python3
"""
Loan Risk Assessment — training pipeline.

Trains and compares Logistic Regression, Random Forest and Gradient Boosting
on the loan dataset, evaluates them on a held-out test set, saves the best
model (by ROC-AUC) plus metrics and charts.

Run:  python3 train.py
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "loan_data.csv")
os.makedirs(os.path.join(BASE, "outputs"), exist_ok=True)
os.makedirs(os.path.join(BASE, "models"), exist_ok=True)

df = pd.read_csv(DATA)
TARGET = "default"
X = df.drop(columns=[TARGET])
y = df[TARGET]

numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = [c for c in X.columns if c not in numeric]
print("Numeric features:    ", numeric)
print("Categorical features:", categorical)

preprocess = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]), numeric),
    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)
print(f"Train: {len(X_train)}  Test: {len(X_test)}  "
      f"Test default rate: {y_test.mean():.1%}")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.06, max_depth=3, random_state=42),
}

results = {}
fitted = {}
plt.figure(figsize=(7, 5.5))
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    results[name] = {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "precision_default": round(precision_score(y_test, pred), 4),
        "recall_default": round(recall_score(y_test, pred), 4),
        "f1_default": round(f1_score(y_test, pred), 4),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    fitted[name] = pipe
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC {results[name]['roc_auc']:.3f})")
    print(f"\n{name}: AUC {results[name]['roc_auc']:.4f} | "
          f"Acc {results[name]['accuracy']:.4f} | "
          f"Recall(default) {results[name]['recall_default']:.4f}")

plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Loan Risk Models")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "outputs", "roc_curve.png"), dpi=130)
plt.close()

best_name = max(results, key=lambda k: results[k]["roc_auc"])
best = fitted[best_name]
cv_auc = cross_val_score(best, X, y, cv=5, scoring="roc_auc")
results["best_model"] = best_name
results["best_model_cv_roc_auc_mean"] = round(cv_auc.mean(), 4)
results["best_model_cv_roc_auc_std"] = round(cv_auc.std(), 4)
print(f"\nBest model: {best_name}  (5-fold CV ROC-AUC "
      f"{cv_auc.mean():.4f} ± {cv_auc.std():.4f})")
print("\nClassification report (best model, test set):")
print(classification_report(y_test, best.predict(X_test),
                            target_names=["Repaid (0)", "Default (1)"]))

# Confusion matrix chart
cm = confusion_matrix(y_test, best.predict(X_test))
fig, ax = plt.subplots(figsize=(4.6, 4.2))
ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=13)
ax.set_xticks([0, 1], ["Pred: Repaid", "Pred: Default"])
ax.set_yticks([0, 1], ["Actual: Repaid", "Actual: Default"])
ax.set_title(f"Confusion Matrix — {best_name}")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "outputs", "confusion_matrix.png"), dpi=130)
plt.close()

# Permutation importance (model-agnostic, on the test set)
perm = permutation_importance(best, X_test, y_test, n_repeats=8,
                              random_state=42, scoring="roc_auc")
imp = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=True)
plt.figure(figsize=(7.5, 4.8))
imp.plot.barh(color="#003366")
plt.xlabel("Drop in ROC-AUC when the feature is shuffled")
plt.title("Feature Importance — What drives loan risk?")
plt.tight_layout()
plt.savefig(os.path.join(BASE, "outputs", "feature_importance.png"), dpi=130)
plt.close()
print("\nTop risk drivers:\n", imp.sort_values(ascending=False).round(4).to_string())

joblib.dump(best, os.path.join(BASE, "models", "loan_risk_model.pkl"))
with open(os.path.join(BASE, "outputs", "metrics.json"), "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved: models/loan_risk_model.pkl, outputs/metrics.json, 3 charts")
