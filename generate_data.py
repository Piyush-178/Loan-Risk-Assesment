#!/usr/bin/env python3
"""
Generate a realistic loan-applicant dataset for the Loan Risk Assessment project.

Why synthetic data? Real borrower-level lending data is confidential and is not
published by banks. This generator builds a statistically realistic dataset from
the risk factors lenders actually use (credit score, debt-to-income ratio,
loan-to-income ratio, employment stability, existing loans), with a documented
latent-risk model + noise, so the prediction task is learnable but not trivial.

Output: data/loan_data.csv  (5,000 applicants, target: default 1 = high risk)
"""
import os
import site
import sys

# Filter out incompatible roaming user site-packages (e.g. NumPy 2.x conflicting with Anaconda C-extensions)
_user_site = site.getusersitepackages()
_user_sites = [_user_site] if isinstance(_user_site, str) else (_user_site or [])
sys.path = [p for p in sys.path if not any(p.lower().startswith(u.lower()) for u in _user_sites)]

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 5000

# ---------------- Applicant profile ----------------
age = RNG.integers(21, 66, N)
annual_income = np.clip(RNG.lognormal(mean=13.05, sigma=0.55, size=N), 150_000, 4_500_000)
annual_income = np.round(annual_income / 1000) * 1000

employment_type = RNG.choice(
    ["Salaried", "Self-Employed", "Business"],
    size=N, p=[0.62, 0.23, 0.15],
)
max_years = np.maximum(age - 19, 0)
employment_years = np.minimum(RNG.gamma(shape=2.2, scale=4.0, size=N), max_years)

credit_score = np.clip(RNG.normal(680, 85, N), 300, 900).round()
existing_loans = RNG.choice([0, 1, 2, 3, 4, 5], size=N, p=[0.28, 0.34, 0.20, 0.10, 0.05, 0.03])

# ---------------- Loan request ----------------
loan_purpose = RNG.choice(
    ["Home", "Education", "Personal", "Business", "Vehicle", "Medical"],
    size=N, p=[0.22, 0.16, 0.24, 0.14, 0.16, 0.08],
)
loan_amount = np.clip(annual_income * RNG.uniform(0.2, 2.2, N), 50_000, 3_000_000)
loan_amount = np.round(loan_amount / 1000) * 1000
loan_tenure_months = RNG.choice([12, 24, 36, 48, 60, 84], size=N, p=[0.12, 0.20, 0.28, 0.18, 0.14, 0.08])

# Debt-to-income ratio: rises with the number of existing loans
dti = 0.07 + 0.115 * existing_loans + RNG.normal(0, 0.045, N)
dti = np.clip(dti, 0.02, 0.85).round(3)
loan_to_income = loan_amount / annual_income

# ---------------- Latent risk model (ground truth) ----------------
purpose_effect = pd.Series(loan_purpose).map(
    {"Home": -0.25, "Education": -0.10, "Vehicle": 0.0,
     "Business": 0.20, "Medical": 0.25, "Personal": 0.35}
).to_numpy()
emptype_effect = pd.Series(employment_type).map(
    {"Salaried": 0.0, "Business": 0.10, "Self-Employed": 0.25}
).to_numpy()

z = (
    -3.70
    - 0.011 * (credit_score - 650)      # higher score  -> lower risk
    + 4.20 * dti                        # higher DTI    -> higher risk
    + 1.30 * loan_to_income             # bigger loan vs income -> higher risk
    - 0.050 * employment_years          # job stability -> lower risk
    + 0.350 * existing_loans            # more loans    -> higher risk
    + purpose_effect
    + emptype_effect
    + RNG.normal(0, 0.55, N)            # real-world noise / unobserved factors
)
p_default = 1 / (1 + np.exp(-z))
default = (RNG.random(N) < p_default).astype(int)

df = pd.DataFrame({
    "age": age,
    "annual_income": annual_income.astype(int),
    "employment_years": employment_years.round(1),
    "employment_type": employment_type,
    "credit_score": credit_score,
    "existing_loans": existing_loans,
    "loan_amount": loan_amount.astype(int),
    "loan_tenure_months": loan_tenure_months,
    "loan_purpose": loan_purpose,
    "debt_to_income_ratio": dti,
    "default": default,
})

# A few realistic missing values (handled by imputation in the pipeline)
miss = RNG.random(N) < 0.015
df.loc[miss, "credit_score"] = np.nan
miss2 = RNG.random(N) < 0.010
df.loc[miss2, "employment_years"] = np.nan

HERE = os.path.dirname(os.path.abspath(__file__))
out = os.path.join(HERE, "data", "loan_data.csv")
os.makedirs(os.path.dirname(out), exist_ok=True)
df.to_csv(out, index=False)
print(f"Saved {out}: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Default rate: {df['default'].mean():.1%}  (1 = defaulted / high risk)")
print(df.head(3).to_string())
