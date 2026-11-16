#!/usr/bin/env python3
"""
Predict the default risk of a single loan applicant with the trained model.

Example:
  python3 predict.py --age 32 --annual-income 850000 --employment-years 6 \
      --employment-type Salaried --credit-score 742 --existing-loans 1 \
      --loan-amount 500000 --loan-tenure-months 36 --loan-purpose Personal \
      --debt-to-income-ratio 0.28

Run with no arguments to score a built-in sample applicant.
"""
import argparse
import os
import site
import sys

# Filter out incompatible roaming user site-packages (e.g. NumPy 2.x conflicting with Anaconda C-extensions)
_user_site = site.getusersitepackages()
_user_sites = [_user_site] if isinstance(_user_site, str) else (_user_site or [])
sys.path = [p for p in sys.path if not any(p.lower().startswith(u.lower()) for u in _user_sites)]

import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "loan_risk_model.pkl")

p = argparse.ArgumentParser(description="Loan risk predictor")
p.add_argument("--age", type=int, default=32)
p.add_argument("--annual-income", type=float, default=850_000)
p.add_argument("--employment-years", type=float, default=6)
p.add_argument("--employment-type", default="Salaried",
               choices=["Salaried", "Self-Employed", "Business"])
p.add_argument("--credit-score", type=float, default=742)
p.add_argument("--existing-loans", type=int, default=1)
p.add_argument("--loan-amount", type=float, default=500_000)
p.add_argument("--loan-tenure-months", type=int, default=36)
p.add_argument("--loan-purpose", default="Personal",
               choices=["Home", "Education", "Personal", "Business",
                        "Vehicle", "Medical"])
p.add_argument("--debt-to-income-ratio", type=float, default=0.28)
args = p.parse_args()

applicant = pd.DataFrame([{
    "age": args.age,
    "annual_income": args.annual_income,
    "employment_years": args.employment_years,
    "employment_type": args.employment_type,
    "credit_score": args.credit_score,
    "existing_loans": args.existing_loans,
    "loan_amount": args.loan_amount,
    "loan_tenure_months": args.loan_tenure_months,
    "loan_purpose": args.loan_purpose,
    "debt_to_income_ratio": args.debt_to_income_ratio,
}])

model = joblib.load(MODEL)
risk = float(model.predict_proba(applicant)[0, 1])

if risk < 0.35:
    band, decision = "LOW", "APPROVE — low default risk"
elif risk < 0.60:
    band, decision = "MODERATE", "MANUAL REVIEW — verify income & documents"
else:
    band, decision = "HIGH", "REJECT / ask for collateral or a co-applicant"

print("Applicant:", applicant.to_dict("records")[0])
print(f"\nDefault risk probability : {risk:.1%}")
print(f"Risk category            : {band}")
print(f"Recommendation           : {decision}")
