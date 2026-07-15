"""
Synthetic data generator for the Credit Card Approval project.

No real applicant data was provided, so this script generates a realistic
synthetic dataset that mirrors the structure of the well-known
"Credit Card Approval Prediction" dataset (application record + credit
record, merged on applicant ID). Swap this file out for a real data loader
(e.g. pd.read_csv('application_record.csv')) when real data is available —
everything downstream (train_models.py, app.py) only depends on the column
names produced here, not on this generator.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 6000

INCOME_TYPES = ["Working", "Commercial associate", "Pensioner", "State servant", "Student"]
EDUCATION = ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary"]
FAMILY_STATUS = ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"]
HOUSING = ["House / apartment", "With parents", "Rented apartment", "Municipal apartment"]
OCCUPATION = ["Laborers", "Core staff", "Managers", "Sales staff", "Drivers", "Accountants", "IT staff", "Medicine staff"]


def _draw_income(income_type):
    base = {
        "Working": 45000,
        "Commercial associate": 60000,
        "Pensioner": 28000,
        "State servant": 40000,
        "Student": 15000,
    }[income_type]
    return max(8000, RNG.normal(base, base * 0.35))


def generate(n=N):
    income_type = RNG.choice(INCOME_TYPES, size=n, p=[0.45, 0.20, 0.20, 0.10, 0.05])
    annual_income = np.array([_draw_income(t) for t in income_type])
    age_years = RNG.integers(21, 70, size=n)
    employment_years = np.clip(
        np.where(income_type == "Pensioner", 0, RNG.gamma(3.0, 2.2, size=n)), 0, 45
    )
    education = RNG.choice(EDUCATION, size=n, p=[0.35, 0.45, 0.15, 0.05])
    family_status = RNG.choice(FAMILY_STATUS, size=n)
    housing = RNG.choice(HOUSING, size=n, p=[0.55, 0.15, 0.20, 0.10])
    occupation = RNG.choice(OCCUPATION, size=n)
    num_children = RNG.poisson(0.5, size=n)
    family_size = num_children + RNG.integers(1, 3, size=n)
    owns_car = RNG.choice([0, 1], size=n, p=[0.4, 0.6])
    owns_realty = RNG.choice([0, 1], size=n, p=[0.35, 0.65])
    credit_history_months = RNG.integers(0, 240, size=n)
    existing_loan_balance = np.clip(RNG.normal(15000, 12000, size=n), 0, None)
    num_credit_lines = RNG.integers(0, 8, size=n)

    # Past payment status, 0 = on time ... 5 = 150+ days overdue / default (mirrors STATUS in credit_record.csv)
    risk_score = (
        -0.00002 * annual_income
        - 0.05 * employment_years
        + 0.00003 * existing_loan_balance
        + 0.15 * num_credit_lines
        - 0.01 * credit_history_months
        + RNG.normal(0, 1.2, size=n)
    )
    payment_status = np.clip(
        np.round((risk_score - risk_score.min()) / (risk_score.max() - risk_score.min()) * 5),
        0, 5
    ).astype(int)

    df = pd.DataFrame({
        "applicant_id": np.arange(1, n + 1),
        "age_years": age_years,
        "income_type": income_type,
        "annual_income": annual_income.round(2),
        "employment_years": employment_years.round(1),
        "education": education,
        "family_status": family_status,
        "housing_type": housing,
        "occupation_type": occupation,
        "num_children": num_children,
        "family_size": family_size,
        "owns_car": owns_car,
        "owns_realty": owns_realty,
        "credit_history_months": credit_history_months,
        "existing_loan_balance": existing_loan_balance.round(2),
        "num_credit_lines": num_credit_lines,
        "payment_status": payment_status,  # 0-5 multi-class, engineered into binary target downstream
    })
    return df


if __name__ == "__main__":
    df = generate()
    df.to_csv("/home/claude/credit_app/data/applicants.csv", index=False)
    print(f"Wrote {len(df)} rows to data/applicants.csv")
    print(df["payment_status"].value_counts().sort_index())
