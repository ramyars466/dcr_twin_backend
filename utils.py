
import numpy as np
import pandas as pd

def compute_8_scores(row: pd.Series, pd_prob: float, loan_amount_max: float):
    loan_amt = float(row.get("loan_amount", 0) or 0.0)
    monthly_income = float(row.get("monthly_income", 0) or 0.0)
    savings = float(row.get("savings_balance", 0) or 0.0)
    spending_score = float(row.get("spending_score", 50) or 50.0)
    credit_util = float(row.get("credit_utilization", 0) or 0.0)
    employment_stability = float(row.get("employment_stability", 0.5) or 0.5)

    # 1) PD
    PD = float(pd_prob)

    # 2) DCRS
    DCRS = float(round((1 - PD) * 100, 4))

    # 3) Behavioural Risk (0-100)
    behavioural = ((spending_score / 100) * 0.4 + (credit_util) * 0.6) * 100
    behavioural = float(np.clip(behavioural, 0, 100))

    # 4) Financial Stability (0-100, higher better)
    income_ratio = 0.0
    if loan_amt > 0:
        income_ratio = monthly_income / loan_amt
    income_ratio_clipped = float(np.clip(income_ratio, 0, 1))
    savings_ratio = float(np.clip(savings / (loan_amt + 1), 0, 1))
    financial_stability = income_ratio_clipped * 60 + savings_ratio * 40
    financial_stability = float(np.clip(financial_stability, 0, 100))

    # 5) Spending Control
    spending_control = float(np.clip(100 - spending_score, 0, 100))

    # 6) Credit Utilization Risk
    cu_risk = float(np.clip(credit_util * 100, 0, 100))

    # 7) Employment Stability Risk (higher = risk)
    employment_risk = float(np.clip((1 - employment_stability) * 100, 0, 100))

    # 8) Portfolio Impact Score (0-100)
    denom = loan_amount_max if loan_amount_max and loan_amount_max > 0 else max(loan_amt, 1.0)
    portfolio_impact = float(np.clip(PD * (loan_amt / denom) * 100, 0, 100))

    return {
        "PD": PD,
        "DCRS": DCRS,
        "Behavioural_Risk": behavioural,
        "Financial_Stability": financial_stability,
        "Spending_Control": spending_control,
        "CU_Risk": cu_risk,
        "Employment_Risk": employment_risk,
        "Portfolio_Impact": portfolio_impact
    }
