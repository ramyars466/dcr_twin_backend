# 🧠 DCR Twin — Dynamic Credit Risk Twin

> **A next-generation AI-powered credit risk analytics and loan management platform.**  
> Live App 👉 [https://dcrtwinbackend-1ramyars.streamlit.app](https://dcrtwinbackend-1ramyars.streamlit.app)

---

## 📌 Overview

**DCR Twin** (Dynamic Credit Risk Twin) is an intelligent, interactive credit risk assessment platform that creates a personalized digital "twin" for every borrower. It combines traditional financial inputs with simulated real-time banking signals to continuously assess repayment risk, simulate future scenarios, and trigger intelligent EMI alerts — all powered by a LightGBM model with **94% accuracy**.

Built for modern lenders, fintech innovators, and financial analysts, DCR Twin brings together explainability, simulation, and compliance export in one seamless Streamlit interface.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📊 Pre-Loan Risk Prediction | Instant Probability of Default (PD) scoring using LightGBM |
| 🕸️ 8-Dimension Radar Chart | Visual risk breakdown across income, savings, spending, stability & more |
| 💡 Post-Loan Simulation | Dynamic PD recalculation with simulated salary drops & spending spikes |
| 🔔 EMI Alert System | Real-time EMI payment status alerts using Account Aggregator (AA) simulation |
| 📝 History & Export | Full prediction audit trail with CSV download for compliance |
| 🧩 Explainable AI | SHAP-based TreeExplainer integrated for model transparency |

---

## 🖥️ Web Pages

### 🏠 Home
Introductory landing page explaining the platform's vision, key capabilities, and model accuracy.

### 📊 Risk Prediction
Input borrower financial profile via sidebar and run a pre-loan PD prediction:
- Outputs: Risk label (VERY LOW / LOW / PARTIAL / HIGH / VERY HIGH RISK)
- Recommended loan sanction amount (100% → 75% → 50% → 0%)
- Interactive 8D radar chart of risk dimensions
- Prediction saved to session history

### 💡 Post-Loan Simulation
Simulates post-disbursement borrower behavior using a mock Account Aggregator (AA) feed:
- Detects EMI payment status (PAID / UNPAID)
- Simulates joblessness and income halt
- Adjusts PD score dynamically based on real-world triggers (+0.17 for missed EMI, +0.20 for jobless status)
- Displays remaining principal, months left, and EMI due amount

### 📝 History & Export
- View all pre-loan and post-loan prediction records in a table
- Download complete history as a `.csv` file for audit and compliance use

### ℹ️ About
Project description, mission statement, and contact information.



### Model Pipeline
- **Algorithm:** LightGBM (`lgb_model_v1.pkl`)
- **Input Features (8):**
  - `monthly_income`, `loan_amount`, `credit_utilization`
  - `num_late_payments`, `savings_balance`, `spending_score`
  - `employment_stability`, `age`
- **Output:** Probability of Default (0–1 continuous score)
- **Accuracy:** 94%

### EMI Calculation

Standard reducing-balance EMI formula:

$$
EMI = \frac{P \cdot r \cdot (1+r)^N}{(1+r)^N - 1}
$$

Where `P` = principal, `r` = monthly interest rate, `N` = loan tenure in months.

---

## ⚡ Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/ramyars466/dcr_twin_backend.git
cd dcr_twin_backend
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

---

## 📦 Dependencies

```txt
streamlit
pandas
numpy
joblib
plotly
shap
matplotlib
lightgbm
scikit-learn
```

> Ensure `lgb_model_v1.pkl` is present in the root directory before launching.

---

## 🔢 Risk Label Reference

| PD Score Range | Risk Label | Loan Sanction |
|---|---|---|
| ≤ 0.0005 | ✅ VERY LOW RISK | 100% sanctioned |
| 0.0005 – 0.50 | 🟦 LOW RISK | 100% sanctioned |
| 0.50 – 0.80 | 🟧 PARTIAL RISK | 75% sanctioned |
| 0.80 – 0.90 | 🟪 HIGH RISK | 50% sanctioned |
| > 0.90 | 🔴 VERY HIGH RISK | Loan rejected |

---

## 📊 8-Dimension Risk Scoring

Each borrower is evaluated across 8 normalized dimensions for radar visualization:

| Dimension | Normalization |
|---|---|
| Income | `monthly_income / 150,000` |
| Loan Size | `loan_amount / 700,000` |
| Credit Utilization | Raw value (0–1) |
| Late Payments | `num_late_payments / 10` |
| Savings | `savings_balance / 300,000` |
| Spending | `spending_score / 100` |
| Employment Stability | Raw value (0–1) |
| Age | `(age - 21) / (60 - 21)` |

---

## 🔄 Post-Loan Dynamic Adjustments

The post-loan simulation applies rule-based PD adjustments on top of the model output:

- **EMI not paid this month:** PD += 0.17
- **Jobless status detected:** PD += 0.20
- **Both conditions:** PD capped at 1.0

---

## 👩‍💻 Built By

**Ramya RS**  
AI & ML Engineering Student — Sir MVIT, Bangalore  
📧 [ramyars066@gmail.com](mailto:ramyars066@gmail.com)  
📞 +91 7204085650  
🔗 [GitHub](https://github.com/ramyars466)

---

## 📄 License

This project is developed for academic and innovation purposes. For licensing inquiries, contact the author.

---

> *DCR Twin — Powered by AI. Trusted by Innovators.*

