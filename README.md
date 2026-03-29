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

## 🖥️ App Pages

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

---

## 🏗️ Architecture
