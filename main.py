
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd, numpy as np, joblib, os
from typing import List, Any, Dict
import shap

from utils import compute_8_scores

app = FastAPI(title="DCR-Twin API", version="1.0")

MODEL_PATH = os.environ.get("DCR_MODEL_PATH", "lgb_model_v1.pkl")
LOAN_AMOUNT_MAX = float(os.environ.get("DCR_LOAN_MAX", "1.0"))

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    load_error = str(e)
else:
    load_error = None

class BorrowerInput(BaseModel):
    borrower_id: str = None
    monthly_income: float = 0.0
    loan_amount: float = 0.0
    credit_utilization: float = 0.0
    num_late_payments: int = 0
    savings_balance: float = 0.0
    spending_score: float = 50.0
    employment_stability: float = 0.5
    age: int = 30
    class Config:
        extra = "allow"

class BatchRequest(BaseModel):
    borrowers: List[BorrowerInput]

def ensure_model():
    if model is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded: {load_error or 'unknown'}")

def input_to_df(b: BorrowerInput) -> pd.DataFrame:
    d = b.dict()
    d_filtered = {k: v for k, v in d.items() if k != "borrower_id"}
    df = pd.DataFrame([d_filtered])
    return df

def explain_instance(df_row: pd.DataFrame, top_k: int = 5) -> List[Dict[str, Any]]:
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(df_row)
        arr = np.array(shap_values)
        if arr.ndim == 2:
            arr = arr[0]
        features = list(df_row.columns)
        feat_info = []
        for f, sv in zip(features, arr):
            feat_info.append({"feature": f, "shap_value": float(sv), "value": float(df_row.iloc[0][f])})
        feat_info = sorted(feat_info, key=lambda x: abs(x["shap_value"]), reverse=True)
        return feat_info[:top_k]
    except Exception as e:
        return [{"error": "shap_failed", "message": str(e)}]

@app.get("/")
def read_root():
    return {"project": "DCR-Twin API", "model_loaded": model is not None}

@app.get("/model_status")
def model_status():
    return {"model_loaded": model is not None, "load_error": load_error}

@app.post("/predict")
def predict_single(borrower: BorrowerInput):
    ensure_model()
    df = input_to_df(borrower)
    try:
        pd_prob = float(model.predict_proba(df)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")
    loan_max = LOAN_AMOUNT_MAX if LOAN_AMOUNT_MAX > 0 else float(df["loan_amount"].max() if "loan_amount" in df.columns else 1.0)
    scores = compute_8_scores(df.iloc[0], pd_prob, loan_max)
    explanation = explain_instance(df, top_k=5)
    response = {"borrower_id": borrower.borrower_id, "scores": scores, "explanation_top5": explanation}
    return response

@app.post("/batch_predict")
def predict_batch(req: BatchRequest):
    ensure_model()
    results = []
    rows = []
    for b in req.borrowers:
        rows.append({k:v for k,v in b.dict().items() if k!="borrower_id"})
    df_batch = pd.DataFrame(rows)
    try:
        probs = model.predict_proba(df_batch)[:,1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")
    loan_max = LOAN_AMOUNT_MAX if LOAN_AMOUNT_MAX > 0 else float(df_batch["loan_amount"].max() if "loan_amount" in df_batch.columns else 1.0)
    for i, b in enumerate(req.borrowers):
        row = df_batch.iloc[i]
        pd_prob = float(probs[i])
        scores = compute_8_scores(row, pd_prob, loan_max)
        explanation = explain_instance(pd.DataFrame([row]), top_k=5)
        results.append({"borrower_id": b.borrower_id, "scores": scores, "explanation_top5": explanation})
    return {"results": results, "count": len(results)}

@app.get("/health")
def health():
    return {"status": "ok"}
