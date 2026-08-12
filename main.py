from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np

# Initialize FastAPI
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts customer churn probability and returns a retention strategy based on telecom data.",
    version="1.0.0"
)

# Load model artifacts
try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
    features = joblib.load('model_features.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

# Define the expected JSON payload structure
# Updated to include ALL standard Telco columns your model was trained on
class CustomerData(BaseModel):
    gender: str = 'Female'
    SeniorCitizen: int = 0
    Partner: str = 'Yes'
    Dependents: str = 'No'
    tenure: int = 2
    PhoneService: str = 'Yes'
    MultipleLines: str = 'No'
    InternetService: str = 'Fiber optic'
    OnlineSecurity: str = 'No'
    OnlineBackup: str = 'No'
    DeviceProtection: str = 'No'
    TechSupport: str = 'No'
    StreamingTV: str = 'No'
    StreamingMovies: str = 'No'
    Contract: str = 'Month-to-month'
    PaperlessBilling: str = 'Yes'
    PaymentMethod: str = 'Electronic check'
    MonthlyCharges: float = 85.5
    TotalCharges: float = 150.0

@app.get("/")
def read_root():
    return {"status": "online", "model": "XGBoost Churn Classifier"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        # Convert incoming JSON to DataFrame
        input_df = pd.DataFrame([customer.dict()])
        
        # Ensure TotalCharges is numeric (fixes common dataset string issue)
        input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce').fillna(0)
        
        # Safely scale numerical features IF your notebook used a scaler
        if scaler and hasattr(scaler, 'feature_names_in_'):
            scale_cols = [c for c in scaler.feature_names_in_ if c in input_df.columns]
            if scale_cols:
                input_df[scale_cols] = scaler.transform(input_df[scale_cols])
        
        # We DO NOT run pd.get_dummies() here because your model expects 
        # the raw 'Contract', 'Dependents', etc. columns.
        
        # Ensure all training columns exist (add missing ones as 0 just in case)
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
                
        # Reorder columns to match exactly what the model was trained on
        input_df = input_df[features]
        
        # Predict
        churn_prob = float(model.predict_proba(input_df)[0][1])
        churn_pred = int(model.predict(input_df)[0])
        
        # Business Logic: Retention Strategy
        if churn_prob > 0.6:
            risk_level = "High Risk"
            strategy = "Immediate Action Required: Offer 20% discount on next month's bill or upgrade to a 1-year contract with waived fees."
        elif churn_prob > 0.3:
            risk_level = "Medium Risk"
            strategy = "Monitor: Send a personalized email offering tech support optimization or add-on services."
        else:
            risk_level = "Low Risk"
            strategy = "Standard Engagement: Enroll in standard loyalty program and quarterly check-ins."
            
        return {
            "churn_probability": round(churn_prob * 100, 2),
            "churn_prediction": bool(churn_pred),
            "risk_level": risk_level,
            "recommended_strategy": strategy
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))