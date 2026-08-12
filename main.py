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

# Load model artifacts (make sure these files are in the same directory)
try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl') # If you didn't use a scaler, remove this
    features = joblib.load('model_features.pkl')
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

# Define the expected JSON payload structure
class CustomerData(BaseModel):
    tenure: int
    MonthlyCharges: float
    TotalCharges: float
    Contract: str  # e.g., 'Month-to-month', 'One year', 'Two year'
    InternetService: str  # e.g., 'DSL', 'Fiber optic', 'No'
    TechSupport: str  # e.g., 'Yes', 'No', 'No internet service'
    # Add any other features your model expects here...

@app.get("/")
def read_root():
    return {"status": "online", "model": "XGBoost Churn Classifier"}

@app.post("/predict")
def predict_churn(customer: CustomerData):
    try:
        # Convert incoming JSON to DataFrame
        input_df = pd.DataFrame([customer.dict()])
        
        # One-hot encode (must match your notebook's preprocessing)
        input_df = pd.get_dummies(input_df, columns=['Contract', 'InternetService', 'TechSupport'])
        
        # Ensure all training columns exist (add missing ones as 0)
        for col in features:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[features]
        
        # Scale numerical features (if you used a scaler in your notebook)
        input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])
        
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