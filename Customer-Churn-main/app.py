import streamlit as st
import pandas as pd
import joblib
import shap
from xgboost import XGBClassifier

st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

# --- Load models and preprocessing objects ---
model = joblib.load('churn_model.pkl')                  # stacked ensemble — used for the actual prediction
scaler = joblib.load('scaler.pkl')
feature_columns = joblib.load('feature_columns.pkl')
best_threshold = joblib.load('best_threshold.pkl')      # tuned threshold (~0.304), not hardcoded

shap_model = XGBClassifier()
shap_model.load_model('churn_shap_model.json')           # separate tuned XGBoost — used only for SHAP explanations
explainer = shap.TreeExplainer(shap_model)

st.title("📱 Telecom Customer Churn Predictor")
st.write("Enter customer details to predict churn risk and get retention recommendations.")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges", 18.0, 120.0, 70.0)
    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

with col2:
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])

with col3:
    partner = st.selectbox("Has Partner", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents", ["Yes", "No"])
    senior_citizen = st.selectbox("Senior Citizen", ["Yes", "No"])
    total_charges = st.number_input("Total Charges", 0.0, 10000.0, 1000.0)

if st.button("Predict Churn Risk", type="primary"):
    # Build a raw input row matching the original preprocessing
    raw = {
        'gender': 0, 'SeniorCitizen': 1 if senior_citizen == "Yes" else 0,
        'Partner': 1 if partner == "Yes" else 0, 'Dependents': 1 if dependents == "Yes" else 0,
        'tenure': tenure, 'PhoneService': 1, 'PaperlessBilling': 1 if paperless_billing == "Yes" else 0,
        'MonthlyCharges': monthly_charges, 'TotalCharges': total_charges,
    }
   

    st.info("💡 Recommended Action: " + (
        "Offer contract upgrade incentive and review payment method." if prob > best_threshold
        else "No action needed — customer profile is stable."
    ))
