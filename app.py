import streamlit as st
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# Load scaler, PCA, and model
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
model = joblib.load('trained_rf_model.pkl')

# App title
st.title("🏦 Loan Default Prediction Application")

# Sidebar input
st.sidebar.header("Loan Application Inputs")

Interest_rate_spread = st.sidebar.number_input('Interest Rate Spread', -3.64, 3.36, 0.0, 0.01)
Upfront_charges = st.sidebar.number_input('Upfront Charges', 0.0, 579000.0, 1000.0, 100.0)
rate_of_interest = st.sidebar.number_input('Rate of Interest', 0.0, 61.0, 7.0, 0.1)
property_value = st.sidebar.number_input('Property Value', 8000.0, 16500000.0, 200000.0, 1000.0)
LTV = st.sidebar.number_input('Loan-to-Value (LTV)', 0.0, 1.0, 0.8, 0.01)

credit_type = st.sidebar.selectbox('Credit Type', ['EXP', 'EQUI', 'CRIF', 'CIB'])
submission = st.sidebar.selectbox('Submission of Application', ['to_inst', 'not_inst'])

dtir1 = st.sidebar.number_input('DTI Ratio (dtir1)', 0.97, 7831.25, 2.0, 0.01)
income = st.sidebar.number_input('Income', 0.0, 3570000.0, 50000.0, 1000.0)
loan_amount = st.sidebar.number_input('Loan Amount', 16500.0, 3580000.0, 100000.0, 1000.0)

# Encode categorical values
credit_type_map = {'EXP': 1, 'EQUI': 2, 'CRIF': 3, 'CIB': 4}
submission_map = {'to_inst': 1, 'not_inst': 0}
credit_type_encoded = credit_type_map[credit_type]
submission_encoded = submission_map[submission]

# Create input DataFrame
input_data = pd.DataFrame([{
    'Interest_rate_spread': Interest_rate_spread,
    'Upfront_charges': Upfront_charges,
    'rate_of_interest': rate_of_interest,
    'property_value': property_value,
    'LTV': LTV,
    'credit_type': credit_type_encoded,
    'submission_of_application': submission_encoded,
    'dtir1': dtir1,
    'income': income,
    'loan_amount': loan_amount
}])

# Ensure correct order of columns
expected_columns = scaler.feature_names_in_
input_data = input_data[expected_columns]

# Scale input
input_scaled = scaler.transform(input_data)

# Apply PCA
input_pca = pca.transform(input_scaled)

# Button to predict
predict_btn = st.button("🔮 Predict Loan Outcome")

if predict_btn:
    prediction = model.predict(input_pca)[0]
    probs = model.predict_proba(input_pca)[0]
    prob_paid = float(probs[0])
    prob_default = float(probs[1])

    st.subheader("📊 Prediction Result:")
    if prediction == 1:
        st.error("⚠️ This loan application is likely to DEFAULT.")
    else:
        st.success("✅ This loan application is likely to be PAID BACK.")

    st.markdown(f"""
    **Confidence Levels:**
    - 🟢 **Paid Back**: {prob_paid * 100:.2f}%
    - 🔴 **Default**: {prob_default * 100:.2f}%
    """)

# Optionally show raw input
if st.checkbox("Show input data"):
    st.json(input_data.to_dict(orient="records")[0])
