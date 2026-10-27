import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
from PIL import Image

# ==========================================
# 1. LOAD YOUR EXISTING MODELS
# ==========================================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE, 'models')

@st.cache_resource
def load_models():
    stack = joblib.load(os.path.join(MODELS_DIR, 'stacking.pkl'))
    preprocessor = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))
    return stack, preprocessor, feature_names

try:
    stack, preprocessor, feature_names = load_models()
except Exception as e:
    st.error(f"Error loading models. Make sure .pkl files exist in models/ folder. Details: {e}")
    st.stop()

st.set_page_config(page_title="Smart Loan App", page_icon="🏦", layout="wide")
st.title("🏦 Smart Loan Approval System")
st.markdown("Choose your preferred way to apply for a loan:")

# ==========================================
# 2. CREATE THE TWO TABS
# ==========================================
tab1, tab2 = st.tabs(["📝 Version 1: Manual Entry", "🤖 Version 2: Smart Auto-Fill (AI)"])

# Dictionary to hold form data
def build_applicant(age, income, credit_score, dti, loan_amt, tenure, rate, savings, existing_loan, employment, collateral):
    """Helper to build the feature dictionary."""
    return {
        'Age': age,
        'Gender': 'Male', # Defaulted for brevity in this demo
        'Marital_Status': 'Single',
        'Education': 'Graduate',
        'Dependents': 0,
        'Residence_Type': 'Urban',
        'City_Tier': 1,
        'Employment_Type': employment,
        'Years_at_Current_Job': 3,
        'Total_Work_Experience': 5,
        'Annual_Income': income,
        'Other_Income': 0,
        'Existing_Loans': 1 if existing_loan > 0 else 0,
        'Existing_Loan_Amount': existing_loan,
        'Monthly_EMI': existing_loan / 12 if existing_loan > 0 else 0,
        'Debt_to_Income': dti,
        'Savings': savings,
        'Investments': 0,
        'Credit_Card_Utilization': 0.3,
        'Number_of_Bank_Accounts': 2,
        'Number_of_Credit_Cards': 1,
        'Credit_Score': credit_score,
        'Loan_Defaults': 0,
        'Missed_Payments': 0,
        'Tax_Return_Filed': 'Yes',
        'PAN_Verified': 'Yes',
        'Aadhaar_Verified': 'Yes',
        'Loan_Purpose': 'Home',
        'Loan_Amount': loan_amt,
        'Loan_Tenure': tenure,
        'Interest_Rate': rate,
        'Collateral': collateral,
        'Property_Value': loan_amt * 1.5 if collateral == 'Yes' else 0,
        'Loan_to_Value': 66.0 if collateral == 'Yes' else 0.0,
    }

def enrich_features(d):
    d = d.copy()
    monthly_income = d['Annual_Income'] / 12
    d['EMI_to_Income'] = d['Monthly_EMI'] / (monthly_income + 1)
    d['Loan_to_Income'] = d['Loan_Amount'] / (d['Annual_Income'] + 1)
    d['Savings_to_Loan'] = d['Savings'] / (d['Loan_Amount'] + 1)
    d['Net_Worth'] = d['Savings'] + d['Investments'] - d['Existing_Loan_Amount']
    d['Has_Collateral'] = 1 if d['Collateral'] == 'Yes' else 0
    return d

def predict_probability(applicant_dict):
    enriched = enrich_features(applicant_dict)
    df_input = pd.DataFrame([enriched])
    X_proc = preprocessor.transform(df_input)
    ml_prob = stack.predict_proba(X_proc)[0, 1]
    
    # Calibration layer (simplified)
    ml_prob_capped = min(ml_prob, 0.85)
    rule_score = 0.5
    if applicant_dict['Credit_Score'] < 600: rule_score -= 0.20
    elif applicant_dict['Credit_Score'] >= 750: rule_score += 0.15
    if applicant_dict['Debt_to_Income'] > 0.5: rule_score -= 0.15
    
    final_prob = max(0.01, min(0.99, 0.5 * ml_prob_capped + 0.5 * rule_score))
    return final_prob

# ==========================================
# 3. TAB 1: MANUAL ENTRY
# ==========================================
with tab1:
    st.header("Manual Application")
    st.write("Fill in the primary fields. (Other fields use defaults for this demo).")
    
    with st.form("manual_form"):
        col1, col2 = st.columns(2)
        with col1:
            age_m = st.number_input("Age", 18, 80, 35)
            income_m = st.number_input("Annual Income (₹)", 0, 100000000, 600000, step=10000)
            credit_m = st.number_input("Credit Score (CIBIL)", 300, 900, 700)
            dti_m = st.number_input("Debt-to-Income Ratio", 0.0, 2.0, 0.3, 0.01)
            savings_m = st.number_input("Savings (₹)", 0, 50000000, 200000, step=10000)
        with col2:
            loan_amt_m = st.number_input("Loan Amount (₹)", 10000, 100000000, 1000000, step=10000)
            tenure_m = st.number_input("Loan Tenure (years)", 1, 30, 5)
            rate_m = st.number_input("Interest Rate (%)", 1.0, 30.0, 10.0, 0.01)
            existing_loan_m = st.number_input("Existing Loan Amount (₹)", 0, 50000000, 0, step=10000)
            employment_m = st.selectbox("Employment Type", ["Private", "Government", "Self-Employed"])
            collateral_m = st.selectbox("Collateral", ["Yes", "No"])
            
        submit_manual = st.form_submit_button("Predict (Manual Entry)", use_container_width=True)

    if submit_manual:
        applicant = build_applicant(age_m, income_m, credit_m, dti_m, loan_amt_m, tenure_m, rate_m, savings_m, existing_loan_m, employment_m, collateral_m)
        prob = predict_probability(applicant)
        decision = "APPROVED ✅" if prob >= 0.5 else "REJECTED ❌"
        
        st.markdown("---")
        st.subheader("Decision Result")
        if prob >= 0.5:
            st.success(f"### {decision} (Probability: {prob:.2%})")
        else:
            st.error(f"### {decision} (Probability: {prob:.2%})")

# ==========================================
# 4. TAB 2: SMART AUTO-FILL (AI/OCR)
# ==========================================
with tab2:
    st.header("Smart Auto-Fill Application")
    st.info("Upload your PAN, Aadhaar, and Salary Slip. Our AI will extract the data to auto-fill the form.")
    
    uploaded_files = st.file_uploader(
        "Upload Documents (Multiple files allowed)", 
        type=["png", "jpg", "jpeg"], 
        accept_multiple_files=True
    )
    
    if 'extracted_data' not in st.session_state:
        st.session_state.extracted_data = {}

    if uploaded_files:
        st.write(f"Processing {len(uploaded_files)} document(s)...")
        
        for file in uploaded_files:
            image = Image.open(file)
            st.image(image, caption=f"Uploaded: {file.name}", width=250)
            
            # --- SIMULATED AI EXTRACTION ---
            # In a real hackathon, you would pass this image to Gemini API or Tesseract
            filename_lower = file.name.lower()
            
            if "salary" in filename_lower or "payslip" in filename_lower:
                st.success("✅ Salary Slip detected! Extracted: Income & Employment.")
                st.session_state.extracted_data['income'] = 850000
                st.session_state.extracted_data['employment'] = "Private"
                st.session_state.extracted_data['existing_loan'] = 150000
                
            elif "pan" in filename_lower or "aadhaar" in filename_lower:
                st.success("✅ ID Card detected! Extracted: Age & Identity.")
                st.session_state.extracted_data['age'] = 30
                
            elif "bank" in filename_lower or "statement" in filename_lower:
                st.success("✅ Bank Statement detected! Extracted: Savings.")
                st.session_state.extracted_data['savings'] = 250000

    st.markdown("### Verify Extracted Data")
    extracted = st.session_state.extracted_data
    
    with st.form("smart_form"):
        st.write("Review the auto-filled values below. You can edit them.")
        col1, col2 = st.columns(2)
        with col1:
            age_s = st.number_input("Age", 18, 80, int(extracted.get('age', 35)))
            income_s = st.number_input("Annual Income (₹)", 0, 100000000, int(extracted.get('income', 600000)))
            # Credit score cannot be OCR'd from a slip, must be manual
            credit_s = st.number_input("Credit Score (Manual Entry Required)", 300, 900, 700)
            dti_s = st.number_input("Debt-to-Income Ratio", 0.0, 2.0, 0.3, 0.01)
            savings_s = st.number_input("Savings (₹)", 0, 50000000, int(extracted.get('savings', 200000)), step=10000)
        with col2:
            loan_amt_s = st.number_input("Loan Amount (₹)", 10000, 100000000, 1000000, step=10000)
            tenure_s = st.number_input("Loan Tenure (years)", 1, 30, 5)
            rate_s = st.number_input("Interest Rate (%)", 1.0, 30.0, 10.0, 0.01)
            existing_loan_s = st.number_input("Existing Loan Amount (₹)", 0, 50000000, int(extracted.get('existing_loan', 0)), step=10000)
            employment_s = st.selectbox("Employment Type", ["Private", "Government", "Self-Employed"], index=0 if extracted.get('employment') == "Private" else 1)
            collateral_s = st.selectbox("Collateral", ["Yes", "No"])
            
        submit_smart = st.form_submit_button("Predict (Smart Auto-Fill)", use_container_width=True)
        
    if submit_smart:
        applicant = build_applicant(age_s, income_s, credit_s, dti_s, loan_amt_s, tenure_s, rate_s, savings_s, existing_loan_s, employment_s, collateral_s)
        prob = predict_probability(applicant)
        decision = "APPROVED ✅" if prob >= 0.5 else "REJECTED ❌"
        
        st.markdown("---")
        st.subheader("Decision Result")
        if prob >= 0.5:
            st.success(f"### {decision} (Probability: {prob:.2%})")
        else:
            st.error(f"### {decision} (Probability: {prob:.2%})")