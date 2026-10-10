import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Smart Loan Approval System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
    models_loaded = True
except Exception as e:
    st.error(f"Models not found. Error: {e}")
    st.info("Make sure models/stacking.pkl, models/preprocessor.pkl, and models/feature_names.pkl exist.")
    models_loaded = False

st.title("🏦 Smart Loan Approval & Recommendation System")
st.markdown("""
This system uses a **Stacking Ensemble** (XGBoost + Random Forest + Logistic Regression)
to predict loan approval decisions. For rejected applicants, it recommends optimal
loan configuration changes to improve approval chances.
""")

st.markdown("---")

if not models_loaded:
    st.stop()

st.sidebar.header("📝 Applicant Information")

with st.sidebar.expander("👤 Demographics", expanded=True):
    age = st.number_input("Age", 18, 80, 35)
    gender = st.selectbox("Gender", ["Male", "Female", "Others"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Widowed"])
    dependents = st.number_input("Dependents", 0, 10, 0)
    education = st.selectbox("Education", ["Graduate", "Post Graduate", "PhD", "Diploma", "High School", "No Formal"])
    residence_type = st.selectbox("Residence Type", ["Urban", "Rural"])
    city_tier = st.selectbox("City Tier", [1, 2, 3, 4])

with st.sidebar.expander("💼 Employment & Income", expanded=True):
    employment_type = st.selectbox("Employment Type", ["Private", "Government", "Self-Employed", "Skilled Labor", "Unemployed"])
    years_job = st.number_input("Years at Current Job", 0, 50, 5)
    total_exp = st.number_input("Total Work Experience", 0, 60, 8)
    annual_income = st.number_input("Annual Income (₹)", 0, 100000000, 600000, step=10000)
    other_income = st.number_input("Other Income (₹)", 0, 10000000, 0, step=1000)

with st.sidebar.expander("💳 Financial Health", expanded=True):
    credit_score = st.number_input("Credit Score", 300, 900, 700)
    existing_loans = st.number_input("Existing Loans", 0, 20, 0)
    existing_loan_amt = st.number_input("Existing Loan Amount (₹)", 0, 50000000, 0, step=10000)
    monthly_emi = st.number_input("Monthly EMI (₹)", 0, 1000000, 0, step=1000)
    debt_to_income = st.number_input("Debt-to-Income Ratio", 0.0, 2.0, 0.3, 0.01)
    savings = st.number_input("Savings (₹)", 0, 50000000, 200000, step=10000)
    investments = st.number_input("Investments (₹)", 0, 50000000, 50000, step=10000)
    credit_card_util = st.number_input("Credit Card Utilization", 0.0, 1.0, 0.3, 0.01)
    num_bank_accounts = st.number_input("Number of Bank Accounts", 0, 20, 2)
    num_credit_cards = st.number_input("Number of Credit Cards", 0, 20, 2)
    loan_defaults = st.number_input("Previous Loan Defaults", 0, 10, 0)
    missed_payments = st.number_input("Missed Payments", 0, 50, 0)

with st.sidebar.expander("📋 Verification", expanded=False):
    tax_filed = st.selectbox("Tax Return Filed", ["Yes", "No"])
    pan_verified = st.selectbox("PAN Verified", ["Yes", "No"])
    aadhaar_verified = st.selectbox("Aadhaar Verified", ["Yes", "No"])

with st.sidebar.expander("💰 Loan Details", expanded=True):
    loan_purpose = st.selectbox("Loan Purpose", ["Home", "Vehicle", "Education", "Medical", "Business", "Personal"])
    loan_amount = st.number_input("Loan Amount (₹)", 10000, 100000000, 1000000, step=10000)
    loan_tenure = st.number_input("Loan Tenure (years)", 1, 30, 5)
    interest_rate = st.number_input("Interest Rate (%)", 1.0, 30.0, 10.0, 0.01)
    collateral = st.selectbox("Collateral", ["Yes", "No"])
    property_value = st.number_input("Property Value (₹)", 0, 100000000, 0, step=10000)
    loan_to_value = st.number_input("Loan to Value (LTV)", 0.0, 200.0, 0.0, 0.01)


def build_applicant():
    return {
        'Age': age,
        'Gender': gender,
        'Marital_Status': marital_status,
        'Education': education,
        'Dependents': dependents,
        'Residence_Type': residence_type,
        'City_Tier': city_tier,
        'Employment_Type': employment_type,
        'Years_at_Current_Job': years_job,
        'Total_Work_Experience': total_exp,
        'Annual_Income': annual_income,
        'Other_Income': other_income,
        'Existing_Loans': existing_loans,
        'Existing_Loan_Amount': existing_loan_amt,
        'Monthly_EMI': monthly_emi,
        'Debt_to_Income': debt_to_income,
        'Savings': savings,
        'Investments': investments,
        'Credit_Card_Utilization': credit_card_util,
        'Number_of_Bank_Accounts': num_bank_accounts,
        'Number_of_Credit_Cards': num_credit_cards,
        'Credit_Score': credit_score,
        'Loan_Defaults': loan_defaults,
        'Missed_Payments': missed_payments,
        'Tax_Return_Filed': tax_filed,
        'PAN_Verified': pan_verified,
        'Aadhaar_Verified': aadhaar_verified,
        'Loan_Purpose': loan_purpose,
        'Loan_Amount': loan_amount,
        'Loan_Tenure': loan_tenure,
        'Interest_Rate': interest_rate,
        'Collateral': collateral,
        'Property_Value': property_value,
        'Loan_to_Value': loan_to_value,
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
    """ML prediction blended with realistic rule-based scoring."""
    enriched = enrich_features(applicant_dict)
    df_input = pd.DataFrame([enriched])
    X_proc = preprocessor.transform(df_input)
    ml_prob = stack.predict_proba(X_proc)[0, 1]

    # Cap ML probability to counter synthetic-data overconfidence
    ml_prob_capped = min(ml_prob, 0.85)

    # --- Rule-based score ---
    rule_score = 0.5

    cs = applicant_dict['Credit_Score']
    if cs >= 800: rule_score += 0.30
    elif cs >= 750: rule_score += 0.22
    elif cs >= 700: rule_score += 0.12
    elif cs >= 670: rule_score += 0.04
    elif cs >= 650: rule_score -= 0.05
    elif cs >= 600: rule_score -= 0.15
    elif cs >= 550: rule_score -= 0.25
    else: rule_score -= 0.35

    dti = applicant_dict['Debt_to_Income']
    if dti <= 0.15: rule_score += 0.12
    elif dti <= 0.25: rule_score += 0.06
    elif dti <= 0.35: rule_score -= 0.03
    elif dti <= 0.45: rule_score -= 0.12
    elif dti <= 0.60: rule_score -= 0.20
    else: rule_score -= 0.32

    mp = applicant_dict['Missed_Payments']
    if mp == 0: rule_score += 0.05
    elif mp == 1: rule_score -= 0.04
    elif mp == 2: rule_score -= 0.12
    elif mp <= 4: rule_score -= 0.22
    else: rule_score -= 0.35

    ld = applicant_dict['Loan_Defaults']
    if ld == 0: rule_score += 0.05
    elif ld == 1: rule_score -= 0.18
    elif ld == 2: rule_score -= 0.28
    else: rule_score -= 0.40

    emp = applicant_dict['Employment_Type']
    if emp == 'Government': rule_score += 0.08
    elif emp == 'Private': rule_score += 0.02
    elif emp == 'Self-Employed': rule_score -= 0.03
    elif emp == 'Skilled Labor': rule_score -= 0.10
    elif emp == 'Unemployed': rule_score -= 0.35

    ccu = applicant_dict['Credit_Card_Utilization']
    if ccu <= 0.30: rule_score += 0.04
    elif ccu <= 0.50: rule_score -= 0.02
    elif ccu <= 0.70: rule_score -= 0.06
    else: rule_score -= 0.12

    if applicant_dict['Tax_Return_Filed'] == 'No': rule_score -= 0.08
    if applicant_dict['PAN_Verified'] == 'No': rule_score -= 0.12
    if applicant_dict['Aadhaar_Verified'] == 'No': rule_score -= 0.08

    if applicant_dict['Collateral'] == 'Yes': rule_score += 0.04
    elif applicant_dict['Loan_Amount'] > 500000: rule_score -= 0.05

    income = applicant_dict['Annual_Income']
    loan = applicant_dict['Loan_Amount']
    lti = loan / (income + 1)
    if lti > 3.0: rule_score -= 0.15
    elif lti > 2.0: rule_score -= 0.08
    elif lti > 1.2: rule_score -= 0.03
    elif lti < 0.8: rule_score += 0.05

    if income >= 2000000: rule_score += 0.08
    elif income >= 1000000: rule_score += 0.04
    elif income < 300000: rule_score -= 0.15
    elif income < 500000: rule_score -= 0.06

    savings = applicant_dict['Savings']
    if savings >= 1000000: rule_score += 0.05
    elif savings >= 300000: rule_score += 0.02
    elif savings < 30000: rule_score -= 0.08

    rule_score = max(0.02, min(0.98, rule_score))

    # Blend: 50% capped ML + 50% rule-based
    final_prob = 0.5 * ml_prob_capped + 0.5 * rule_score
    final_prob = max(0.01, min(0.99, final_prob))
    return final_prob


col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button("🔍 Predict Loan Decision", type="primary", width='stretch')

if predict_button:
    applicant = build_applicant()
    prob = predict_probability(applicant)
    decision = "APPROVED" if prob >= 0.5 else "REJECTED"

    st.markdown("---")
    st.markdown("## 📊 Decision")

    col1, col2, col3 = st.columns(3)

    with col1:
        if decision == "APPROVED":
            st.success(f"### ✅ {decision}")
        else:
            st.error(f"### ❌ {decision}")

    with col2:
        st.metric("Approval Probability", f"{prob:.2%}")

    with col3:
        risk = "Low" if prob > 0.7 else "Medium" if prob > 0.4 else "High"
        st.metric("Risk Category", risk)

    st.markdown("---")

    st.markdown("## 🔍 Why This Decision?")

    try:
        xgb_model = stack.named_estimators_['xgb']
        explainer = shap.TreeExplainer(xgb_model)
        enriched = enrich_features(applicant)
        df_input = pd.DataFrame([enriched])
        X_proc = preprocessor.transform(df_input)
        single_shap = explainer.shap_values(X_proc)[0]

        contributions = list(zip(feature_names, single_shap))
        contributions.sort(key=lambda x: abs(x[1]), reverse=True)

        top_10 = contributions[:10]
        features = [c[0] for c in top_10][::-1]
        values = [c[1] for c in top_10][::-1]
        colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in values]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(features, values, color=colors)
        ax.set_xlabel('SHAP value (impact on approval)')
        ax.set_title('Top 10 Decision Factors')
        ax.axvline(0, color='black', linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.caption("🟢 Green = pushes toward approval | 🔴 Red = pushes toward rejection")

    except Exception as e:
        st.warning(f"SHAP explanation unavailable: {e}")

    st.markdown("---")

    if decision == "REJECTED":
        st.markdown("## 💡 Recommendation Engine")
        st.info("Here are loan configurations that could improve your approval chances.")

        recommendations = []
        for amt_factor in [0.9, 0.8, 0.7, 0.6, 0.5]:
            for tenure_add in [0, 3, 5, 10]:
                for rate_factor in [1.0, 0.9, 0.85]:
                    mod = applicant.copy()
                    mod['Loan_Amount'] = int(loan_amount * amt_factor)
                    mod['Loan_Tenure'] = loan_tenure + tenure_add
                    mod['Interest_Rate'] = round(interest_rate * rate_factor, 2)
                    new_prob = predict_probability(mod)
                    recommendations.append({
                        'Loan Amount (₹)': f"{mod['Loan_Amount']:,}",
                        'Tenure (yr)': mod['Loan_Tenure'],
                        'Rate (%)': mod['Interest_Rate'],
                        'Amount Change': f"{(amt_factor - 1) * 100:+.0f}%",
                        'Rate Change': f"{(rate_factor - 1) * 100:+.0f}%",
                        'Approval Probability': f"{new_prob:.1%}",
                        'Flipped': "✅ Yes" if new_prob >= 0.5 and prob < 0.5 else "—",
                        '_prob': new_prob
                    })

        rec_df = pd.DataFrame(recommendations).sort_values('_prob', ascending=False)
        rec_df = rec_df.drop(columns=['_prob'])
        rec_df = rec_df.drop_duplicates(subset=['Loan Amount (₹)', 'Tenure (yr)', 'Rate (%)'])

        flipped = rec_df[rec_df['Flipped'] == "✅ Yes"].head(5)

        if len(flipped) > 0:
            st.success("### ✅ Configurations that FLIP the decision to Approved:")
            st.dataframe(flipped, width='stretch', hide_index=True)
        else:
            st.warning("No single configuration flips the decision. Consider these top improvements:")
            st.dataframe(rec_df.head(5), width='stretch', hide_index=True)
            st.markdown("**Additional suggestions:**")
            st.markdown("""
            - Add a co-applicant to increase total income
            - Provide additional collateral
            - Improve credit score over time (6+ months of on-time payments)
            - Consult a financial advisor to reduce DTI
            """)

    st.markdown("---")
    with st.expander("📄 Applicant Summary"):
        summary = pd.DataFrame(
            [(k, str(v)) for k, v in applicant.items()],
            columns=["Field", "Value"]
        )
        st.dataframe(summary, width='stretch')

st.markdown("---")
st.caption("🎓 AIML Project — Smart Loan Approval & Configuration Recommendation System")
st.caption("Model: Stacking Ensemble (XGBoost + Random Forest + Logistic Regression)")