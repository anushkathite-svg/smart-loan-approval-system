# ?? Smart Loan Approval & Recommendation System

An end-to-end ML system that predicts loan approval decisions and recommends optimal loan configurations for rejected applicants.

## ?? Live Demo

**?? [Try the app](https://smart-loan-approval-system-nj6u6o7rkqek89wcr6n6vt.streamlit.app)**

## ?? Features

- **Predicts** loan approval using a Stacking Ensemble (XGBoost + Random Forest + Logistic Regression)
- **Explains** decisions with SHAP feature importance
- **Recommends** loan configuration changes for rejected applicants
- **Segments** applicants into risk tiers with K-Means clustering

## ?? Model Performance

| Model | Accuracy | F1 | AUC |
|-------|----------|-----|-----|
| XGBoost | 95.27% | 0.9637 | 0.9919 |
| **Stacking** | **95.23%** | **0.9631** | **0.9914** |
| Logistic Regression | 94.87% | 0.9600 | 0.9897 |
| Random Forest | 94.57% | 0.9582 | 0.9899 |
| SVM | 94.73% | 0.9592 | 0.9876 |
| KNN | 91.00% | 0.9289 | 0.9628 |

## ?? Top SHAP Features

1. Credit_Score (3.06)
2. Debt_to_Income (1.01)
3. Annual_Income (0.80)
4. Missed_Payments (0.76)
5. Interest_Rate (0.51)

## ??? Tech Stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, Git

## ?? Run Locally

    git clone https://github.com/anushkathite-svg/smart-loan-approval-system.git
    cd smart-loan-approval-system
    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py

## ?? Project Structure

- notebooks/ - 7 Jupyter notebooks (EDA to Recommender)
- app/streamlit_app.py - Web interface
- models/ - Trained models and SHAP plots
- data/ - Raw and processed data

## ?? Author

Anushka Thite — [@anushkathite-svg](https://github.com/anushkathite-svg)

## ?? License

MIT
