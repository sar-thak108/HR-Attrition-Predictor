"""
Streamlit dashboard: upload an employee record (or fill the form), get a
risk score and a plain-language SHAP-based explanation.

Run with: streamlit run dashboard.py
"""
import streamlit as st
import pandas as pd
import joblib
import shap
import sys, os

sys.path.append(os.path.dirname(__file__))
from importlib import import_module

MODEL_DIR = "/home/claude/AI-HR-Attrition-Project/models"

st.set_page_config(page_title="Attrition Risk & Explainability Dashboard", layout="centered")
st.title("Employee attrition risk — explainable AI dashboard")
st.caption("Predicts attrition risk and explains WHY, using SHAP on real HR data.")

best_model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
label_encoders = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")
feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
best_model_name = joblib.load(f"{MODEL_DIR}/best_model_name.pkl")
rf_model = joblib.load(f"{MODEL_DIR}/random_forest_for_shap.pkl")

st.subheader("Enter employee details")
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 60, 30)
    overtime = st.selectbox("Works overtime?", ["Yes", "No"])
    job_satisfaction = st.slider("Job satisfaction (1=low, 4=high)", 1, 4, 3)
    work_life_balance = st.slider("Work-life balance (1=bad, 4=best)", 1, 4, 3)
    monthly_income = st.number_input("Monthly income", 1000, 20000, 5000)
    years_at_company = st.slider("Years at company", 0, 40, 3)

with col2:
    distance = st.slider("Distance from home (km)", 1, 30, 10)
    marital_status = st.selectbox("Marital status", ["Single", "Married", "Divorced"])
    business_travel = st.selectbox("Business travel", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
    department = st.selectbox("Department", ["Sales", "Research & Development", "Human Resources"])
    stock_option = st.slider("Stock option level", 0, 3, 0)
    num_companies = st.slider("Number of companies worked at", 0, 10, 2)

if st.button("Predict attrition risk", type="primary"):
    employee = {
        "Age": age, "BusinessTravel": business_travel, "DailyRate": 800,
        "Department": department, "DistanceFromHome": distance, "Education": 3,
        "EducationField": "Life Sciences", "EnvironmentSatisfaction": 3, "Gender": "Male",
        "HourlyRate": 60, "JobInvolvement": 3, "JobLevel": 2, "JobRole": "Sales Executive",
        "JobSatisfaction": job_satisfaction, "MaritalStatus": marital_status,
        "MonthlyIncome": monthly_income, "MonthlyRate": 15000, "NumCompaniesWorked": num_companies,
        "OverTime": overtime, "PercentSalaryHike": 12, "PerformanceRating": 3,
        "RelationshipSatisfaction": 3, "StockOptionLevel": stock_option,
        "TotalWorkingYears": years_at_company + 2, "TrainingTimesLastYear": 2,
        "WorkLifeBalance": work_life_balance, "YearsAtCompany": years_at_company,
        "YearsInCurrentRole": min(years_at_company, 3), "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": min(years_at_company, 2)
    }

    row = pd.DataFrame([employee])[feature_names].copy()
    for col, le in label_encoders.items():
        row[col] = le.transform(row[col].astype(str))

    if best_model_name == "Logistic Regression":
        risk = best_model.predict_proba(scaler.transform(row))[0, 1]
    else:
        risk = best_model.predict_proba(row)[0, 1]

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(row)
    sv = shap_values[1][0] if isinstance(shap_values, list) else (
        shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[0]
    )
    contrib_df = pd.DataFrame({
        "Feature": feature_names, "SHAP_contribution": sv
    }).sort_values("SHAP_contribution", key=abs, ascending=False).head(5)

    st.markdown("---")
    if risk > 0.5:
        st.error(f"### Risk score: {risk:.0%} — elevated risk")
    else:
        st.success(f"### Risk score: {risk:.0%} — lower risk")

    st.subheader("Why this prediction (top drivers)")
    for _, r in contrib_df.iterrows():
        direction = "increases" if r["SHAP_contribution"] > 0 else "decreases"
        st.write(f"- **{r['Feature']}** {direction} this employee's risk")

    st.subheader("Suggested HR action")
    top_feature = contrib_df.iloc[0]["Feature"]
    suggestions = {
        "OverTime": "Review workload distribution and overtime frequency for this employee.",
        "JobSatisfaction": "Schedule a 1:1 check-in focused on role fit and day-to-day satisfaction.",
        "WorkLifeBalance": "Discuss flexible scheduling or workload adjustments.",
        "MonthlyIncome": "Review compensation benchmarking against role and market.",
        "StockOptionLevel": "Consider equity/retention incentive review.",
    }
    st.info(suggestions.get(top_feature, "Discuss the top driver above directly with the employee."))
