"""
Step 4: Standalone prediction script. Loads the saved model and predicts
attrition risk for a new employee record, with a SHAP-based explanation.

Usage: python3 04_predict_new_employee.py
(Edit the `new_employee` dict below with real values, or import
`predict_employee()` into the dashboard.)
"""
import pandas as pd
import numpy as np
import joblib
import shap

MODEL_DIR = "/home/claude/AI-HR-Attrition-Project/models"

best_model = joblib.load(f"{MODEL_DIR}/best_model.pkl")
scaler = joblib.load(f"{MODEL_DIR}/scaler.pkl")
label_encoders = joblib.load(f"{MODEL_DIR}/label_encoders.pkl")
feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
best_model_name = joblib.load(f"{MODEL_DIR}/best_model_name.pkl")
rf_model = joblib.load(f"{MODEL_DIR}/random_forest_for_shap.pkl")  # used for SHAP explanation regardless of best model


def predict_employee(employee_dict):
    """
    employee_dict: a dict with the same fields as the training data
    (BusinessTravel, Department, etc. as raw strings; numeric fields as numbers).
    Returns (risk_probability, top_shap_contributions_df).
    """
    row = pd.DataFrame([employee_dict])[feature_names].copy()

    # Encode categoricals using the SAME encoders fit during training
    for col, le in label_encoders.items():
        row[col] = le.transform(row[col].astype(str))

    # Predict risk using the best model
    if best_model_name == "Logistic Regression":
        row_scaled = scaler.transform(row)
        risk = best_model.predict_proba(row_scaled)[0, 1]
    else:
        risk = best_model.predict_proba(row)[0, 1]

    # Explain using the Random Forest + SHAP (consistent explainer regardless of best model)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(row)
    sv = shap_values[1][0] if isinstance(shap_values, list) else (
        shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[0]
    )

    contrib_df = pd.DataFrame({
        "Feature": feature_names,
        "Value": row.iloc[0].values,
        "SHAP_contribution": sv
    }).sort_values("SHAP_contribution", key=abs, ascending=False).head(5)

    return risk, contrib_df


if __name__ == "__main__":
    # Example new employee — EDIT THESE VALUES to test a real case
    new_employee = {
        "Age": 29, "BusinessTravel": "Travel_Frequently", "DailyRate": 800,
        "Department": "Sales", "DistanceFromHome": 15, "Education": 3,
        "EducationField": "Marketing", "EnvironmentSatisfaction": 2, "Gender": "Male",
        "HourlyRate": 60, "JobInvolvement": 2, "JobLevel": 1, "JobRole": "Sales Executive",
        "JobSatisfaction": 1, "MaritalStatus": "Single", "MonthlyIncome": 3200,
        "MonthlyRate": 15000, "NumCompaniesWorked": 3, "OverTime": "Yes",
        "PercentSalaryHike": 12, "PerformanceRating": 3, "RelationshipSatisfaction": 2,
        "StockOptionLevel": 0, "TotalWorkingYears": 5, "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 1, "YearsAtCompany": 2, "YearsInCurrentRole": 1,
        "YearsSinceLastPromotion": 0, "YearsWithCurrManager": 1
    }

    risk, explanation = predict_employee(new_employee)
    print(f"\nPredicted attrition risk: {risk:.1%}")
    print(f"\nTop factors driving this prediction:")
    print(explanation.to_string(index=False))
