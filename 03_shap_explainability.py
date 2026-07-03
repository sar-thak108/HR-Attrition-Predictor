"""
Step 3: SHAP explainability — global feature importance + individual
employee waterfall explanations. Uses the Random Forest (tree explainer
is exact and fast) trained in step 2.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import shap

MODEL_DIR = "/home/claude/AI-HR-Attrition-Project/models"
FIG_DIR = "/home/claude/AI-HR-Attrition-Project/figures"
OUT_DIR = "/home/claude/AI-HR-Attrition-Project/outputs"

rf_model = joblib.load(f"{MODEL_DIR}/random_forest_for_shap.pkl")
X_test = joblib.load(f"{MODEL_DIR}/X_test.pkl")
y_test = joblib.load(f"{MODEL_DIR}/y_test.pkl")
feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")

# ---------------------------------------------------------------
# 1. Build the SHAP explainer and compute values
# ---------------------------------------------------------------
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# shap_values shape handling (binary classifier -> take class-1 contributions)
if isinstance(shap_values, list):
    sv = shap_values[1]
elif shap_values.ndim == 3:
    sv = shap_values[:, :, 1]
else:
    sv = shap_values

# ---------------------------------------------------------------
# 2. Global feature importance (mean absolute SHAP value)
# ---------------------------------------------------------------
mean_abs_shap = np.abs(sv).mean(axis=0)
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Mean_Abs_SHAP": mean_abs_shap
}).sort_values("Mean_Abs_SHAP", ascending=False)

print("=== Top 10 features by global SHAP importance ===")
print(importance_df.head(10).to_string(index=False))
importance_df.to_csv(f"{OUT_DIR}/shap_global_importance.csv", index=False)

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
top15 = importance_df.head(15).sort_values("Mean_Abs_SHAP")
ax.barh(top15["Feature"], top15["Mean_Abs_SHAP"], color="#7F77DD")
ax.set_xlabel("Mean |SHAP value| (average impact on attrition risk)")
ax.set_title("Global feature importance (SHAP) — Random Forest")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_global_importance.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. Classify top features: psychological/behavioral vs administrative
#    (documented judgment call, not automatic — this is the step that
#    feeds the mediation-integration part of the project)
# ---------------------------------------------------------------
psychological_vars = {"JobSatisfaction", "EnvironmentSatisfaction", "RelationshipSatisfaction",
                       "WorkLifeBalance", "JobInvolvement", "OverTime", "PerformanceRating"}
administrative_vars = {"Age", "DistanceFromHome", "MonthlyIncome", "DailyRate", "HourlyRate",
                        "MonthlyRate", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole",
                        "YearsSinceLastPromotion", "YearsWithCurrManager", "NumCompaniesWorked",
                        "StockOptionLevel", "TrainingTimesLastYear", "PercentSalaryHike",
                        "Education", "JobLevel"}

top10 = importance_df.head(10).copy()
top10["Category"] = top10["Feature"].apply(
    lambda f: "Psychological/behavioral" if f in psychological_vars
    else ("Administrative" if f in administrative_vars else "Categorical/other")
)
print("\n=== Top 10 features categorized ===")
print(top10.to_string(index=False))
top10.to_csv(f"{OUT_DIR}/shap_top_features_categorized.csv", index=False)

# ---------------------------------------------------------------
# 4. Individual waterfall explanation for ONE real test-set employee
#    who the model correctly flagged as high risk
# ---------------------------------------------------------------
proba = rf_model.predict_proba(X_test)[:, 1]
# find a genuinely high-risk case that actually left
high_risk_idx = np.where((y_test.values == 1) & (proba > np.median(proba)))[0]
if len(high_risk_idx) > 0:
    idx = high_risk_idx[0]
else:
    idx = int(np.argmax(proba))

row = X_test.iloc[idx]
row_shap = sv[idx]
base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

contrib_df = pd.DataFrame({
    "Feature": feature_names,
    "Value": row.values,
    "SHAP_contribution": row_shap
}).sort_values("SHAP_contribution", key=abs, ascending=False).head(8)

print(f"\n=== Individual explanation for test employee index {idx} ===")
print(f"Base rate (avg predicted risk): {base_value:.3f}")
print(f"This employee's predicted risk: {proba[idx]:.3f}")
print(contrib_df.to_string(index=False))
contrib_df.to_csv(f"{OUT_DIR}/shap_individual_example.csv", index=False)

# Waterfall-style plot for this one employee
fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#D85A30" if v > 0 else "#1D9E75" for v in contrib_df["SHAP_contribution"]]
ax.barh(contrib_df["Feature"], contrib_df["SHAP_contribution"], color=colors)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_xlabel("SHAP contribution (pushes risk up / down)")
ax.set_title(f"Why this employee is flagged as high risk\n(base rate {base_value:.1%} -> predicted {proba[idx]:.1%})")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/shap_individual_waterfall.png", dpi=150)
plt.close()

print("\nSaved SHAP outputs to", FIG_DIR, "and", OUT_DIR)
