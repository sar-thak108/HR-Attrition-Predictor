"""
Step 1: Load, clean, and explore the IBM HR Analytics Employee Attrition dataset.
Real dataset, 1470 employees, 35 columns. No synthetic data used.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
FIG_DIR = "/home/claude/AI-HR-Attrition-Project/figures"

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("/home/claude/AI-HR-Attrition-Project/data/employee_attrition.csv", encoding="utf-8-sig")
print("Shape:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum().sum(), "total missing cells")
print("\nDuplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------------
# 2. Drop constant / non-informative columns (documented, not silently dropped)
#    EmployeeCount, StandardHours, Over18 are constant across all 1470 rows.
#    EmployeeNumber is just an ID.
# ---------------------------------------------------------------
constant_cols = [c for c in df.columns if df[c].nunique() == 1]
print("\nConstant columns (dropped):", constant_cols)
df = df.drop(columns=constant_cols + ["EmployeeNumber"])

# ---------------------------------------------------------------
# 3. Target variable
# ---------------------------------------------------------------
df["Attrition_Flag"] = df["Attrition"].map({"Yes": 1, "No": 0})
attrition_rate = df["Attrition_Flag"].mean()
print(f"\nOverall attrition rate: {attrition_rate:.1%}")
print(df["Attrition"].value_counts())

# ---------------------------------------------------------------
# 4. EDA visualizations (saved as PNGs — real output, real numbers)
# ---------------------------------------------------------------

# 4a. Attrition by OverTime
fig, ax = plt.subplots(figsize=(6, 4))
pd.crosstab(df["OverTime"], df["Attrition"], normalize="index").plot(kind="bar", stacked=True, ax=ax, color=["#5DCAA5", "#D85A30"])
ax.set_title("Attrition rate by overtime status")
ax.set_ylabel("Proportion")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/attrition_by_overtime.png", dpi=150)
plt.close()

# 4b. Attrition by JobSatisfaction
fig, ax = plt.subplots(figsize=(6, 4))
pd.crosstab(df["JobSatisfaction"], df["Attrition"], normalize="index").plot(kind="bar", stacked=True, ax=ax, color=["#5DCAA5", "#D85A30"])
ax.set_title("Attrition rate by job satisfaction level (1=low, 4=high)")
ax.set_ylabel("Proportion")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/attrition_by_jobsatisfaction.png", dpi=150)
plt.close()

# 4c. Attrition by WorkLifeBalance
fig, ax = plt.subplots(figsize=(6, 4))
pd.crosstab(df["WorkLifeBalance"], df["Attrition"], normalize="index").plot(kind="bar", stacked=True, ax=ax, color=["#5DCAA5", "#D85A30"])
ax.set_title("Attrition rate by work-life balance (1=bad, 4=best)")
ax.set_ylabel("Proportion")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/attrition_by_worklifebalance.png", dpi=150)
plt.close()

# 4d. Monthly Income distribution by Attrition
fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", ax=ax, palette=["#5DCAA5", "#D85A30"])
ax.set_title("Monthly income distribution by attrition")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/income_by_attrition.png", dpi=150)
plt.close()

# 4e. Correlation heatmap (numeric features only)
numeric_df = df.select_dtypes(include=[np.number])
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(numeric_df.corr(), cmap="RdBu_r", center=0, ax=ax, cbar_kws={"shrink": 0.7})
ax.set_title("Correlation matrix — numeric features")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/correlation_heatmap.png", dpi=150)
plt.close()

# 4f. YearsAtCompany distribution
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(data=df, x="YearsAtCompany", hue="Attrition", multiple="stack", bins=20,
             palette=["#5DCAA5", "#D85A30"], ax=ax)
ax.set_title("Tenure distribution by attrition")
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/tenure_distribution.png", dpi=150)
plt.close()

print("\nSaved 6 EDA figures to", FIG_DIR)

# ---------------------------------------------------------------
# 5. Key HR insights (computed from real data, not assumed)
# ---------------------------------------------------------------
insights = []
ot_rate = df.groupby("OverTime")["Attrition_Flag"].mean()
insights.append(f"Employees working overtime leave at {ot_rate['Yes']:.1%} vs {ot_rate['No']:.1%} for those who don't.")

js_rate = df.groupby("JobSatisfaction")["Attrition_Flag"].mean()
insights.append(f"Lowest job satisfaction (1) group attrition: {js_rate[1]:.1%} vs highest (4): {js_rate[4]:.1%}.")

wlb_rate = df.groupby("WorkLifeBalance")["Attrition_Flag"].mean()
insights.append(f"Worst work-life balance (1) group attrition: {wlb_rate[1]:.1%} vs best (4): {wlb_rate[4]:.1%}.")

single_rate = df.groupby("MaritalStatus")["Attrition_Flag"].mean()
insights.append(f"Single employees attrition: {single_rate['Single']:.1%} vs Married: {single_rate['Married']:.1%}.")

income_leave = df[df.Attrition == "Yes"]["MonthlyIncome"].median()
income_stay = df[df.Attrition == "No"]["MonthlyIncome"].median()
insights.append(f"Median monthly income — left: {income_leave:.0f}, stayed: {income_stay:.0f}.")

print("\nKey insights:")
for i in insights:
    print(" -", i)

# Save cleaned dataset for the next step
df.to_csv("/home/claude/AI-HR-Attrition-Project/data/employee_attrition_clean.csv", index=False)
with open("/home/claude/AI-HR-Attrition-Project/outputs/eda_insights.txt", "w") as f:
    f.write(f"Overall attrition rate: {attrition_rate:.1%}\n\n")
    f.write("Key HR insights (computed from real data):\n")
    for i in insights:
        f.write(f"- {i}\n")

print("\nCleaned dataset saved. EDA insights saved to outputs/eda_insights.txt")
