"""
Step 2: Encode features, train multiple models, evaluate honestly (with class
imbalance handled), and save a comparison table. Real training, real metrics.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)

FIG_DIR = "/home/claude/AI-HR-Attrition-Project/figures"
MODEL_DIR = "/home/claude/AI-HR-Attrition-Project/models"
OUT_DIR = "/home/claude/AI-HR-Attrition-Project/outputs"

# ---------------------------------------------------------------
# 1. Load cleaned data
# ---------------------------------------------------------------
df = pd.read_csv("/home/claude/AI-HR-Attrition-Project/data/employee_attrition_clean.csv")

y = df["Attrition_Flag"]
X = df.drop(columns=["Attrition", "Attrition_Flag"])

# ---------------------------------------------------------------
# 2. Encode categoricals
# ---------------------------------------------------------------
cat_cols = X.select_dtypes(include="object").columns.tolist()
label_encoders = {}
for c in cat_cols:
    le = LabelEncoder()
    X[c] = le.fit_transform(X[c])
    label_encoders[c] = le

feature_names = X.columns.tolist()

# ---------------------------------------------------------------
# 3. Train/test split (stratified — dataset is imbalanced: 16% attrition)
# ---------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Train attrition rate: {y_train.mean():.1%}, Test attrition rate: {y_test.mean():.1%}")

# Scale numeric features for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------
# 4. Train models
#    class_weight='balanced' used because attrition is rare (16%) — without
#    this, models default to predicting "stays" for nearly everyone.
# ---------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced", random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
                              eval_metric="logloss", random_state=42),
}

results = []
roc_data = {}
trained_models = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1": f1, "ROC_AUC": auc})

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)
    trained_models[name] = model

    print(f"\n{name}")
    print(f"  Accuracy={acc:.3f}  Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}  ROC-AUC={auc:.3f}")
    print("  Confusion matrix:\n", confusion_matrix(y_test, y_pred))

results_df = pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)
print("\n=== Model comparison (sorted by ROC-AUC) ===")
print(results_df.to_string(index=False))
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

# ---------------------------------------------------------------
# 5. ROC curve comparison plot
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 6))
colors = ["#378ADD", "#D85A30", "#639922", "#7F77DD"]
for (name, (fpr, tpr, auc)), color in zip(roc_data.items(), colors):
    ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", color=color)
ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC curve comparison across models")
ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG_DIR}/roc_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------
# 6. Select best model (by ROC-AUC, tie-break by Recall — missing an
#    at-risk employee is costlier than a false alarm in this use case)
# ---------------------------------------------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = trained_models[best_model_name]
print(f"\nBest model selected: {best_model_name}")

# ---------------------------------------------------------------
# 7. Save model, scaler, encoders, and feature list
# ---------------------------------------------------------------
joblib.dump(best_model, f"{MODEL_DIR}/best_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(label_encoders, f"{MODEL_DIR}/label_encoders.pkl")
joblib.dump(feature_names, f"{MODEL_DIR}/feature_names.pkl")
joblib.dump(best_model_name, f"{MODEL_DIR}/best_model_name.pkl")

# Also save the trained Random Forest specifically for SHAP (tree explainer is fast + exact)
joblib.dump(trained_models["Random Forest"], f"{MODEL_DIR}/random_forest_for_shap.pkl")
joblib.dump(X_test, f"{MODEL_DIR}/X_test.pkl")
joblib.dump(y_test, f"{MODEL_DIR}/y_test.pkl")
joblib.dump(X_train, f"{MODEL_DIR}/X_train.pkl")

print("\nModel, scaler, encoders saved to", MODEL_DIR)
