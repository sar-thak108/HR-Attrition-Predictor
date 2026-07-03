# Explainable AI-based employee attrition prediction — HR decision support

Real, executed pipeline on the IBM HR Analytics Employee Attrition dataset
(1,470 employees, 35 features). Every number in this README was produced by
running the code in this repo — nothing here is illustrative or invented.

## What this project does

1. Predicts which employees are at risk of leaving
2. Explains *why*, for the company overall and for each individual employee, using SHAP
3. Provides a dashboard where an HR user enters an employee's details and gets a risk score with a plain-language explanation and a suggested action

## Project structure

```
AI-HR-Attrition-Project/
├── data/
│   ├── employee_attrition.csv          # Raw dataset (1470 rows, 35 cols)
│   └── employee_attrition_clean.csv    # After dropping constant columns
├── notebooks/
│   ├── 01_eda_preprocessing.py         # Load, clean, EDA, insights
│   ├── 02_model_training.py            # Train + evaluate 4 models
│   ├── 03_shap_explainability.py       # Global + individual SHAP
│   ├── 04_predict_new_employee.py      # Standalone prediction script
│   └── dashboard.py                    # Streamlit HR dashboard
├── models/                             # Saved model, scaler, encoders (.pkl)
├── figures/                            # All EDA + SHAP plots (.png)
├── outputs/                            # CSVs: comparison table, SHAP tables, insights
├── reports/                            # (research report goes here — next step)
├── requirements.txt
└── README.md
```

## How to run it

```bash
pip install -r requirements.txt
cd notebooks
python3 01_eda_preprocessing.py      # ~5 sec, produces figures/ + cleaned CSV
python3 02_model_training.py         # trains 4 models, saves best one
python3 03_shap_explainability.py    # SHAP global + individual explanations
python3 04_predict_new_employee.py   # test prediction on one example employee
streamlit run dashboard.py           # launches the interactive dashboard
```

## Results (real, from this run)

**Dataset**: 1,470 employees, 16.1% overall attrition rate (237 left, 1,233 stayed) —
confirmed imbalanced, handled with `class_weight="balanced"` / `scale_pos_weight`.

**Key EDA insights** (see `outputs/eda_insights.txt`):
- Overtime workers leave at 30.5% vs 10.4% for non-overtime workers
- Lowest job satisfaction group: 22.8% attrition vs 11.3% for highest satisfaction
- Worst work-life balance: 31.2% attrition vs 17.6% for best work-life balance
- Single employees: 25.5% attrition vs 12.5% for married employees
- Median income of leavers: 3,202 vs 5,204 for those who stayed

**Model comparison** (test set, 294 employees held out):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | 0.752 | 0.367 | **0.766** | 0.497 | **0.807** |
| XGBoost | 0.820 | 0.440 | 0.468 | 0.454 | 0.789 |
| Random Forest | 0.820 | 0.350 | 0.149 | 0.209 | 0.778 |
| Decision Tree | 0.759 | 0.342 | 0.553 | 0.423 | 0.638 |

**Logistic Regression was selected as the best model** — highest ROC-AUC and, importantly,
the highest recall (76.6%). In an attrition context, missing an at-risk employee is more
costly than a false alarm, so recall matters more than raw accuracy. It's also the most
interpretable model of the four — a genuine finding, not an assumption, and one that
supports using an interpretable model in an HR context rather than defaulting to the
most complex option.

**Top SHAP features (global importance, from the Random Forest explainer)**:

| Rank | Feature | Category |
|---|---|---|
| 1 | OverTime | Psychological/behavioral |
| 2 | Age | Administrative |
| 3 | MonthlyIncome | Administrative |
| 4 | StockOptionLevel | Administrative |
| 5 | YearsAtCompany | Administrative |

Note: only 2 of the top 10 SHAP features are psychological/behavioral (OverTime,
EnvironmentSatisfaction) — most are administrative/demographic. This is an honest
finding, not a convenient one, and it's worth discussing directly in the report: it means
the ML model's view of "what matters" and your dissertation's psychological-mechanism
view are answering different questions, which is itself a legitimate discussion point
in the integration section rather than something to paper over.

## What's NOT yet in this repo (next steps)

- **Literature review** — needs real search and reading, not fabricated. Happy to help
  build a real literature matrix from actual papers next.
- **Mediation analysis integration** — needs your 102-person item-level survey data
  (BFNE, Edmondson's, UWES-9 responses) to be uploaded so the psychological pipeline
  can be built and connected to the SHAP findings above.
- **Hyperparameter tuning** — the models above use reasonable defaults, not
  exhaustively tuned. GridSearchCV/RandomizedSearchCV can be added if you want to
  push the metrics further, though the gains are usually marginal on a dataset this size.
- **Written research report and slides** — best built once the literature review and
  mediation integration are in place, so the narrative is complete.
