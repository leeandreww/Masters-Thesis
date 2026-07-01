"""
=============================================================================
Section 5.6 – Proxy Feature Identification Analysis
Outputs:
  - mutual_information_results.csv
  - logistic_regression_results.csv

Install: pip install pandas numpy scikit-learn
Run:     python proxy_feature_csv.py
=============================================================================
"""

# Suppress warnings so the output focuses on the analysis results.
import warnings
warnings.filterwarnings("ignore")

# Path handling is based on the script location so the file can run from any CWD.
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss

# ── Configuration ────────────────────────────────────────────────────────────
# Resolve the data file from either the local folder or the project root.
SCRIPT_DIR    = Path(__file__).resolve().parent
LOCAL_DATA_PATH = SCRIPT_DIR / "recruitment_data.csv"
ROOT_DATA_PATH  = SCRIPT_DIR.parent / "recruitment_data.csv"
DATA_PATH       = LOCAL_DATA_PATH if LOCAL_DATA_PATH.exists() else ROOT_DATA_PATH
MI_OUTPUT_PATH = SCRIPT_DIR / "mutual_information_results.csv"
LR_OUTPUT_PATH = SCRIPT_DIR / "logistic_regression_results.csv"
# Age threshold used to define the binary age target for the proxy tests.
AGE_THRESHOLD = 35
RANDOM_STATE  = 42
N_SPLITS      = 5
# Mutual information is estimated multiple times to reduce randomness.
MI_SEEDS      = [42, 123, 456, 789, 1000]

# These are the non-protected features tested for proxy behavior.
NEUTRAL_FEATURES = [
    "EducationLevel", "ExperienceYears", "PreviousCompanies",
    "DistanceFromCompany", "InterviewScore", "SkillScore",
    "PersonalityScore", "RecruitmentStrategy",
]

# ── Load & prepare ───────────────────────────────────────────────────────────
# Load the recruitment dataset and derive the binary age target.
df = pd.read_csv(DATA_PATH)
df["AgeBinary"] = (df["Age"] >= AGE_THRESHOLD).astype(int)

# Separate neutral features from the two protected targets being analyzed.
X_raw    = df[NEUTRAL_FEATURES]
y_gender = df["Gender"]
y_age    = df["AgeBinary"]

# Standardize features for the logistic regression analysis.
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)
# Stratified cross-validation keeps class proportions stable across folds.
cv       = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

# ════════════════════════════════════════════════════════════════════════════
# CSV 1 — Mutual Information
# ════════════════════════════════════════════════════════════════════════════
# Mutual information measures how strongly each neutral feature relates to each target.
print("Running Mutual Information analysis …")

# Repeat the estimator with several seeds to report mean and standard deviation.
mi_gender_runs = np.array([
    mutual_info_classif(X_raw, y_gender, random_state=s) for s in MI_SEEDS
])
mi_age_runs = np.array([
    mutual_info_classif(X_raw, y_age, random_state=s) for s in MI_SEEDS
])

# Assemble the mutual-information results table and sort by gender association.
mi_df = pd.DataFrame({
    "Rank"              : range(1, len(NEUTRAL_FEATURES) + 1),
    "Feature"           : NEUTRAL_FEATURES,
    "MI_Gender_Mean"    : mi_gender_runs.mean(axis=0),
    "MI_Gender_Std"     : mi_gender_runs.std(axis=0),
    "MI_Age_Mean"       : mi_age_runs.mean(axis=0),
    "MI_Age_Std"        : mi_age_runs.std(axis=0),
}).sort_values("MI_Gender_Mean", ascending=False).reset_index(drop=True)

# Re-rank after sorting so the table has contiguous ranks.
mi_df["Rank"] = range(1, len(mi_df) + 1)
mi_df = mi_df.round(6)

# Save the mutual-information table for the report.
mi_df.to_csv(MI_OUTPUT_PATH, index=False)
print(f"  Saved -> {MI_OUTPUT_PATH}  ({len(mi_df)} rows)")

# ════════════════════════════════════════════════════════════════════════════
# CSV 2 — Logistic Regression
# ════════════════════════════════════════════════════════════════════════════
# Logistic regression is used to test whether neutral features can predict the protected targets.
print("Running Logistic Regression analysis …")

def fit_lr(X_scaled, y):
    # Train the model on the full standardized feature set.
    model = LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)
    model.fit(X_scaled, y)

    # Cross-validated predicted probabilities give a less optimistic AUC estimate.
    prob_cv  = cross_val_predict(model, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
    auc_cv   = roc_auc_score(y, prob_cv)

    # Fit-on-full-data probabilities are used to compute a McFadden pseudo-R^2.
    probs_f  = model.predict_proba(X_scaled)[:, 1]
    ll_full  = -log_loss(y, probs_f, normalize=False)

    # Null model log-likelihood uses the base rate of the target variable.
    p_null   = y.mean()
    ll_null  = len(y) * (p_null * np.log(p_null + 1e-15)
                         + (1 - p_null) * np.log(1 - p_null + 1e-15))
    mcfadden = 1 - (ll_full / ll_null)

    # Baseline accuracy for a trivial majority-class classifier.
    baseline = max(y.mean(), 1 - y.mean())

    # Coefficients indicate the direction and strength of each proxy relationship.
    coef_df  = pd.DataFrame({
        "Feature"    : NEUTRAL_FEATURES,
        "Coefficient": model.coef_[0],
    }).sort_values("Coefficient", key=abs, ascending=False).reset_index(drop=True)
    return auc_cv, mcfadden, baseline, coef_df

# Fit one logistic model for gender and one for age.
auc_g, r2_g, base_g, coef_gender = fit_lr(X_scaled, y_gender)
auc_a, r2_a, base_a, coef_age    = fit_lr(X_scaled, y_age)

# ── Section A: Gender coefficients ──────────────────────────────────────────
# Add metadata so the gender coefficient table is self-describing.
gender_coef = coef_gender.copy()
gender_coef.insert(0, "Rank", range(1, len(gender_coef) + 1))
gender_coef["Target"]    = "Gender"
gender_coef["Direction"] = gender_coef["Coefficient"].apply(
    lambda x: "Female (1)" if x > 0 else "Male (0)"
)

# ── Section B: Age coefficients ─────────────────────────────────────────────
# Add metadata so the age coefficient table is self-describing.
age_coef = coef_age.copy()
age_coef.insert(0, "Rank", range(1, len(age_coef) + 1))
age_coef["Target"]    = "Age"
age_coef["Direction"] = age_coef["Coefficient"].apply(
    lambda x: "Old (1)" if x > 0 else "Young (0)"
)

# ── Section C: Model-fit summary ─────────────────────────────────────────────
# Summarize predictive strength and basic reference values for the report.
summary = pd.DataFrame({
    "Metric" : [
        "Cross_Validated_ROC_AUC",
        "Majority_Class_Baseline",
        "McFadden_R2",
        "Neutral_Features_Used",
        "Observations_n",
    ],
    "Gender_Model": [round(auc_g, 4), round(base_g, 4), round(r2_g, 4), 8, 1500],
    "Age_Model"   : [round(auc_a, 4), round(base_a, 4), round(r2_a, 4), 8, 1500],
})

# ── Combine all three sections into one CSV with section labels ──────────────
# Build a single output table so the CSV can hold the full analysis in order.
rows = []

# Section marker for the gender coefficient block.
rows.append({"Section": "COEFFICIENTS: GENDER MODEL",
             "Rank": "", "Feature": "", "Coefficient": "",
             "Target": "", "Direction": ""})
# Append the gender coefficient rows one by one.
for _, r in gender_coef.iterrows():
    rows.append({"Section": "Gender_Coefficients",
                 "Rank": r["Rank"], "Feature": r["Feature"],
                 "Coefficient": round(r["Coefficient"], 6),
                 "Target": r["Target"], "Direction": r["Direction"]})

# Insert a blank separator row before the age block.
rows.append({"Section": "", "Rank": "", "Feature": "",
             "Coefficient": "", "Target": "", "Direction": ""})  # blank spacer

# Section marker for the age coefficient block.
rows.append({"Section": "COEFFICIENTS: AGE MODEL",
             "Rank": "", "Feature": "", "Coefficient": "",
             "Target": "", "Direction": ""})
# Append the age coefficient rows one by one.
for _, r in age_coef.iterrows():
    rows.append({"Section": "Age_Coefficients",
                 "Rank": r["Rank"], "Feature": r["Feature"],
                 "Coefficient": round(r["Coefficient"], 6),
                 "Target": r["Target"], "Direction": r["Direction"]})

# Insert another blank separator row before the summary block.
rows.append({"Section": "", "Rank": "", "Feature": "",
             "Coefficient": "", "Target": "", "Direction": ""})  # blank spacer

# Section marker for model-fit summary statistics.
rows.append({"Section": "MODEL FIT SUMMARY",
             "Rank": "", "Feature": "", "Coefficient": "",
             "Target": "", "Direction": ""})
# Append each summary metric as a final grouped row.
for _, r in summary.iterrows():
    rows.append({"Section": "Model_Fit",
                 "Rank": "", "Feature": r["Metric"],
                 "Coefficient": "",
                 "Target": str(r["Gender_Model"]),
                 "Direction": str(r["Age_Model"])})

lr_df = pd.DataFrame(rows, columns=["Section", "Rank", "Feature",
                                     "Coefficient", "Target", "Direction"])

# Write the final combined CSV for the proxy-feature analysis section.
lr_df.to_csv(LR_OUTPUT_PATH, index=False)
print(f"  Saved -> {LR_OUTPUT_PATH}  ({len(lr_df)} rows)")
print("\nDone.")