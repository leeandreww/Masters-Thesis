import os
import warnings
import pandas as pd
import numpy as np

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# Suppress non-critical warnings to keep output readable
warnings.filterwarnings('ignore')
# Seed numpy for reproducibility
np.random.seed(42)

# Features used as inputs to the models and for fairness computations
FEATURE_COLS = [
    'Age', 'Gender', 'EducationLevel', 'ExperienceYears',
    'PreviousCompanies', 'DistanceFromCompany',
    'InterviewScore', 'SkillScore', 'PersonalityScore',
    'RecruitmentStrategy'
]

# Privileged / unprivileged definitions for gender and age-binary
PRIV_G   = [{'Gender':    0}]
UNPRIV_G = [{'Gender':    1}]
PRIV_A   = [{'AgeBinary': 0}]
UNPRIV_A = [{'AgeBinary': 1}]


def make_aif_dataset(X_arr, y_arr, prot_arr, prot_col):
    # Construct a DataFrame with features, label and protected attribute
    df_tmp = pd.DataFrame(X_arr, columns=FEATURE_COLS)
    df_tmp['HiringDecision'] = y_arr
    df_tmp[prot_col] = prot_arr
    # Convert into AIF360 BinaryLabelDataset for metric computation
    return BinaryLabelDataset(
        df=df_tmp,
        label_names=['HiringDecision'],
        protected_attribute_names=[prot_col],
        favorable_label=1,
        unfavorable_label=0
    )


def compute_bias_metrics(y_true, y_pred, prot_arr, prot_col,
                         privileged, unprivileged):
    # Build AIF360 datasets using the global X_te features
    ds_true = make_aif_dataset(X_te, y_true, prot_arr, prot_col)
    ds_pred = make_aif_dataset(X_te, y_pred, prot_arr, prot_col)

    # Compute classification-level fairness metrics via AIF360
    m = ClassificationMetric(
        ds_true, ds_pred,
        unprivileged_groups=unprivileged,
        privileged_groups=privileged
    )

    # Positive prediction rates for each group (for reporting)
    priv_rate   = float(y_pred[prot_arr == 0].mean())
    unpriv_rate = float(y_pred[prot_arr == 1].mean())

    return {
        'spd'        : round(m.statistical_parity_difference(), 4),
        'di'         : round(m.disparate_impact(), 4),
        'eod'        : round(m.equal_opportunity_difference(), 4),
        'aod'        : round(m.average_odds_difference(), 4),
        'priv_rate'  : round(priv_rate, 4),
        'unpriv_rate': round(unpriv_rate, 4),
    }


def eval_model(name, y_pred, g_arr, ab_arr):
    # Calculate standard performance metrics on the held-out test labels
    perf = {
        'acc' : round(accuracy_score(y_te, y_pred), 4),
        'prec': round(precision_score(y_te, y_pred, zero_division=0), 4),
        'rec' : round(recall_score(y_te, y_pred, zero_division=0), 4),
        'f1'  : round(f1_score(y_te, y_pred, zero_division=0), 4),
    }
    # Compute fairness metrics for gender and age
    gender_m = compute_bias_metrics(
        y_te, y_pred, g_arr,  'Gender',    PRIV_G, UNPRIV_G)
    age_m = compute_bias_metrics(
        y_te, y_pred, ab_arr, 'AgeBinary', PRIV_A, UNPRIV_A)

    # Print a compact summary for quick inspection
    print(f"\n───────────────────────────────────────────────────────")
    print(f"{name}")
    print(f"  Acc={perf['acc']:.4f}  P={perf['prec']:.4f}  "
          f"R={perf['rec']:.4f}  F1={perf['f1']:.4f}")
    print(f"  [Gender] SPD={gender_m['spd']:+.4f}  DI={gender_m['di']:.4f}  "
          f"EOD={gender_m['eod']:+.4f}  AOD={gender_m['aod']:+.4f}  "
          f"| Male:{gender_m['priv_rate']:.4f}  Female:{gender_m['unpriv_rate']:.4f}")
    print(f"  [Age]    SPD={age_m['spd']:+.4f}  DI={age_m['di']:.4f}  "
          f"EOD={age_m['eod']:+.4f}  AOD={age_m['aod']:+.4f}  "
          f"| Young:{age_m['priv_rate']:.4f}  Old:{age_m['unpriv_rate']:.4f}")

    # Combine performance and fairness metrics into a single dict for export
    return {**perf,
            **{f'g_{k}': v for k, v in gender_m.items()},
            **{f'a_{k}': v for k, v in age_m.items()} }


def find_thresholds(probs, prot_arr, target_rate):
    # Find per-group probability thresholds to match a target positive rate
    th = {}
    for gv in np.unique(prot_arr):
        lo, hi = 0.0, 1.0
        for _ in range(20):
            mid = (lo + hi) / 2
            rate = float((probs[prot_arr == gv] >= mid).mean())
            if rate < target_rate:
                hi = mid
            else:
                lo = mid
        th[gv] = (lo + hi) / 2
    return th


def apply_thresholds(probs, prot_arr, th):
    # Apply group-specific thresholds to convert probabilities into labels
    yp = np.zeros(len(probs), dtype=int)
    for gv, t in th.items():
        yp[prot_arr == gv] = (probs[prot_arr == gv] >= t).astype(int)
    return yp

# data preparation
# Resolve the CSV path relative to this script so it works from different CWDs
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, 'recruitment_data.csv')
if not os.path.exists(data_path):
    data_path = os.path.join(script_dir, '..', 'recruitment_data.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"recruitment_data.csv not found in {script_dir} or parent")

# Load data and create binary age attribute used for fairness checks
df = pd.read_csv(data_path)
df['AgeBinary'] = (df['Age'] >= 35).astype(int)

# Extract arrays used for modeling and fairness evaluation
X      = df[FEATURE_COLS].values
y      = df['HiringDecision'].values
gender = df['Gender'].values
age_b  = df['AgeBinary'].values

# Scale features to [0,1] for consistent model inputs
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test while keeping class balance via stratify=y
(X_tr, X_te,
 y_tr, y_te,
 g_tr, g_te,
 ab_tr, ab_te) = train_test_split(
    X_scaled, y, gender, age_b,
    test_size=0.2, random_state=42, stratify=y
)

# Use training set positive rate as the target for threshold-adjustment
OVERALL_RATE = float(y_tr.mean())

print("Dataset records:", len(df))
print(f"Overall hire rate (train): {OVERALL_RATE:.4f}")

# baseline XGBoost (GradientBoostingClassifier)
print("\n\nTraining baseline XGB model...")
xgb = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=4, random_state=42,
    subsample=0.8, min_samples_leaf=5
)
# Train baseline GBM and compute predictions + probabilities on test set
xgb.fit(X_tr, y_tr)
xgb_pred = xgb.predict(X_te)
xgb_prob = xgb.predict_proba(X_te)[:, 1]

baseline_metrics = eval_model("XGB Baseline", xgb_pred, g_te, ab_te)

# re-weighted XGB
print("\n\nApplying re-weighting and retraining XGB...")
# Compute instance weights from AIF360 reweighing preprocessor
ds_tr_g  = make_aif_dataset(X_tr, y_tr, g_tr, 'Gender')
RW       = Reweighing(unprivileged_groups=UNPRIV_G, privileged_groups=PRIV_G)
ds_tr_rw = RW.fit_transform(ds_tr_g)
sw_aif   = ds_tr_rw.instance_weights

xgb_rw = GradientBoostingClassifier(
    n_estimators=300, learning_rate=0.05,
    max_depth=4, random_state=42,
    subsample=0.8, min_samples_leaf=5
)
# Fit using the computed sample weights to reduce group bias during training
xgb_rw.fit(X_tr, y_tr, sample_weight=sw_aif)
xgb_rw_pred = xgb_rw.predict(X_te)
xgb_rw_prob = xgb_rw.predict_proba(X_te)[:, 1]

rw_metrics = eval_model("XGB Re-weighted", xgb_rw_pred, g_te, ab_te)

# threshold-adjusted XGB
print("\n\nComputing thresholds for XGB predictions...")
# Find group-specific thresholds to match the overall training positive rate
xgb_th = find_thresholds(xgb_prob, g_te, OVERALL_RATE)
print(f"XGB thresholds: Male(0)={xgb_th[0]:.4f} Female(1)={xgb_th[1]:.4f}")
xgb_ta_pred = apply_thresholds(xgb_prob, g_te, xgb_th)

ta_metrics = eval_model("XGB Threshold Adj.", xgb_ta_pred, g_te, ab_te)

# summary
print("\n\n=== Summary ===")
print("Baseline performance and gender bias: ", baseline_metrics)
print("Re-weighted performance and gender bias: ", rw_metrics)
print("Threshold-adjusted performance and gender bias: ", ta_metrics)

perf_records = []
fair_records = []
for label, metrics in [
    ('Baseline', baseline_metrics),
    ('Re-weighted', rw_metrics),
    ('Threshold Adj.', ta_metrics),
]:
    perf_records.append({
        'Variant': label,
        'acc': metrics['acc'],
        'prec': metrics['prec'],
        'rec': metrics['rec'],
        'f1': metrics['f1'],
    })
    fair_records.append({
        'Variant': label,
        'g_spd': metrics['g_spd'],
        'g_di': metrics['g_di'],
        'g_eod': metrics['g_eod'],
        'g_aod': metrics['g_aod'],
        'a_spd': metrics.get('a_spd',''),
        'a_di': metrics.get('a_di',''),
        'a_eod': metrics.get('a_eod',''),
        'a_aod': metrics.get('a_aod',''),
    })

out_dir = os.path.dirname(__file__)
pd.DataFrame(perf_records).to_csv(os.path.join(out_dir, 'xgb_performance.csv'), index=False)
pd.DataFrame(fair_records).to_csv(os.path.join(out_dir, 'xgb_fairness.csv'), index=False)
print("Saved xgb_performance.csv and xgb_fairness.csv in", out_dir)
