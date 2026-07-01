
import os
import warnings
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# Suppress warnings to keep output concise during experiments
warnings.filterwarnings('ignore')
# Set seed for reproducible splits and model randomness
np.random.seed(42)

# Feature columns used by the model and for constructing AIF360 datasets
FEATURE_COLS = [
    'Age', 'Gender', 'EducationLevel', 'ExperienceYears',
    'PreviousCompanies', 'DistanceFromCompany',
    'InterviewScore', 'SkillScore', 'PersonalityScore',
    'RecruitmentStrategy'
]

# Define privileged/unprivileged groups for Gender and AgeBinary
PRIV_G   = [{'Gender':    0}]
UNPRIV_G = [{'Gender':    1}]
# age groups as in all.py
PRIV_A   = [{'AgeBinary': 0}]
UNPRIV_A = [{'AgeBinary': 1}]


def make_aif_dataset(X_arr, y_arr, prot_arr, prot_col):
    # Build a DataFrame from feature array and attach label & protected attr
    df_tmp = pd.DataFrame(X_arr, columns=FEATURE_COLS)
    df_tmp['HiringDecision'] = y_arr
    df_tmp[prot_col] = prot_arr
    # Return an AIF360 BinaryLabelDataset for metric computations
    return BinaryLabelDataset(
        df=df_tmp,
        label_names=['HiringDecision'],
        protected_attribute_names=[prot_col],
        favorable_label=1,
        unfavorable_label=0
    )


def compute_bias_metrics(y_true, y_pred, prot_arr, prot_col,
                         privileged, unprivileged):
    # Create AIF360 datasets for true and predicted labels (using global X_te)
    ds_true = make_aif_dataset(X_te, y_true, prot_arr, prot_col)
    ds_pred = make_aif_dataset(X_te, y_pred, prot_arr, prot_col)

    # ClassificationMetric compares the true vs predicted datasets
    m = ClassificationMetric(
        ds_true, ds_pred,
        unprivileged_groups=unprivileged,
        privileged_groups=privileged
    )

    # Group-specific positive prediction rates for reporting
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
    # Compute standard performance metrics on the global test labels y_te
    perf = {
        'acc' : round(accuracy_score(y_te, y_pred), 4),
        'prec': round(precision_score(y_te, y_pred, zero_division=0), 4),
        'rec' : round(recall_score(y_te, y_pred, zero_division=0), 4),
        'f1'  : round(f1_score(y_te, y_pred, zero_division=0), 4),
    }
    # Compute fairness metrics for gender and age using helper
    gender_m = compute_bias_metrics(
        y_te, y_pred, g_arr,  'Gender',    PRIV_G, UNPRIV_G)
    age_m = compute_bias_metrics(
        y_te, y_pred, ab_arr, 'AgeBinary', PRIV_A, UNPRIV_A)

    # Print concise summary to console for quick inspection
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

    # Merge performance and fairness results for later export
    return {**perf,
            **{f'g_{k}': v for k, v in gender_m.items()},
            **{f'a_{k}': v for k, v in age_m.items()} }


def find_thresholds(probs, prot_arr, target_rate):
    # For each protected group, binary-search a probability threshold
    # so that the fraction of positives equals `target_rate` approximately.
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
    # Apply group-specific thresholds to probability scores
    yp = np.zeros(len(probs), dtype=int)
    for gv, t in th.items():
        yp[prot_arr == gv] = (probs[prot_arr == gv] >= t).astype(int)
    return yp

# data preparation 
# Resolve data path relative to script location to support both
# running from this folder or the project root
script_dir = os.path.dirname(__file__)
data_path = os.path.join(script_dir, 'recruitment_data.csv')
if not os.path.exists(data_path):
    data_path = os.path.join(script_dir, '..', 'recruitment_data.csv')

if not os.path.exists(data_path):
    raise FileNotFoundError(f"recruitment_data.csv not found in {script_dir} or parent")

# Read dataset and create binary age indicator used in fairness checks
df = pd.read_csv(data_path)
df['AgeBinary'] = (df['Age'] >= 35).astype(int)

# Extract feature matrix, labels and protected attributes
X      = df[FEATURE_COLS].values
y      = df['HiringDecision'].values
gender = df['Gender'].values
age_b  = df['AgeBinary'].values

# Feature scaling to [0,1] for consistent model input ranges
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split preserving class balance via stratify=y
(X_tr, X_te,
 y_tr, y_te,
 g_tr, g_te,
 ab_tr, ab_te) = train_test_split(
    X_scaled, y, gender, age_b,
    test_size=0.2, random_state=42, stratify=y
)

# Overall positive rate in training used as threshold target
OVERALL_RATE = float(y_tr.mean())

print("Dataset records:", len(df))
print(f"Overall hire rate (train): {OVERALL_RATE:.4f}")

# baseline random forest
print("\n\nTraining baseline RF model...")
rf = RandomForestClassifier(
    n_estimators=500, random_state=42, class_weight='balanced',
    max_depth=10, min_samples_leaf=5
)
# Fit baseline model and compute predictions + probabilities
rf.fit(X_tr, y_tr)
rf_pred = rf.predict(X_te)
rf_prob = rf.predict_proba(X_te)[:, 1]

baseline_metrics = eval_model("RF Baseline", rf_pred, g_te, ab_te)

# re-weighted random forest
print("\n\nApplying re-weighting and retraining RF...")
# Create AIF360 dataset on training split and compute reweighing instance weights
ds_tr_g  = make_aif_dataset(X_tr, y_tr, g_tr, 'Gender')
RW       = Reweighing(unprivileged_groups=UNPRIV_G, privileged_groups=PRIV_G)
ds_tr_rw = RW.fit_transform(ds_tr_g)
sw_aif   = ds_tr_rw.instance_weights

rf_rw = RandomForestClassifier(
    n_estimators=500, random_state=42, class_weight='balanced',
    max_depth=10, min_samples_leaf=5
)
# Fit using instance weights from the reweighing preprocessor
rf_rw.fit(X_tr, y_tr, sample_weight=sw_aif)
rf_rw_pred = rf_rw.predict(X_te)
rf_rw_prob = rf_rw.predict_proba(X_te)[:, 1]

rw_metrics = eval_model("RF Re-weighted", rf_rw_pred, g_te, ab_te)

# threshold-adjusted RF
print("\n\nComputing thresholds for RF predictions...")
# Compute group-specific thresholds to match overall training positive rate
rf_th = find_thresholds(rf_prob, g_te, OVERALL_RATE)
print(f"RF thresholds: Male(0)={rf_th[0]:.4f} Female(1)={rf_th[1]:.4f}")
rf_ta_pred = apply_thresholds(rf_prob, g_te, rf_th)

ta_metrics = eval_model("RF Threshold Adj.", rf_ta_pred, g_te, ab_te)

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
pd.DataFrame(perf_records).to_csv(os.path.join(out_dir, 'rf_performance.csv'), index=False)
pd.DataFrame(fair_records).to_csv(os.path.join(out_dir, 'rf_fairness.csv'), index=False)
print("Saved rf_performance.csv and rf_fairness.csv in", out_dir)
