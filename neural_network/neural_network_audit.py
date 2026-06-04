
import os
import warnings
import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score)

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing

# Suppress non-critical warnings to keep output clean
warnings.filterwarnings('ignore')
# Fix the random seed for reproducible results
np.random.seed(42)

# Columns used as input features for the models
FEATURE_COLS = [
    'Age', 'Gender', 'EducationLevel', 'ExperienceYears',
    'PreviousCompanies', 'DistanceFromCompany',
    'InterviewScore', 'SkillScore', 'PersonalityScore',
    'RecruitmentStrategy'
]

# Definitions for privileged / unprivileged groups used by AIF360
# For Gender: 0=Male considered privileged here, 1=Female unprivileged
PRIV_G   = [{'Gender':    0}]
UNPRIV_G = [{'Gender':    1}]

# For AgeBinary: 0=Young privileged, 1=Old unprivileged (as encoded below)
PRIV_A   = [{'AgeBinary': 0}]
UNPRIV_A = [{'AgeBinary': 1}]


def make_aif_dataset(X_arr, y_arr, prot_arr, prot_col):
    """Create an AIF360 BinaryLabelDataset from arrays.

    Parameters:
    - X_arr: 2D numpy array of features (columns must match FEATURE_COLS)
    - y_arr: 1D array of binary labels (1=favorable, 0=unfavorable)
    - prot_arr: 1D array containing protected attribute values
    - prot_col: string name to assign to the protected attribute column

    Returns an AIF360 BinaryLabelDataset configured for fairness metrics.
    """
    df_tmp = pd.DataFrame(X_arr, columns=FEATURE_COLS)
    # Add the labels and protected attribute into the DataFrame
    df_tmp['HiringDecision'] = y_arr
    df_tmp[prot_col] = prot_arr
    # Construct and return the BinaryLabelDataset expected by aif360
    return BinaryLabelDataset(
        df=df_tmp,
        label_names=['HiringDecision'],
        protected_attribute_names=[prot_col],
        favorable_label=1,
        unfavorable_label=0
    )


def compute_bias_metrics(y_true, y_pred, prot_arr, prot_col,
                         privileged, unprivileged):
    """Compute a set of fairness metrics using AIF360.

    Notes:
    - This function uses the global `X_te` feature matrix when creating
      the AIF360 datasets for the true and predicted labels. That mirrors
      the original script's behavior (no change to calling signature).
    - Returns a dict with statistical parity difference (spd),
      disparate impact (di), equal opportunity difference (eod),
      average odds difference (aod), and group-specific positive rates.
    """
    # Create AIF360 datasets for the true and predicted labels (uses X_te)
    ds_true = make_aif_dataset(X_te, y_true, prot_arr, prot_col)
    ds_pred = make_aif_dataset(X_te, y_pred, prot_arr, prot_col)

    # Compute classification-level fairness metrics
    m = ClassificationMetric(
        ds_true, ds_pred,
        unprivileged_groups=unprivileged,
        privileged_groups=privileged
    )

    # Positive prediction rates per group (convenience values)
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
    """Evaluate model performance and fairness for gender and age.

    - `name`: label printed to summarize the variant
    - `y_pred`: predicted labels on the test set
    - `g_arr`: protected attribute array for Gender on the test set
    - `ab_arr`: protected attribute array for AgeBinary on the test set

    Returns a merged dictionary containing performance and fairness values.
    """
    # Standard classification performance metrics (computed against global y_te)
    perf = {
        'acc' : round(accuracy_score(y_te, y_pred), 4),
        'prec': round(precision_score(y_te, y_pred, zero_division=0), 4),
        'rec' : round(recall_score(y_te, y_pred, zero_division=0), 4),
        'f1'  : round(f1_score(y_te, y_pred, zero_division=0), 4),
    }
    # Compute fairness metrics for Gender and AgeBinary
    gender_m = compute_bias_metrics(
        y_te, y_pred, g_arr,  'Gender',    PRIV_G, UNPRIV_G)
    age_m = compute_bias_metrics(
        y_te, y_pred, ab_arr, 'AgeBinary', PRIV_A, UNPRIV_A)

    # Print a concise summary to the console for quick inspection
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

    # Merge performance and fairness metrics into a single dict for saving
    return {**perf,
            **{f'g_{k}': v for k, v in gender_m.items()},
            **{f'a_{k}': v for k, v in age_m.items()} }


def find_thresholds(probs, prot_arr, target_rate):
    """Find per-group probability thresholds that achieve target positive rate.

    For each group value in `prot_arr` the function uses binary search
    over probability thresholds in [0,1] to find a threshold such that the
    fraction of samples with prob >= threshold is approximately `target_rate`.
    """
    th = {}
    for gv in np.unique(prot_arr):
        lo, hi = 0.0, 1.0
        # 20 iterations of bisection gives sufficient precision for threshold
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
    """Apply group-specific thresholds to probabilities to produce labels."""
    yp = np.zeros(len(probs), dtype=int)
    for gv, t in th.items():
        yp[prot_arr == gv] = (probs[prot_arr == gv] >= t).astype(int)
    return yp


# ---------------------- Data preparation ----------------------
# Read input recruitment data and create a binary age indicator
df = pd.read_csv('recruitment_data.csv')
df['AgeBinary'] = (df['Age'] >= 35).astype(int)

# Extract feature matrix, labels and protected attributes
X      = df[FEATURE_COLS].values
y      = df['HiringDecision'].values
gender = df['Gender'].values
age_b  = df['AgeBinary'].values

# Scale features to [0,1] which helps neural network training
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train / test while preserving label balance with stratify
(X_tr, X_te,
 y_tr, y_te,
 g_tr, g_te,
 ab_tr, ab_te) = train_test_split(
    X_scaled, y, gender, age_b,
    test_size=0.2, random_state=42, stratify=y
)

# Overall positive (hire) rate in the training set used for thresholding
OVERALL_RATE = float(y_tr.mean())

print("Dataset records:", len(df))
print(f"Overall hire rate (train): {OVERALL_RATE:.4f}")


# ---------------------- Baseline neural network ----------------------
print("\n\nTraining baseline NN model...")
nn = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16), activation='relu',
    solver='adam', alpha=0.0001, batch_size=32,
    learning_rate_init=0.001, max_iter=500,
    random_state=42, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20
)
# Fit on training data and compute test predictions and probabilities
nn.fit(X_tr, y_tr)
nn_pred = nn.predict(X_te)
nn_prob = nn.predict_proba(X_te)[:, 1]

baseline_metrics = eval_model("NN Baseline", nn_pred, g_te, ab_te)


# ---------------------- Re-weighted neural network ----------------------
print("\n\nApplying re-weighting and retraining NN...")
# Create an AIF360 dataset for training to compute instance weights
ds_tr_g  = make_aif_dataset(X_tr, y_tr, g_tr, 'Gender')
RW       = Reweighing(unprivileged_groups=UNPRIV_G, privileged_groups=PRIV_G)
ds_tr_rw = RW.fit_transform(ds_tr_g)
# Extract instance weights computed by the Reweighing preprocessor
sw_aif   = ds_tr_rw.instance_weights

nn_rw = MLPClassifier(
    hidden_layer_sizes=(64, 32, 16), activation='relu',
    solver='adam', alpha=0.0001, batch_size=32,
    learning_rate_init=0.001, max_iter=500,
    random_state=42, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20
)
# If instance weights are invalid or missing, fall back to unweighted fit
if sw_aif is None:
    nn_rw.fit(X_tr, y_tr)
else:
    sw_finite = np.isfinite(sw_aif).all()
    sw_nonneg = (sw_aif >= 0).all()
    if not (sw_finite and sw_nonneg):
        nn_rw.fit(X_tr, y_tr)
    else:
        # Provide the valid sample weights to the estimator
        nn_rw.fit(X_tr, y_tr, sample_weight=sw_aif)
nn_rw_pred = nn_rw.predict(X_te)
nn_rw_prob = nn_rw.predict_proba(X_te)[:, 1]

rw_metrics = eval_model("NN Re-weighted", nn_rw_pred, g_te, ab_te)


# ---------------------- Threshold-adjusted NN ----------------------
print("\n\nComputing thresholds for NN predictions...")
# Find per-group thresholds so that each group's positive rate matches overall rate
nn_th = find_thresholds(nn_prob, g_te, OVERALL_RATE)
print(f"NN thresholds: Male(0)={nn_th[0]:.4f} Female(1)={nn_th[1]:.4f}")
nn_ta_pred = apply_thresholds(nn_prob, g_te, nn_th)

ta_metrics = eval_model("NN Threshold Adj.", nn_ta_pred, g_te, ab_te)


# ---------------------- Summary & saving results ----------------------
print("\n\n=== Summary ===")
print("Baseline performance and gender bias: ", baseline_metrics)
print("Re-weighted performance and gender bias: ", rw_metrics)
print("Threshold-adjusted performance and gender bias: ", ta_metrics)

# write two CSVs: one for performance metrics and one for fairness metrics
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

#  directory for output CSVs (same folder as this script)
out_dir = os.path.dirname(__file__)
pd.DataFrame(perf_records).to_csv(os.path.join(out_dir, 'nn_performance.csv'), index=False)
pd.DataFrame(fair_records).to_csv(os.path.join(out_dir, 'nn_fairness.csv'), index=False)
print("Saved nn_performance.csv and nn_fairness.csv in", out_dir)
