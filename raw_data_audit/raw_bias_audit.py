
import os
import pandas as pd

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric

# Resolve the script directory so file paths work regardless of where the script is launched from.
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load the recruitment dataset from the same folder as this script.
csv_input = os.path.join(script_dir, 'recruitment_data.csv')
if not os.path.exists(csv_input):
    raise FileNotFoundError(f"Data file not found: {csv_input}")
df = pd.read_csv(csv_input)

# Create a binary age indicator used for the fairness audit.
# 0 = Young (20–34) = privileged
# 1 = Old   (35+)   = unprivileged
df['AgeBinary'] = (df['Age'] >= 35).astype(int)

# Print a short dataset summary so the audit output is easy to interpret.
print("=" * 60)
print("  Stage 1: Raw Dataset Bias Audit")
print("=" * 60)
print(f"\n  Dataset size : {len(df):,} records")
print(f"  Features     : {df.shape[1] - 2} (excl. target & AgeBinary)")
print(f"\n  Gender — Male (0): {(df['Gender']==0).sum():,} | "
      f"Female (1): {(df['Gender']==1).sum():,}")
print(f"  Age    — Young (0, 20–34): {(df['AgeBinary']==0).sum():,} | "
      f"Old (1, 35+): {(df['AgeBinary']==1).sum():,}")

# Feature columns used to construct the AIF360 dataset.
# The target column and AgeBinary are excluded from this list.
feature_cols = [
    'Age', 'Gender', 'EducationLevel', 'ExperienceYears',
    'PreviousCompanies', 'DistanceFromCompany',
    'InterviewScore', 'SkillScore', 'PersonalityScore',
    'RecruitmentStrategy'
]

def make_binary_label_dataset(dataframe, protected_col, feature_cols):
    """
    Build an AIF360 BinaryLabelDataset from a pandas DataFrame.
    The protected_col must already be binary (0/1) and present
    in the dataframe. The label is always 'HiringDecision'.
    """
    # Collect the feature, protected attribute, and target columns expected by AIF360.
    cols_needed = feature_cols + [protected_col, 'HiringDecision']
    # Deduplicate in case protected_col is already in feature_cols
    cols_needed = list(dict.fromkeys(cols_needed))
    df_aif = dataframe[cols_needed].copy().reset_index(drop=True)

    # Wrap the dataframe in an AIF360 dataset so fairness metrics can be computed.
    dataset = BinaryLabelDataset(
        df=df_aif,
        label_names=['HiringDecision'],
        protected_attribute_names=[protected_col],
        favorable_label=1,
        unfavorable_label=0
    )
    return dataset

# Compute the fairness metrics for each protected attribute and store them in a table.
results = []

# Define the two bias audits: gender and age.
configs = [
    {
        'label'         : 'Gender',
        'description'   : 'Male (priv.) vs Female (unpriv.)',
        'protected_col' : 'Gender',
        'privileged'    : [{'Gender': 0}],   # Male = 0
        'unprivileged'  : [{'Gender': 1}],   # Female = 1
    },
    {
        'label'         : 'Age',
        'description'   : 'Young 20–34 (priv.) vs Old 35+ (unpriv.)',
        'protected_col' : 'AgeBinary',
        'privileged'    : [{'AgeBinary': 0}],  # Young = 0
        'unprivileged'  : [{'AgeBinary': 1}],  # Old   = 1
    },
]

for cfg in configs:
    # Build the AIF360 dataset for the current protected attribute.
    ds = make_binary_label_dataset(df, cfg['protected_col'], feature_cols)

    # Compute statistical parity difference and disparate impact on the raw labels.
    metric = BinaryLabelDatasetMetric(
        dataset=ds,
        unprivileged_groups=cfg['unprivileged'],
        privileged_groups=cfg['privileged']
    )

    spd = metric.statistical_parity_difference()
    di  = metric.disparate_impact()

    # Compare the actual hiring rates between the privileged and unprivileged groups.
    priv_mask   = df[cfg['protected_col']] == 0
    unpriv_mask = df[cfg['protected_col']] == 1
    priv_rate   = df.loc[priv_mask,   'HiringDecision'].mean()
    unpriv_rate = df.loc[unpriv_mask, 'HiringDecision'].mean()

    # For the raw dataset audit, use the labels themselves as both ground truth and predictions.
    ds_pred = ds.copy()   
    clf_metric = ClassificationMetric(
        dataset=ds,
        classified_dataset=ds_pred,
        unprivileged_groups=cfg['unprivileged'],
        privileged_groups=cfg['privileged']
    )
    eod = clf_metric.equal_opportunity_difference()
    aod = clf_metric.average_odds_difference()

    # Collect all metrics for tabular reporting and CSV export.
    row = {
        'Protected Attribute' : cfg['label'],
        'Groups'              : cfg['description'],
        'Priv. Hire Rate'     : round(priv_rate, 4),
        'Unpriv. Hire Rate'   : round(unpriv_rate, 4),
        'SPD'                 : round(spd, 4),
        'DI'                  : round(di, 4),
        'EOD'                 : round(eod, 4),
        'AOD'                 : round(aod, 4),
        'n'                   : len(df),
    }
    results.append(row)

    # Print a compact per-attribute summary for the console report.
    print(f"\n  [{cfg['label']}]  {cfg['description']}")
    print(f"    Priv. hire rate   : {priv_rate:.4f}")
    print(f"    Unpriv. hire rate : {unpriv_rate:.4f}")
    print(f"    SPD = {spd:.4f}  |  DI = {di:.4f}  |  "
          f"EOD = {eod:.4f}  |  AOD = {aod:.4f}")

# Convert the collected rows into a DataFrame for display and export.
results_df = pd.DataFrame(results)

print("\n" + "=" * 60)
print("  SUMMARY TABLE")
print("=" * 60)
print(results_df.to_string(index=False))

# Save the audit results to disk next to the script.
csv_path = os.path.join(script_dir, 'raw_bias_audit.csv')
results_df.to_csv(csv_path, index=False)
print(f"\n  CSV saved → {csv_path}")

# Print short interpretation notes to explain how to read the metrics.
print("\n" + "=" * 60)
print("  INTERPRETATION")
print("=" * 60)
for row in results:
    attr = row['Protected Attribute']
    spd  = row['SPD']
    di   = row['DI']
    priv = row['Priv. Hire Rate']
    unpr = row['Unpriv. Hire Rate']

    di_status = "✅ above 0.80 threshold" if di >= 0.80 else "❌ below 0.80 threshold"
    spd_status = "✅ near-zero" if abs(spd) < 0.02 else "⚠️  non-trivial"

    # Report the group rates alongside the fairness interpretation for each attribute.
    print(f"\n  {attr}:")
    print(f"    Privileged hire rate   : {priv:.4f}")
    print(f"    Unprivileged hire rate : {unpr:.4f}")
    print(f"    SPD = {spd:.4f}  → {spd_status}")
    print(f"    DI  = {di:.4f}  → {di_status}")
    print(f"    EOD = {row['EOD']:.4f}  → 0.0 (ground-truth audit; no prediction error)")
    print(f"    AOD = {row['AOD']:.4f}  → 0.0 (ground-truth audit; no prediction error)")

# Summarize the important caveat for raw-dataset auditing.
print("\n  Note: EOD and AOD are structurally zero in a raw dataset audit")
print("  because the ground truth labels carry no prediction error.")
print("  These metrics become meaningful in Sections 4.7 and 4.8")
print("  when evaluated against trained model predictions.")
print("\n" + "=" * 60)
print("  Stage 1 complete.")
print("=" * 60)