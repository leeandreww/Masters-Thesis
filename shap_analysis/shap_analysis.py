
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
from joblib import dump, load

# Suppress non-essential warnings to keep the analysis output readable.
warnings.filterwarnings('ignore')

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

OUTPUT_DIR = 'shap_analysis'
# Create the output folder if it does not already exist.
os.makedirs(OUTPUT_DIR, exist_ok=True)
# Fix the random seed so sampling and model behavior are reproducible.
np.random.seed(42)

MCOLS = ['#2E75B6', '#375623', '#C00000']
HEADER_BG = '#1F3864'
ROW_A     = '#FFFFFF'
ROW_B     = '#EEF2F7'
YELLOW    = '#FFF2CC'   # protected attributes
GREEN     = '#E2EFDA'   # positive delta


# Load the recruitment dataset and create the binary age attribute used later.
df = pd.read_csv('recruitment_data.csv')
df['AgeBinary'] = (df['Age'] >= 35).astype(int)

# Feature names used consistently for model training and SHAP reporting.
FEATURE_COLS = [
    'Age', 'Gender', 'EducationLevel', 'ExperienceYears',
    'PreviousCompanies', 'DistanceFromCompany',
    'InterviewScore', 'SkillScore', 'PersonalityScore',
    'RecruitmentStrategy'
]

# Display labels mirror the feature columns here, but keep a separate list so
# the script can rename them later without changing the model inputs.
FEATURE_LABELS = [
    'Age', 'Gender', 'EducationLevel', 'ExperienceYears',
    'PreviousCompanies', 'DistanceFromCompany',
    'InterviewScore', 'SkillScore', 'PersonalityScore',
    'RecruitmentStrategy'
]

# Extract the model matrix, label vector, and protected attribute arrays.
X      = df[FEATURE_COLS].values
y      = df['HiringDecision'].values
gender = df['Gender'].values
age_b  = df['AgeBinary'].values

# Scale the features to [0, 1] before training and explanation.
scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test with stratification to preserve class balance.
X_tr, X_te, y_tr, y_te, g_tr, g_te, ab_tr, ab_te = train_test_split(
    X_scaled, y, gender, age_b,
    test_size=0.2, random_state=42, stratify=y
)
# Report the split sizes so the analysis can be audited quickly.
print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")


# Train or load the baseline and re-weighted models used in the SHAP comparison.

MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

def model_path(name):
    # Helper for consistent model file naming.
    return os.path.join(MODEL_DIR, f'{name}.joblib')

if all(os.path.exists(model_path(n)) for n in [
        'scaler', 'nn_base', 'rf_base', 'xgb_base',
        'nn_rw', 'rf_rw', 'xgb_rw',
]):
    # If all serialized models are present, reuse them instead of retraining.
    print('Loading pre-trained models from', MODEL_DIR)
    scaler   = load(model_path('scaler'))
    nn_base  = load(model_path('nn_base'))
    rf_base  = load(model_path('rf_base'))
    xgb_base = load(model_path('xgb_base'))
    nn_rw    = load(model_path('nn_rw'))
    rf_rw    = load(model_path('rf_rw'))
    xgb_rw   = load(model_path('xgb_rw'))
    training_required = False
    X_scaled = scaler.transform(X)
    X_tr, X_te, y_tr, y_te, g_tr, g_te, ab_tr, ab_te = train_test_split(
        X_scaled, y, gender, age_b,
        test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train: {len(X_tr)}  |  Test: {len(X_te)}")
else:
    # Otherwise, the script will train the models from scratch.
    print('No saved model set found. Training models now...')
    training_required = True

def train_and_save_models():
    # Build all model variants and save them for later SHAP computation.
    global scaler, nn_base, rf_base, xgb_base, nn_rw, rf_rw, xgb_rw
    global X_tr, X_te, y_tr, y_te, g_tr, g_te, ab_tr, ab_te

    # Import fairness tooling only when training is actually needed.
    from aif360.datasets import BinaryLabelDataset
    from aif360.algorithms.preprocessing import Reweighing

    def make_aif_dataset(X_arr, y_arr, prot_arr, prot_col):
        # Construct an AIF360 dataset so reweighing can operate on the training data.
        df_tmp = pd.DataFrame(X_arr, columns=FEATURE_COLS)
        df_tmp['HiringDecision'] = y_arr
        df_tmp[prot_col] = prot_arr
        return BinaryLabelDataset(
            df=df_tmp,
            label_names=['HiringDecision'],
            protected_attribute_names=[prot_col],
            favorable_label=1, unfavorable_label=0
        )

    # Build a Gender-based AIF360 dataset for the reweighing preprocessor.
    ds_tr_g  = make_aif_dataset(X_tr, y_tr, g_tr, 'Gender')
    RW_alg   = Reweighing(
        unprivileged_groups=[{'Gender': 1}],
        privileged_groups  =[{'Gender': 0}]
    )
    # The instance weights are used to dampen group imbalance during training.
    sw_aif = RW_alg.fit_transform(ds_tr_g).instance_weights

    # baseline model
    print("\nTraining baseline models...")
    # Neural network baseline trained on the original data.
    nn_base = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16), activation='relu',
        solver='adam', alpha=0.0001, batch_size=32,
        learning_rate_init=0.001, max_iter=500,
        random_state=42, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20
    )
    nn_base.fit(X_tr, y_tr)

    # Random forest baseline trained with class weighting.
    rf_base = RandomForestClassifier(
        n_estimators=500, random_state=42,
        class_weight='balanced', max_depth=10, min_samples_leaf=5
    )
    rf_base.fit(X_tr, y_tr)

    # Gradient boosting baseline used as the XGB-style model in this script.
    xgb_base = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, random_state=42,
        subsample=0.8, min_samples_leaf=5
    )
    xgb_base.fit(X_tr, y_tr)

    # re-weighted models
    print("Training re-weighted models...")
    # Neural network retrained with sample weights from reweighing.
    nn_rw = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16), activation='relu',
        solver='adam', alpha=0.0001, batch_size=32,
        learning_rate_init=0.001, max_iter=500,
        random_state=42, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20
    )
    nn_rw.fit(X_tr, y_tr, sample_weight=sw_aif)

    # Random forest retrained with the same fairness-derived weights.
    rf_rw = RandomForestClassifier(
        n_estimators=500, random_state=42,
        max_depth=10, min_samples_leaf=5
    )
    rf_rw.fit(X_tr, y_tr, sample_weight=sw_aif)

    # Gradient boosting retrained with fairness-derived instance weights.
    xgb_rw = GradientBoostingClassifier(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, random_state=42,
        subsample=0.8, min_samples_leaf=5
    )
    xgb_rw.fit(X_tr, y_tr, sample_weight=sw_aif)

    # Persist all trained artifacts so later runs can reuse them.
    print("All 6 models trained.")

    dump(scaler, model_path('scaler'))
    dump(nn_base, model_path('nn_base'))
    dump(rf_base, model_path('rf_base'))
    dump(xgb_base, model_path('xgb_base'))
    dump(nn_rw, model_path('nn_rw'))
    dump(rf_rw, model_path('rf_rw'))
    dump(xgb_rw, model_path('xgb_rw'))
    print('Saved all models into', MODEL_DIR)


if training_required:
    # Train only when the serialized model set is missing.
    train_and_save_models()


# Select a manageable background sample for SHAP and a smaller explanation set.
rng    = np.random.RandomState(42)
n_bg   = 100
n_exp  = 150

# Background data approximates the training distribution for explainers.
X_bg  = X_tr[rng.choice(len(X_tr), n_bg,  replace=False)]
# Explanation data is drawn from the held-out test set.
X_exp = X_te[rng.choice(len(X_te), min(n_exp, len(X_te)), replace=False)]


# Create SHAP explanations for each model family.
def get_rf_explanation(model, X_bg, X_exp, feature_names):
    # TreeExplainer via the unified SHAP API.
    """TreeExplainer via unified API — returns shap.Explanation."""
    explainer = shap.Explainer(
        model,
        X_bg,
        feature_names=feature_names,
    )
    exp = explainer(X_exp, check_additivity=False)
    if len(exp.shape) == 3:
        # Binary classifiers can emit 3D output; keep the positive class.
        exp = exp[:, :, 1]
    return exp

def get_xgb_explanation(model, X_bg, X_exp, feature_names):
    # Gradient boosting is also handled by SHAP's tree-based explainer path.
    """TreeExplainer via unified API — GBT returns 2-D directly."""
    explainer = shap.Explainer(
        model,
        X_bg,
        feature_names=feature_names,
        model_output='probability', 
    )
    return explainer(X_exp, check_additivity=False)

def get_nn_explanation(model, X_bg, X_exp, feature_names,
                       nsamples=200):
    # Neural networks use the slower KernelExplainer wrapper here.
    """KernelExplainer wrapped into shap.Explanation."""
    explainer = shap.KernelExplainer(
        lambda x: model.predict_proba(x)[:, 1], X_bg
    )
    shap_vals = explainer.shap_values(X_exp, nsamples=nsamples, silent=True)
    return shap.Explanation(
        values       = shap_vals,
        base_values  = np.full(len(X_exp), explainer.expected_value),
        data         = X_exp,
        feature_names= feature_names
    )

# Compute baseline SHAP values for the three model families.
print("\nComputing SHAP — NN baseline (KernelExplainer, ~2-5 min)...")
nn_base_exp  = get_nn_explanation(nn_base, X_bg, X_exp, FEATURE_LABELS)
print("Computing SHAP — RF baseline (TreeExplainer)...")
rf_base_exp  = get_rf_explanation(rf_base, X_bg, X_exp, FEATURE_LABELS)
print("Computing SHAP — XGB baseline (TreeExplainer)...")
xgb_base_exp = get_xgb_explanation(xgb_base, X_bg, X_exp, FEATURE_LABELS)

# Compute SHAP values for the re-weighted model variants.
print("Computing SHAP — NN re-weighted (KernelExplainer, ~2-5 min)...")
nn_rw_exp    = get_nn_explanation(nn_rw,  X_bg, X_exp, FEATURE_LABELS)
print("Computing SHAP — RF re-weighted (TreeExplainer)...")
rf_rw_exp    = get_rf_explanation(rf_rw,  X_bg, X_exp, FEATURE_LABELS)
print("Computing SHAP — XGB re-weighted (TreeExplainer)...")
xgb_rw_exp   = get_xgb_explanation(xgb_rw, X_bg, X_exp, FEATURE_LABELS)

print("All SHAP values computed.")


# Build comparative tables that summarize baseline vs re-weighted SHAP means.
PROTECTED = {'Gender', 'Age'}

def build_shap_table(base_exp, rw_exp, model_name, table_num, csv_path):
    # Compute mean absolute SHAP values for both variants.
    """Save one SHAP comparison table as CSV."""
    base_means = np.abs(base_exp.values).mean(axis=0)  
    rw_means   = np.abs(rw_exp.values  ).mean(axis=0)

    # Rank features by the baseline ordering so both columns align visually.
    base_rank_order = np.argsort(base_means)[::-1]   
    base_ranks = np.empty(len(base_means), dtype=int)
    base_ranks[base_rank_order] = np.arange(1, len(base_means)+1)

    rw_rank_order = np.argsort(rw_means)[::-1]
    rw_ranks = np.empty(len(rw_means), dtype=int)
    rw_ranks[rw_rank_order] = np.arange(1, len(rw_means)+1)

    # Use the baseline feature order for the output table.
    display_order = base_rank_order   
    rows = []
    for rank, fi in enumerate(display_order, start=1):
        fname = FEATURE_LABELS[fi]
        bv    = float(base_means[fi])
        rv    = float(rw_means[fi])
        delta = rv - bv
        rows.append({
            'Rank'             : rank,
            'Feature'          : fname,
            'Baseline Mean|SHAP|': round(bv, 4),
            'Re-weighted Mean|SHAP|': round(rv, 4),
            'Δ Change'         : round(delta, 4),
            'Rank (Base)'      : int(base_ranks[fi]),
            'Rank (Re-wt)'     : int(rw_ranks[fi]),
            'Protected'        : fname in PROTECTED,
        })
    df_table = pd.DataFrame(rows)

    # Save without the helper column used only for internal highlighting.
    df_table.drop(columns='Protected').to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    return df_table

# Generate one table per model family for the report.
print("\n" + "="*55)
print("  Generating SHAP tables...")
print("="*55)

t_nn  = build_shap_table(nn_base_exp,  nn_rw_exp,  'NN',  1,
                          os.path.join(OUTPUT_DIR, 'table_nn_shap.csv'))
t_rf  = build_shap_table(rf_base_exp,  rf_rw_exp,  'RF',  2,
                          os.path.join(OUTPUT_DIR, 'table_rf_shap.csv'))
t_xgb = build_shap_table(xgb_base_exp, xgb_rw_exp, 'XGB', 3,
                          os.path.join(OUTPUT_DIR, 'table_xgb_shap.csv'))


# Create beeswarm plots for the baseline models.
print("\n" + "="*55)
print("  Generating beeswarm plots...")
print("="*55)

beeswarm_configs = [
    (nn_base_exp,  'NN',  os.path.join(OUTPUT_DIR, 'beeswarm_nn.png')),
    (rf_base_exp,  'RF',  os.path.join(OUTPUT_DIR, 'beeswarm_rf.png')),
    (xgb_base_exp, 'XGB', os.path.join(OUTPUT_DIR, 'beeswarm_xgb.png')),
]

for shap_exp, mname, filepath in beeswarm_configs:
    print(f"  Generating beeswarm plot — {mname}...")

    # Start each plot on a fresh figure.
    plt.figure(figsize=(10, 6.5))

    # Beeswarm shows the distribution of SHAP values per feature.
    shap.plots.beeswarm(
        shap_exp,
        max_display = 10,                   
        color_bar   = True,                  
        show        = False,               
        plot_size   = None,                  
    )
    # ─────────────────────────────────────────────────────────

    ax = plt.gca()
    # Add a title and a zero reference line for readability.
    ax.set_title(
        f'SHAP Beeswarm — {mname} Baseline\n'
        'x-axis: SHAP value  (← Not Hired  |  Hired →) ',
        fontsize=10, fontweight='bold', color='#1F2D3D', pad=12, loc='left'
    )
    ax.axvline(0, color='#555555', linewidth=0.8, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(filepath, dpi=160, bbox_inches='tight', facecolor='#FFFFFF')
    plt.close()
    print(f"    Saved: {filepath}")


# Build a cross-model bar chart to compare mean absolute SHAP values.
print("\nGenerating cross-model bar chart (Figure 4.9.4)...")
 
means = {
    'NN' : np.abs(nn_base_exp.values).mean(axis=0),
    'RF' : np.abs(rf_base_exp.values).mean(axis=0),
    'XGB': np.abs(xgb_base_exp.values).mean(axis=0),
}
 
# Use the NN ordering as the shared feature order for all three models.
order_idx  = np.argsort(means['NN'])[::-1]   
top_labels = [FEATURE_LABELS[i] for i in order_idx]
 
fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FAFAFA')
 
x = np.arange(len(order_idx))
w = 0.25
 
for i, (mname, col) in enumerate(zip(['NN', 'RF', 'XGB'], MCOLS)):
    vals = means[mname][order_idx]
    # Offset each model's bars so the three models can be compared side by side.
    bars = ax.bar(x + (i - 1) * w, vals, width=w, color=col,
                  edgecolor='white', linewidth=0.7, label=mname)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                v + 0.001, f'{v:.4f}', ha='center', va='bottom',
                fontsize=7, fontweight='bold', color=col)
 
ax.set_xticks(x)
ax.set_xticklabels(top_labels, fontsize=9, rotation=0, ha='center', va='top')
# Stagger x-axis labels slightly to reduce overlap for longer feature names.
for i, tick in enumerate(ax.get_xticklabels()):
    dy = -0.03 if i % 2 == 0 else -0.06
    tick.set_y(dy)
ax.tick_params(axis='x', pad=0)
ax.set_ylabel('Mean |SHAP Value|', fontsize=12)
ax.set_ylim(0, max(m.max() for m in means.values()) * 1.35)
plt.subplots_adjust(bottom=0.20)

ax.set_title(
    'Cross-Model SHAP Comparison — '
    'Mean |SHAP Values| (NN, RF, XGB Baselines)\n'
    'Features sorted by NN mean |SHAP| (highest → lowest)',
    fontsize=12, fontweight='bold', color='#1F2D3D', pad=12
)
ax.legend(fontsize=11, framealpha=0.95)
for sp in ['top', 'right']:
    ax.spines[sp].set_visible(False)
ax.grid(axis='y', linestyle='--', alpha=0.4, color='#CCCCCC')
ax.tick_params(length=0, labelsize=10.5)
 
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'crossmodel_comparison_shap.png'),
            dpi=160, bbox_inches='tight', facecolor='#FFFFFF')
plt.close()
print('  Saved: ' + os.path.join(OUTPUT_DIR, 'crossmodel_comparison_shap.png'))


# Print a compact summary of the main SHAP magnitudes for the report.
print("\n\n" + "="*60)
print("  SECTION 4.9 — MEAN |SHAP| SUMMARY")
print("="*60)
means = {
    'NN' : np.abs(nn_base_exp.values ).mean(axis=0),
    'RF' : np.abs(rf_base_exp.values ).mean(axis=0),
    'XGB': np.abs(xgb_base_exp.values).mean(axis=0),
}
order_idx = np.argsort(means['NN'])[::-1]
print(f"\n  {'Feature':<26} {'NN':>8} {'RF':>8} {'XGB':>8}")
print("  " + "─"*54)
for i in order_idx:
    # Mark protected attributes directly in the text summary.
    mark = " ◄" if FEATURE_LABELS[i] in PROTECTED else "  "
    print(f"  {FEATURE_LABELS[i]:<26}{mark}"
          f"{means['NN'][i]:>8.4f} "
          f"{means['RF'][i]:>8.4f} "
          f"{means['XGB'][i]:>8.4f}")
print("\n  ◄ = protected attribute")
print("\n  Output files:")
files = [
    os.path.join(OUTPUT_DIR, 'table_nn_shap.csv'),
    os.path.join(OUTPUT_DIR, 'table_rf_shap.csv'),
    os.path.join(OUTPUT_DIR, 'table_xgb_shap.csv'),
    os.path.join(OUTPUT_DIR, 'beeswarm_nn.png'),
    os.path.join(OUTPUT_DIR, 'beeswarm_rf.png'),
    os.path.join(OUTPUT_DIR, 'beeswarm_xgb.png'),
    os.path.join(OUTPUT_DIR, 'crossmodel_comparison_shap.png'),
]
for f in files:
    print(f"    {f}")
print("\n" + "="*60)