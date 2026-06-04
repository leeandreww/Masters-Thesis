
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# Determine script directory and load the CSV dataset
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'recruitment_data.csv')
df = pd.read_csv(csv_path)

# Confirm successful load and dataset shape
print("Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# Prepare features (X) and target (y)
# ---------------------------------------------------------------------------
# Drop the target column from features and keep it separately for training
X = df.drop('HiringDecision', axis=1)
y = df['HiringDecision']

# ---------------------------------------------------------------------------
# Train/Test split
# ---------------------------------------------------------------------------
# Use an 80/20 split, fixed random_state for reproducibility and stratify
# to preserve the original class balance in both training and test sets.
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # Fixed seed for reproducibility
    stratify=y           # Preserve the class balance in both splits
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# ---------------------------------------------------------------------------
# Train Random Forest model
# ---------------------------------------------------------------------------
# Use a reasonably large forest (500 trees) and class_weight='balanced'
# to account for the class imbalance in the target variable.
rf = RandomForestClassifier(
    n_estimators=500,        # Number of decision trees
    random_state=42,         # Fixed seed for reproducibility
    class_weight='balanced'  # Handle class imbalance during training
)

# Fit model on training data
rf.fit(X_train, y_train)
print("\nRandom Forest model trained successfully!")

# ---------------------------------------------------------------------------
# Feature importances (Gini importance from RandomForest)
# ---------------------------------------------------------------------------
# Extract importance values and sort descending for reporting and plotting
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

print("\n--- Feature Importances ---")
for feature, importance in importances.items():
    # Print each feature's raw importance and percentage contribution
    print(f"  {feature:<25} {importance:.6f}  ({importance*100:.2f}%)")

# ---------------------------------------------------------------------------
# Evaluate model on held-out test set
# ---------------------------------------------------------------------------
y_pred = rf.predict(X_test)
print(f"\n--- Model Performance on Test Set ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------
# Plot feature importance chart (horizontal bar chart)
# ---------------------------------------------------------------------------
# Sort importances ascending for a more natural horizontal bar layout
importances_plot = importances.sort_values(ascending=True)

# Color code bars by importance thresholds for visual emphasis
colors = []
for imp in importances_plot.values:
    if imp >= 0.20:
        colors.append('#1F3864')
    elif imp >= 0.10:
        colors.append('#2E75B6')
    elif imp >= 0.05:
        colors.append('#9DC3E6')
    else:
        colors.append('#D6E4F0')

# Optional mapping of column names to nicer labels (identity mapping here)
label_map = {
    'RecruitmentStrategy'  : 'RecruitmentStrategy',
    'InterviewScore'       : 'InterviewScore',
    'SkillScore'           : 'SkillScore',
    'PersonalityScore'     : 'PersonalityScore',
    'EducationLevel'       : 'EducationLevel',
    'ExperienceYears'      : 'ExperienceYears',
    'DistanceFromCompany'  : 'DistanceFromCompany',
    'Age'                  : 'Age',
    'PreviousCompanies'    : 'PreviousCompanies',
    'Gender'               : 'Gender'
}
labels = [label_map[f] for f in importances_plot.index]

# Create figure and axes with a clean white background
fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

# Draw horizontal bars with chosen styles
bars = ax.barh(labels, importances_plot.values, color=colors,
               edgecolor='white', linewidth=0.6, height=0.65)

# Annotate each bar with the numeric importance value for clarity
for bar, val in zip(bars, importances_plot.values):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', ha='left',
            fontsize=10, color='#222222')

# Style grid, spines and labels for a publication-ready chart
ax.grid(axis='x', linestyle='--', alpha=0.4, color='#CCCCCC', zorder=0)
for spine in ['top', 'right', 'bottom']:
    ax.spines[spine].set_visible(False)
ax.spines['left'].set_color('#AAAAAA')

ax.set_xlabel('Feature Importance Score (Gini Impurity Reduction)', fontsize=12, labelpad=10, color='#333333')
ax.set_title('Random Forest Feature Importance\nRecruitment Dataset — 500 Trees, Stratified 80/20 Split',
             fontsize=14, fontweight='bold', pad=16, color='#1F2D3D', linespacing=1.6)
ax.tick_params(axis='both', labelsize=11, colors='#333333', length=0)
ax.set_xlim(0, 0.38)

plt.tight_layout()

# Save the figure to the same directory as the script
output_path = os.path.join(script_dir, 'rf_feature_importance.png')
plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='#FFFFFF')
print(f"\nChart saved as '{output_path}' in your folder.")

# Display the figure interactively (will open a window when run locally)
plt.show()