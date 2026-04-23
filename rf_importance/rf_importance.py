
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# load dataset
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'recruitment_data.csv')
df = pd.read_csv(csv_path)

print("Dataset loaded successfully!")
print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")

# split into features (X) and target (y) ---
X = df.drop('HiringDecision', axis=1)
y = df['HiringDecision']

# split into training set (80%) and test set (20%) ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% for testing
    random_state=42,     # Fixed seed for reproducibility
    stratify=y           # Preserve the 69:31 class balance in both splits
)

print(f"\nTraining samples : {len(X_train)}")
print(f"Testing samples  : {len(X_test)}")

# train the Random Forest model 
rf = RandomForestClassifier(
    n_estimators=500,        # Number of decision trees
    random_state=42,         # Fixed seed for reproducibility
    class_weight='balanced'  # Handles the 69:31 class imbalance
)

rf.fit(X_train, y_train)
print("\nRandom Forest model trained successfully!")

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False) 

print("\n--- Feature Importances ---")
for feature, importance in importances.items():
    print(f"  {feature:<25} {importance:.6f}  ({importance*100:.2f}%)")

y_pred = rf.predict(X_test)
print(f"\n--- Model Performance on Test Set ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print(classification_report(y_test, y_pred))

# plot the feature importance chart 

importances_plot = importances.sort_values(ascending=True)

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

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('#FFFFFF')
ax.set_facecolor('#FFFFFF')

bars = ax.barh(labels, importances_plot.values, color=colors,
               edgecolor='white', linewidth=0.6, height=0.65)

for bar, val in zip(bars, importances_plot.values):
    ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}', va='center', ha='left',
            fontsize=10, color='#222222')

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

# save the chart
output_path = os.path.join(script_dir, 'rf_feature_importance.png')
plt.savefig(output_path, dpi=180, bbox_inches='tight', facecolor='#FFFFFF')
print(f"\nChart saved as '{output_path}' in your folder.")

plt.show()