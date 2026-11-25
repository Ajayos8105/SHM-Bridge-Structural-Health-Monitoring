"""
Structural Health Monitoring – Vibration-Based Damage Classifier
=================================================================
Dataset: Yonghe Bridge Modal Frequencies (FDD)
  - 4 modal frequencies × 216 samples
  - Samples 1–192: Healthy (undamaged)
  - Samples 193–216: Damaged

This script:
  1. Loads and explores the data
  2. Engineers features from the 4 modal frequencies
  3. Trains & compares multiple classifiers (RF, SVM, XGBoost, kNN, Logistic Regression)
  4. Evaluates with cross-validation and a held-out test set
  5. Explains feature importance (permutation importance + SHAP-style bar charts)
  6. Saves all plots to disk
"""

import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, accuracy_score, f1_score,
                             roc_curve, auc)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier

import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# ──────────────────────────────────────────────
# 1. Load Data
# ──────────────────────────────────────────────
mat = sio.loadmat(str(DATA_DIR / 'Yonghe_Modal_FDD.mat'))
freq_fdd = mat['freq_fdd']  # shape: (4, 216) — 4 modes × 216 samples

print(f"Raw data shape: {freq_fdd.shape}")
print(f"  Mode frequencies (rows) × Samples (columns)")

n_healthy = 192
n_damaged = freq_fdd.shape[1] - n_healthy  # 24

# Transpose to (samples, features)
X_raw = freq_fdd.T  # (216, 4)
y = np.array([0]*n_healthy + [1]*n_damaged)  # 0=healthy, 1=damaged

mode_names = ['Mode 1 (f₁)', 'Mode 2 (f₂)', 'Mode 3 (f₃)', 'Mode 4 (f₄)']

df = pd.DataFrame(X_raw, columns=mode_names)
df['Label'] = ['Healthy' if lbl == 0 else 'Damaged' for lbl in y]

print(f"\nClass distribution: Healthy={n_healthy}, Damaged={n_damaged}")
print(f"\nDescriptive statistics (Healthy):")
print(df[df.Label == 'Healthy'][mode_names].describe().round(5))
print(f"\nDescriptive statistics (Damaged):")
print(df[df.Label == 'Damaged'][mode_names].describe().round(5))

# ──────────────────────────────────────────────
# 2. EDA Plots
# ──────────────────────────────────────────────
# 2a. Time-series of each mode
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for i, ax in enumerate(axes.flat):
    ax.scatter(range(n_healthy), freq_fdd[i, :n_healthy],
               s=15, label='Healthy', color='steelblue', alpha=0.7)
    ax.scatter(range(n_healthy, freq_fdd.shape[1]),
               freq_fdd[i, n_healthy:],
               s=30, marker='x', label='Damaged', color='crimson')
    ax.axvline(n_healthy - 0.5, color='k', ls='--', lw=0.8)
    ax.set_xlabel('Sample index')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(mode_names[i])
    ax.legend(fontsize=8)
plt.suptitle('Modal Frequencies – Yonghe Bridge', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig(OUT_DIR / 'plot1_modal_frequencies.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved plot1_modal_frequencies.png")

# 2b. Box plots by class
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, ax in enumerate(axes):
    sns.boxplot(data=df, x='Label', y=mode_names[i], ax=ax,
                palette={'Healthy': 'steelblue', 'Damaged': 'crimson'})
    ax.set_title(mode_names[i])
plt.suptitle('Distribution by Structural Condition', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(OUT_DIR / 'plot2_boxplots.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot2_boxplots.png")

# 2c. Pairplot
g = sns.pairplot(df, hue='Label', palette={'Healthy': 'steelblue', 'Damaged': 'crimson'},
                 plot_kws={'alpha': 0.6, 's': 20})
g.figure.suptitle('Pair Plot of Modal Frequencies', y=1.01)
g.savefig(OUT_DIR / 'plot3_pairplot.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot3_pairplot.png")

# 2d. Correlation heatmap
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(df[mode_names].corr(), annot=True, cmap='coolwarm', center=0,
            fmt='.3f', ax=ax)
ax.set_title('Correlation Between Modal Frequencies')
plt.tight_layout()
plt.savefig(OUT_DIR / 'plot4_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot4_correlation.png")

# ──────────────────────────────────────────────
# 3. Feature Engineering
# ──────────────────────────────────────────────
# Start with raw modal frequencies, then add engineered features
features = {}
for i in range(4):
    features[f'f{i+1}'] = X_raw[:, i]

# Pairwise ratios (capture relative stiffness changes)
for i in range(4):
    for j in range(i+1, 4):
        features[f'f{i+1}/f{j+1}'] = X_raw[:, i] / X_raw[:, j]

# Pairwise differences
for i in range(4):
    for j in range(i+1, 4):
        features[f'f{j+1}-f{i+1}'] = X_raw[:, j] - X_raw[:, i]

# Summary statistics across modes (per sample)
features['mean_freq'] = X_raw.mean(axis=1)
features['std_freq']  = X_raw.std(axis=1)
features['range_freq'] = X_raw.max(axis=1) - X_raw.min(axis=1)
features['cv_freq']   = X_raw.std(axis=1) / X_raw.mean(axis=1)  # coeff of variation

X_feat = pd.DataFrame(features)
feature_names = list(X_feat.columns)
print(f"\nEngineered {len(feature_names)} features:")
print(feature_names)

# ──────────────────────────────────────────────
# 4. Train/Test Split + Scaling
# ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X_feat, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain: {X_train_sc.shape[0]} samples | Test: {X_test_sc.shape[0]} samples")
print(f"Train class balance: Healthy={sum(y_train==0)}, Damaged={sum(y_train==1)}")
print(f"Test  class balance: Healthy={sum(y_test==0)},  Damaged={sum(y_test==1)}")

# ──────────────────────────────────────────────
# 5. Model Training & Comparison
# ──────────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=5000, random_state=42,
                                               class_weight='balanced'),
    'k-NN (k=5)':          KNeighborsClassifier(n_neighbors=5),
    'SVM (RBF)':           SVC(kernel='rbf', probability=True, random_state=42,
                               class_weight='balanced'),
    'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42,
                                                   class_weight='balanced'),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=200, use_label_encoder=False,
                                          eval_metric='logloss', random_state=42,
                                          scale_pos_weight=n_healthy/n_damaged),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("\n" + "="*70)
print("MODEL COMPARISON (5-Fold Stratified CV on Training Set)")
print("="*70)

for name, model in models.items():
    scores = cross_val_score(model, X_train_sc, y_train, cv=cv, scoring='f1')
    results[name] = {
        'cv_f1_mean': scores.mean(),
        'cv_f1_std':  scores.std(),
        'model': model,
    }
    print(f"  {name:25s}  F1 = {scores.mean():.4f} ± {scores.std():.4f}")

# Bar plot of CV scores
fig, ax = plt.subplots(figsize=(10, 5))
names = list(results.keys())
means = [results[n]['cv_f1_mean'] for n in names]
stds  = [results[n]['cv_f1_std'] for n in names]
bars = ax.barh(names, means, xerr=stds, color='steelblue', edgecolor='navy', alpha=0.85)
ax.set_xlabel('F1 Score (5-Fold CV)')
ax.set_title('Classifier Comparison – Cross-Validated F1')
ax.set_xlim(0, 1.1)
for bar, m in zip(bars, means):
    ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
            f'{m:.3f}', va='center', fontsize=10)
plt.tight_layout()
plt.savefig(OUT_DIR / 'plot5_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Saved plot5_model_comparison.png")

# ──────────────────────────────────────────────
# 6. Best Model – Final Evaluation on Test Set
# ──────────────────────────────────────────────
best_name = max(results, key=lambda k: results[k]['cv_f1_mean'])
print(f"\n★ Best model by CV F1: {best_name}")

best_model = results[best_name]['model']
best_model.fit(X_train_sc, y_train)
y_pred = best_model.predict(X_test_sc)
y_proba = best_model.predict_proba(X_test_sc)[:, 1] if hasattr(best_model, 'predict_proba') else None

print(f"\n--- Test Set Results ({best_name}) ---")
print(classification_report(y_test, y_pred, target_names=['Healthy', 'Damaged']))

test_acc = accuracy_score(y_test, y_pred)
test_f1  = f1_score(y_test, y_pred)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")

# Confusion matrix
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Healthy', 'Damaged'])
disp.plot(ax=ax, cmap='Blues')
ax.set_title(f'Confusion Matrix – {best_name}\nAccuracy={test_acc:.2%}  F1={test_f1:.3f}')
plt.tight_layout()
plt.savefig(OUT_DIR / 'plot6_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot6_confusion_matrix.png")

# ROC curve
if y_proba is not None:
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, color='steelblue', label=f'{best_name} (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve – Test Set')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / 'plot7_roc_curve.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved plot7_roc_curve.png")

# ──────────────────────────────────────────────
# 7. Also train & evaluate ALL models on test set
# ──────────────────────────────────────────────
print("\n" + "="*70)
print("ALL MODELS – TEST SET PERFORMANCE")
print("="*70)
all_test_results = []
for name, info in results.items():
    m = info['model']
    m.fit(X_train_sc, y_train)
    yp = m.predict(X_test_sc)
    all_test_results.append({
        'Model': name,
        'CV F1 (mean)': info['cv_f1_mean'],
        'CV F1 (std)': info['cv_f1_std'],
        'Test Accuracy': accuracy_score(y_test, yp),
        'Test F1': f1_score(y_test, yp),
    })
test_df = pd.DataFrame(all_test_results).sort_values('Test F1', ascending=False)
print(test_df.to_string(index=False))

# ──────────────────────────────────────────────
# 8. Feature Importance Analysis
# ──────────────────────────────────────────────
print("\n" + "="*70)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*70)

# 8a. Random Forest built-in importance (Gini)
rf = models['Random Forest']
rf.fit(X_train_sc, y_train)
rf_imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 8))
rf_imp.plot.barh(ax=ax, color='steelblue', edgecolor='navy', alpha=0.85)
ax.set_xlabel('Gini Importance')
ax.set_title('Random Forest – Feature Importance (Gini)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'plot8_rf_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot8_rf_feature_importance.png")

# 8b. Permutation importance on the best model
perm_imp = permutation_importance(best_model, X_test_sc, y_test,
                                   n_repeats=30, random_state=42,
                                   scoring='f1')
perm_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance Mean': perm_imp.importances_mean,
    'Importance Std': perm_imp.importances_std,
}).sort_values('Importance Mean', ascending=True)

fig, ax = plt.subplots(figsize=(9, 8))
ax.barh(perm_df['Feature'], perm_df['Importance Mean'],
        xerr=perm_df['Importance Std'],
        color='darkorange', edgecolor='saddlebrown', alpha=0.85)
ax.set_xlabel('Mean F1 Decrease')
ax.set_title(f'Permutation Importance – {best_name} (Test Set)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'plot9_permutation_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot9_permutation_importance.png")

# Print top features
print(f"\nTop 10 Features by Permutation Importance ({best_name}):")
top_perm = perm_df.sort_values('Importance Mean', ascending=False).head(10)
for i, row in top_perm.iterrows():
    print(f"  {row['Feature']:20s}  {row['Importance Mean']:.4f} ± {row['Importance Std']:.4f}")

# 8c. XGBoost feature importance
xgb = models['XGBoost']
xgb.fit(X_train_sc, y_train)
xgb_imp = pd.Series(xgb.feature_importances_, index=feature_names).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 8))
xgb_imp.plot.barh(ax=ax, color='forestgreen', edgecolor='darkgreen', alpha=0.85)
ax.set_xlabel('XGBoost Importance (Gain)')
ax.set_title('XGBoost – Feature Importance')
plt.tight_layout()
plt.savefig(OUT_DIR / 'plot10_xgb_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot10_xgb_feature_importance.png")

# 8d. Logistic Regression coefficients (interpretability)
lr = models['Logistic Regression']
lr.fit(X_train_sc, y_train)
lr_coefs = pd.Series(lr.coef_[0], index=feature_names).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(9, 8))
colors = ['crimson' if c < 0 else 'steelblue' for c in lr_coefs]
lr_coefs.plot.barh(ax=ax, color=colors, edgecolor='navy', alpha=0.8)
ax.set_xlabel('Coefficient (Standardized)')
ax.set_title('Logistic Regression – Feature Coefficients\n(+ve → Damaged, −ve → Healthy)')
ax.axvline(0, color='k', lw=0.8)
plt.tight_layout()
plt.savefig(OUT_DIR / 'plot11_lr_coefficients.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot11_lr_coefficients.png")

# ──────────────────────────────────────────────
# 9. Summary Combined Feature Importance Plot
# ──────────────────────────────────────────────
# Normalize all importances to [0,1] and average across methods
from sklearn.preprocessing import MinMaxScaler

imp_combined = pd.DataFrame({
    'RF Gini': rf_imp,
    'Permutation': perm_df.set_index('Feature')['Importance Mean'],
    'XGBoost': xgb_imp,
    'LR |coef|': lr_coefs.abs(),
})
# Normalize each column
for col in imp_combined.columns:
    mn, mx = imp_combined[col].min(), imp_combined[col].max()
    if mx > mn:
        imp_combined[col] = (imp_combined[col] - mn) / (mx - mn)
    else:
        imp_combined[col] = 0

imp_combined['Avg Importance'] = imp_combined.mean(axis=1)
imp_combined = imp_combined.sort_values('Avg Importance', ascending=True)

fig, ax = plt.subplots(figsize=(10, 9))
imp_combined['Avg Importance'].plot.barh(ax=ax, color='mediumpurple',
                                          edgecolor='indigo', alpha=0.85)
ax.set_xlabel('Averaged Normalized Importance')
ax.set_title('Combined Feature Importance (RF + Permutation + XGBoost + LR)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'plot12_combined_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved plot12_combined_importance.png")

print("\nTop 10 Features (Combined Importance):")
top_combined = imp_combined['Avg Importance'].sort_values(ascending=False).head(10)
for feat, val in top_combined.items():
    print(f"  {feat:20s}  {val:.4f}")

print("\n" + "="*70)
print("ALL PLOTS AND ANALYSIS COMPLETE")
print("="*70)
