#!/usr/bin/env python
# coding: utf-8

# # Assignment 2 - Supervised Learning Methods
# ## Task 2: Preprocessing and Feature Analysis
# **Dataset:** Breast Cancer Wisconsin (Diagnostic)
# **Goal:** Stratified train/test split, StandardScaler (fit on training only),
# near-zero variance analysis, high-correlation analysis, and feature selection decision.

# ---
# ### 2.1 Imports and Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
import os
from collections import Counter

warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Output directories
output_dir = '/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/python_code'
img_dir = os.path.join(output_dir, 'images')
os.makedirs(img_dir, exist_ok=True)

# Reproducibility
np.random.seed(42)

# Colour palette
PALETTE     = ['#4A90D9', '#E8603C', '#2EAF7D', '#9B59B6', '#F39C12', '#C0392B', '#1ABC9C']
CLR_BEN     = '#4A90D9'
CLR_MAL     = '#E8603C'

sns.set_theme(
    style='whitegrid',
    palette=PALETTE,
    font='DejaVu Sans',
    font_scale=1.1,
    rc={
        'figure.dpi'        : 120,
        'figure.figsize'    : (12, 5),
        'axes.spines.top'   : False,
        'axes.spines.right' : False,
        'axes.titleweight'  : 'bold',
        'axes.titlesize'    : 13,
        'axes.labelsize'    : 11,
        'xtick.labelsize'   : 9,
        'ytick.labelsize'   : 9,
        'legend.frameon'    : False,
        'legend.fontsize'   : 9,
    }
)
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=PALETTE)
print('Libraries loaded successfully.')


# ---
# ### 2.2 Load Dataset

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')   # 0 = malignant, 1 = benign

print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")


# ---
# ### 2.3 Stratified Train-Test Split
#
# The dataset is partitioned into 80% training (n = 455) and 20% test (n = 114).
# stratify=y ensures class proportions are preserved in both subsets.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = 42,
    stratify     = y
)

print(f"\n{'Split':<12} {'n':>6} {'Malignant':>12} {'Benign':>8} {'Mal%':>7} {'Ben%':>7}")
for name, X_s, y_s in [('Training', X_train, y_train),
                        ('Test',     X_test,  y_test),
                        ('Full',     X,       y)]:
    n_mal = (y_s == 0).sum()
    n_ben = (y_s == 1).sum()
    print(f"{name:<12} {len(y_s):>6} {n_mal:>12} {n_ben:>8} "
          f"{n_mal / len(y_s) * 100:>6.1f}% {n_ben / len(y_s) * 100:>6.1f}%")

print(f"\nX_train : {X_train.shape}  |  X_test : {X_test.shape}")
print(f"y_train : {y_train.shape}   |  y_test : {y_test.shape}")

# Class balance plot across splits
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

splits = [
    ('Full Dataset\n(n = 569)',  y,       ),
    ('Training Set\n(n = 455)',  y_train, ),
    ('Test Set\n(n = 114)',      y_test,  ),
]

for ax, (title, y_s) in zip(axes, splits):
    cnts = y_s.value_counts().sort_index()
    bars = ax.bar(
        ['Malignant', 'Benign'], cnts.values,
        color=[CLR_MAL, CLR_BEN], width=0.48,
        edgecolor='white', linewidth=1.8, alpha=0.90
    )
    for bar, cnt in zip(bars, cnts.values):
        pct = cnt / len(y_s) * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.5,
            f'{cnt}\n({pct:.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold'
        )
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Samples', fontsize=10)
    ax.set_ylim(0, max(cnts.values) * 1.30)
    ax.tick_params(labelsize=10)
    ax.set_facecolor('#fafafa')

fig.suptitle('Class Balance: Full, Training, and Test Sets',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task2_01_split_balance.png'), dpi=300, bbox_inches='tight')
plt.close()


# ---
# ### 2.4 Feature Scaling with StandardScaler
#
# StandardScaler transforms each feature to zero mean and unit variance
# using ONLY training-set statistics. The test set is transformed
# (not fitted) to prevent data leakage.

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # no .fit() here

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X.columns, index=X_test.index)

# Verification
train_means = X_train_scaled.mean().abs()
train_stds  = X_train_scaled.std()
test_means  = X_test_scaled.mean().abs()

print("\nPost-scaling verification:")
print(f"  Training set max |mean|  : {train_means.max():.2e}  (expected ≈ 0)")
print(f"  Training set mean std    : {train_stds.mean():.6f}  (expected ≈ 1)")
print(f"  Training set min std     : {train_stds.min():.6f}")
print(f"  Training set max std     : {train_stds.max():.6f}")
print(f"  Test set max |mean|      : {test_means.max():.4f}  (expected != 0, leakage check)")

# Before vs After Scaling on 6 representative features
sample_feats = [
    'mean radius', 'mean area', 'mean concavity',
    'worst perimeter', 'area error', 'mean fractal dimension'
]
display_names = [
    'Mean Radius', 'Mean Area', 'Mean Concavity',
    'Worst Perimeter', 'Area Error', 'Mean Fractal Dim'
]

fig, axes = plt.subplots(2, 6, figsize=(22, 8))

for col, (feat, dname) in enumerate(zip(sample_feats, display_names)):
    clr = PALETTE[col % len(PALETTE)]

    ax_top = axes[0, col]
    ax_top.hist(X_train[feat], bins=32, color=clr,
                alpha=0.80, edgecolor='white', linewidth=0.5)
    ax_top.set_title(dname, fontsize=9, fontweight='bold')
    if col == 0:
        ax_top.set_ylabel('Before Scaling\n\nCount', fontsize=9, labelpad=4)
    ax_top.set_xlabel('Raw value', fontsize=8)
    ax_top.set_facecolor('#fafafa')
    ax_top.tick_params(labelsize=7.5)

    ax_bot = axes[1, col]
    ax_bot.hist(X_train_scaled[feat], bins=32, color=clr,
                alpha=0.80, edgecolor='white', linewidth=0.5)
    ax_bot.axvline(0, color='crimson', linestyle='--',
                   linewidth=1.8, alpha=0.85, label='μ = 0')
    if col == 0:
        ax_bot.set_ylabel('After Scaling\n\nCount', fontsize=9, labelpad=4)
        ax_bot.legend(fontsize=8)
    ax_bot.set_xlabel('z-score', fontsize=8)
    ax_bot.set_facecolor('#fafafa')
    ax_bot.tick_params(labelsize=7.5)

fig.suptitle(
    'Feature Distributions Before vs After StandardScaler\n'
    '(training set — 6 representative features)',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task2_02_scaling.png'), dpi=300, bbox_inches='tight')
plt.close()


# ---
# ### 2.5 Feature Selection Analysis
# #### Near-Zero Variance

variances = X_train.var(ddof=1)
std_devs  = X_train.std(ddof=1)
means     = X_train.mean().abs()
cv        = std_devs / means

var_df = pd.DataFrame({
    'variance': variances,
    'std'     : std_devs,
    '|mean|'  : means,
    'CV'      : cv,
}).sort_values('variance')

print("\nFeatures sorted by variance (training set, raw scale):")
print(var_df.round(6).to_string())
print()
near_zero = var_df[var_df['variance'] < 1e-4]
print(f"Features with variance < 1e-4 : {len(near_zero)}  (near-zero threshold)")
print(f"Lowest variance  : '{var_df.index[0]}'  (var = {var_df['variance'].iloc[0]:.6f})")
print(f"Highest variance : '{var_df.index[-1]}'  (var = {var_df['variance'].iloc[-1]:.2f})")

# Variance plots
fig, axes = plt.subplots(1, 2, figsize=(19, 6))

ax1 = axes[0]
var_s = var_df['variance'].sort_values()
bar_c = [CLR_MAL if v < 1e-4 else PALETTE[0] for v in var_s]
ax1.barh(var_s.index, var_s.values,
         color=bar_c, height=0.65,
         edgecolor='white', linewidth=0.7, alpha=0.88)
ax1.set_xscale('log')
ax1.set_xlabel('Variance (log scale)', fontsize=11)
ax1.set_title('Feature Variance: Training Set (log scale)', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelsize=7.5)
ax1.axvline(1e-4, color=CLR_MAL, linestyle='--', linewidth=1.5,
            alpha=0.7, label='1e-4 reference line')
ax1.legend(fontsize=9)
ax1.set_facecolor('#fafafa')

ax2 = axes[1]
cv_s = var_df['CV'].sort_values()
bar_c2 = [CLR_MAL if v < 0.1 else PALETTE[2] for v in cv_s]
ax2.barh(cv_s.index, cv_s.values,
         color=bar_c2, height=0.65,
         edgecolor='white', linewidth=0.7, alpha=0.88)
ax2.axvline(0.1, color=CLR_MAL, linestyle='--', linewidth=1.5,
            alpha=0.7, label='Low-CV threshold (0.10)')
ax2.set_xlabel('Coefficient of Variation (std / |mean|)', fontsize=11)
ax2.set_title('Coefficient of Variation: Training Set', fontsize=12, fontweight='bold')
ax2.tick_params(axis='y', labelsize=7.5)
ax2.legend(fontsize=9)
ax2.set_facecolor('#fafafa')

fig.suptitle('Near-Zero Variance Analysis', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task2_03_variance.png'), dpi=300, bbox_inches='tight')
plt.close()


# #### High-Correlation Candidates

corr = X.corr()
pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
)
pairs.columns  = ['Feature A', 'Feature B', 'Pearson r']
pairs['|r|']   = pairs['Pearson r'].abs()
pairs          = pairs.sort_values('|r|', ascending=False).reset_index(drop=True)

high_corr = pairs[pairs['|r|'] >= 0.90].copy().reset_index(drop=True)

print(f"\nFeature pairs with |r| >= 0.90 : {len(high_corr)}")
print()
print(f"{'Feature A':<28} {'Feature B':<28} {'Pearson r':>10}")
print("─" * 68)
for _, row in high_corr.iterrows():
    print(f"{row['Feature A']:<28} {row['Feature B']:<28} {row['Pearson r']:>10.4f}")

involved = Counter(high_corr['Feature A'].tolist() + high_corr['Feature B'].tolist())
inv_df = (pd.DataFrame.from_dict(involved, orient='index', columns=['high-corr pairs'])
            .sort_values('high-corr pairs', ascending=False))
print(f"\nFeatures most involved in |r| >= 0.90 pairs (top 12):")
print(inv_df.head(12).to_string())

# Redundancy heatmap
geo_feats = [
    'mean radius',     'mean perimeter',     'mean area',
    'worst radius',    'worst perimeter',    'worst area',
    'radius error',    'perimeter error',    'area error',
    'mean compactness','mean concavity',     'mean concave points',
    'worst compactness','worst concavity',   'worst concave points',
]
corr_sub = X[geo_feats].corr()

fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(
    corr_sub,
    annot=True, fmt='.2f',
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.8, linecolor='white',
    ax=ax,
    annot_kws={'size': 8.5, 'weight': 'bold'},
    cbar_kws={'shrink': 0.65, 'label': 'Pearson r', 'pad': 0.02}
)
ax.set_title(
    'Redundancy Heatmap\n'
    '(top candidates for removal in a feature-reduction strategy)',
    fontsize=13, fontweight='bold', pad=14
)
ax.tick_params(axis='x', rotation=45, labelsize=9)
plt.setp(ax.get_xticklabels(), ha='right', rotation_mode='anchor')
ax.tick_params(axis='y', rotation=0,  labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task2_04_redundancy_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()


# ---
# ### 2.6 Save Scaled Splits for Downstream Tasks

X_train_scaled.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
X_test_scaled.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

print("\nSaved:")
print(f"  X_train_scaled : {X_train_scaled.shape}")
print(f"  X_test_scaled  : {X_test_scaled.shape}")
print(f"  y_train        : {y_train.shape}")
print(f"  y_test         : {y_test.shape}")
print("\nTask 2 complete. Images saved to:", img_dir)
