#!/usr/bin/env python
# coding: utf-8

# Assignment 2: Supervised Learning Methods
# Single-file version — Tasks 1 through 5
# Figures saved to images/ sub-directory

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import matplotlib as mpl
import seaborn as sns
import warnings
import os
import time
from collections import Counter

warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_auc_score,
)

# ── Output directories ────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR  = os.path.join(BASE_DIR, 'images')
os.makedirs(IMG_DIR, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE     = ['#4A90D9', '#E8603C', '#2EAF7D', '#9B59B6', '#F39C12', '#C0392B', '#1ABC9C']
CLR_BEN     = '#4A90D9'
CLR_MAL     = '#E8603C'
PALETTE_DIV = 'RdBu_r'
PALETTE_SEQ = 'YlOrRd'

sns.set_theme(
    style='whitegrid', palette=PALETTE, font='DejaVu Sans', font_scale=1.1,
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
np.random.seed(42)

print("Libraries loaded successfully")


# =============================================================================
# TASK 1 — Data Loading and Exploration
# =============================================================================
print("\n" + "=" * 60)
print("TASK 1 — Data Loading and Exploration")
print("=" * 60)

# ── 1.1 Load Dataset ──────────────────────────────────────────────────────────
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')   # 0 = malignant, 1 = benign

print(f"{'Samples:'} {X.shape[0]}")
print(f"{'Features:'} {X.shape[1]}")
print("Target encoding: 0 = malignant  |  1 = benign")
print(f"Missing values: {X.isnull().sum().sum()}")
print()
print(X.head().to_string())

# ── 1.2 Data Types and Summary Statistics ─────────────────────────────────────
dtype_df = pd.DataFrame({
    'dtype'    : X.dtypes,
    'non-null' : X.notnull().sum(),
    'missing'  : X.isnull().sum(),
    'missing %': (X.isnull().sum() / len(X) * 100).round(2),
})
print("\nData types:")
print(dtype_df.to_string())
print("\nSummary statistics:")
print(X.describe().T.round(4).to_string())

# ── 1.3 Class Balance ─────────────────────────────────────────────────────────
counts = y.value_counts().sort_index()
total  = len(y)
labels = ['Malignant (0)', 'Benign (1)']

print(f"\n{'Class':<20} {'Count':>8} {'Percentage':>12}")
for lbl, cnt in zip(labels, counts.values):
    print(f"{lbl:<20} {cnt:>8}   {cnt / total * 100:>8.1f}%")
print(f"{'Total':<20} {total:>8}   {'100.0%':>9}")
print(f"\nImbalance ratio (Benign : Malignant) = {counts[1] / counts[0]:.2f} : 1")
print(f"Minority class (malignant) share      = {counts[0] / total * 100:.1f}%")

# Figure 1: class balance horizontal bar chart
counts_dict    = {'Malignant': int(counts[0]), 'Benign': int(counts[1])}
total_fig      = sum(counts_dict.values())
cls_labels     = ['Malignant', 'Benign']
aligned_counts = [counts_dict[lbl] for lbl in cls_labels]

fig, ax = plt.subplots(figsize=(8, 5))
plot_labels = cls_labels[::-1]
plot_counts = aligned_counts[::-1]
plot_clrs   = [CLR_BEN, CLR_MAL]   # reversed: Benign on top
bars = ax.barh(plot_labels, plot_counts, color=plot_clrs,
               height=0.5, edgecolor='white', linewidth=1.8, alpha=0.92)
for bar, cnt in zip(bars, plot_counts):
    pct = cnt / total_fig * 100
    ax.text(bar.get_width() + 3,
            bar.get_y() + bar.get_height() / 2,
            f'{cnt}  ({pct:.1f}%)',
            va='center', fontsize=11, fontweight='bold', color='#333333')
ax.set_xlabel('Number of Samples', fontsize=11)
ax.set_title('Sample Counts per Class', fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0, max(plot_counts) * 1.30)
ax.axvline(total_fig / 2, color='grey', linestyle='--', linewidth=1.2,
           alpha=0.55, label='Perfect balance')
ax.legend(fontsize=9, loc='lower right')
ax.tick_params(axis='y', labelsize=11)
ax.set_facecolor('#fafafa')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'task1_01_class_balance.png'), dpi=300, bbox_inches='tight')
plt.close()

# ── 1.4 Feature Distributions by Class ───────────────────────────────────────
mean_feats  = [c for c in X.columns if c.startswith('mean')]
se_feats    = [c for c in X.columns if c.endswith('error')]
worst_feats = [c for c in X.columns if c.startswith('worst')]

short_lbl = lambda feat: (feat.replace('mean ', '')
                             .replace(' error', '')
                             .replace('worst ', '')
                             .title())

X_vis = X.copy()
X_vis['Class'] = y.map({0: 'Malignant', 1: 'Benign'})

print(f"\nMean features  ({len(mean_feats)})  : {mean_feats}")
print(f"SE features    ({len(se_feats)})  : {se_feats}")
print(f"Worst features ({len(worst_feats)}) : {worst_feats}")


def plot_distributions(feats, group_title, save_name):
    fig, axes = plt.subplots(2, 5, figsize=(22, 9))
    axes = axes.flatten()

    for i, feat in enumerate(feats):
        ax = axes[i]
        mal_data = X_vis[X_vis['Class'] == 'Malignant'][feat].values
        ben_data = X_vis[X_vis['Class'] == 'Benign'][feat].values

        bp = ax.boxplot(
            [mal_data, ben_data],
            labels=['Mal.', 'Ben.'],
            patch_artist=True,
            widths=0.42,
            medianprops ={'color': 'black', 'linewidth': 2.5},
            whiskerprops={'linewidth': 1.3, 'color': '#555555'},
            capprops    ={'linewidth': 2.0, 'color': '#555555'},
            flierprops  ={'marker': 'o', 'markersize': 3.5,
                          'alpha': 0.35, 'markeredgewidth': 0,
                          'markerfacecolor': '#999999'},
        )
        for box, clr in zip(bp['boxes'], [CLR_MAL, CLR_BEN]):
            box.set_facecolor(clr)
            box.set_alpha(0.65)
            box.set_linewidth(1.8)
            box.set_edgecolor(clr)

        np.random.seed(42)
        for pos, d, clr in [(1, mal_data, CLR_MAL), (2, ben_data, CLR_BEN)]:
            jit = np.random.uniform(-0.17, 0.17, len(d))
            ax.scatter(pos + jit, d, color=clr,
                       alpha=0.22, s=8, zorder=3, linewidths=0)

        ax.set_title(short_lbl(feat), fontsize=10, fontweight='bold')
        ax.set_xlabel('')
        ax.tick_params(labelsize=8)
        ax.set_facecolor('#fafafa')
        ax.spines['left'].set_linewidth(0.8)
        ax.spines['bottom'].set_linewidth(0.8)

    mal_patch = mpatches.Patch(facecolor=CLR_MAL, alpha=0.7, label='Malignant')
    ben_patch = mpatches.Patch(facecolor=CLR_BEN, alpha=0.7, label='Benign')
    fig.legend(handles=[mal_patch, ben_patch],
               loc='upper right', fontsize=11,
               frameon=True, framealpha=0.9, edgecolor='#cccccc',
               bbox_to_anchor=(0.99, 0.99))
    fig.suptitle(f'Feature Distributions by Class: {group_title}',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, save_name), dpi=300, bbox_inches='tight')
    plt.close()


plot_distributions(mean_feats,  'Mean Measurements',                     'task1_02_dist_mean.png')
plot_distributions(se_feats,    'Standard Error (SE) Measurements',      'task1_03_dist_se.png')
plot_distributions(worst_feats, 'Worst (Largest Observed) Measurements', 'task1_04_dist_worst.png')

# ── 1.5 Correlation Analysis ──────────────────────────────────────────────────
corr  = X.corr()
pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
)
pairs.columns = ['Feature A', 'Feature B', 'Pearson r']
pairs['|r|']  = pairs['Pearson r'].abs()
pairs         = pairs.sort_values('|r|', ascending=False).reset_index(drop=True)

print(f"\nTotal feature pairs evaluated : {len(pairs)}")
print(f"Pairs with |r| ≥ 0.90         : {(pairs['|r|'] >= 0.90).sum()}")
print(f"Pairs with |r| ≥ 0.95         : {(pairs['|r|'] >= 0.95).sum()}")
print("\nTop 15 most correlated pairs:")
print(pairs[['Feature A', 'Feature B', 'Pearson r']].head(15).to_string(index=False))

# Figure 5: full correlation heatmap
fig, ax = plt.subplots(figsize=(17, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr, mask=mask,
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.35, linecolor='white', ax=ax,
    cbar_kws={
        'shrink': 0.65, 'label': 'Pearson r', 'pad': 0.02,
        'ticks' : [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
    }
)
feat_list = list(corr.columns)
for _, row in pairs[pairs['|r|'] >= 0.90].iterrows():
    i = feat_list.index(row['Feature B'])
    j = feat_list.index(row['Feature A'])
    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                edgecolor='gold', linewidth=2.8, zorder=5))
ax.set_title(
    'Feature Correlation Matrix — Breast Cancer Wisconsin (n=569)\n'
    'Gold border: |Pearson r| ≥ 0.90',
    fontsize=14, fontweight='bold', pad=16
)
ax.tick_params(axis='x', rotation=90, labelsize=7.5)
ax.tick_params(axis='y', rotation=0,  labelsize=7.5)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'task1_05_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# TASK 2 — Preprocessing and Feature Analysis
# =============================================================================
print("\n" + "=" * 60)
print("TASK 2 — Preprocessing and Feature Analysis")
print("=" * 60)

# ── 2.1 Stratified Train-Test Split ──────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"{'Split':<12} {'n':>6} {'Malignant':>12} {'Benign':>8} {'Mal%':>7} {'Ben%':>7}")
for split_name, X_s, y_s in [('Training', X_train, y_train),
                               ('Test',     X_test,  y_test),
                               ('Full',     X,       y)]:
    n_mal = (y_s == 0).sum()
    n_ben = (y_s == 1).sum()
    print(f"{split_name:<12} {len(y_s):>6} {n_mal:>12} {n_ben:>8} "
          f"{n_mal / len(y_s) * 100:>6.1f}% {n_ben / len(y_s) * 100:>6.1f}%")
print(f"\nX_train : {X_train.shape}  |  X_test : {X_test.shape}")
print(f"y_train : {y_train.shape}   |  y_test : {y_test.shape}")

# Figure 6: class balance across splits
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
_splits = [
    ('Full Dataset\n(n = 569)', y),
    ('Training Set\n(n = 455)', y_train),
    ('Test Set\n(n = 114)',     y_test),
]
for ax, (title, y_s) in zip(axes, _splits):
    cnts = y_s.value_counts().sort_index()
    bars = ax.bar(
        ['Malignant', 'Benign'], cnts.values,
        color=[CLR_MAL, CLR_BEN], width=0.48,
        edgecolor='white', linewidth=1.8, alpha=0.90
    )
    for bar, cnt in zip(bars, cnts.values):
        pct = cnt / len(y_s) * 100
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.5,
                f'{cnt}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_ylabel('Samples', fontsize=10)
    ax.set_ylim(0, max(cnts.values) * 1.30)
    ax.tick_params(labelsize=10)
    ax.set_facecolor('#fafafa')
fig.suptitle('Class Balance: Full, Training, and Test Sets',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'task2_01_class_balance_splits.png'), dpi=300, bbox_inches='tight')
plt.close()

# ── 2.2 Feature Scaling with StandardScaler ───────────────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)   # fit on train only

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
X_test_scaled  = pd.DataFrame(X_test_scaled,  columns=X.columns, index=X_test.index)

train_means = X_train_scaled.mean().abs()
train_stds  = X_train_scaled.std()
test_means  = X_test_scaled.mean().abs()

print("\nPost-scaling verification:")
print(f"  Training set max |mean|  : {train_means.max():.2e}  (expected ≈ 0)")
print(f"  Training set mean std    : {train_stds.mean():.6f}  (expected ≈ 1)")
print(f"  Training set min std     : {train_stds.min():.6f}")
print(f"  Training set max std     : {train_stds.max():.6f}")
print(f"  Test set max |mean|      : {test_means.max():.4f}  (leakage check)")

# Figure 7: before vs after scaling
sample_feats  = [
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
    ax_top.hist(X_train[feat], bins=32, color=clr, alpha=0.80,
                edgecolor='white', linewidth=0.5)
    ax_top.set_title(dname, fontsize=9, fontweight='bold')
    if col == 0:
        ax_top.set_ylabel('Before Scaling\n\nCount', fontsize=9, labelpad=4)
    ax_top.set_xlabel('Raw value', fontsize=8)
    ax_top.set_facecolor('#fafafa')
    ax_top.tick_params(labelsize=7.5)

    ax_bot = axes[1, col]
    ax_bot.hist(X_train_scaled[feat], bins=32, color=clr, alpha=0.80,
                edgecolor='white', linewidth=0.5)
    ax_bot.axvline(0, color='crimson', linestyle='--', linewidth=1.8,
                   alpha=0.85, label='μ = 0')
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
plt.savefig(os.path.join(IMG_DIR, 'task2_02_scaling_before_after.png'), dpi=300, bbox_inches='tight')
plt.close()

# ── 2.3 Feature Selection Analysis ───────────────────────────────────────────
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
near_zero = var_df[var_df['variance'] < 1e-4]
print(f"\nFeatures with variance < 1e-4 : {len(near_zero)}  (near-zero threshold)")
print(f"Lowest variance  : '{var_df.index[0]}'  (var = {var_df['variance'].iloc[0]:.6f})")
print(f"Highest variance : '{var_df.index[-1]}'  (var = {var_df['variance'].iloc[-1]:.2f})")

# Figure 8: variance & coefficient of variation
fig, axes = plt.subplots(1, 2, figsize=(19, 6))

ax1   = axes[0]
var_s = var_df['variance'].sort_values()
bar_c = [CLR_MAL if v < 1e-4 else PALETTE[0] for v in var_s]
ax1.barh(var_s.index, var_s.values, color=bar_c, height=0.65,
         edgecolor='white', linewidth=0.7, alpha=0.88)
ax1.set_xscale('log')
ax1.set_xlabel('Variance (log scale)', fontsize=11)
ax1.set_title('Feature Variance: Training Set (log scale)', fontsize=12, fontweight='bold')
ax1.tick_params(axis='y', labelsize=7.5)
ax1.axvline(1e-4, color=CLR_MAL, linestyle='--', linewidth=1.5,
            alpha=0.7, label='1e-4 reference line (scale artefact below)')
ax1.legend(fontsize=9)
ax1.set_facecolor('#fafafa')

ax2   = axes[1]
cv_s  = var_df['CV'].sort_values()
bar_c2 = [CLR_MAL if v < 0.1 else PALETTE[2] for v in cv_s]
ax2.barh(cv_s.index, cv_s.values, color=bar_c2, height=0.65,
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
plt.savefig(os.path.join(IMG_DIR, 'task2_03_variance_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# High-correlation candidates
high_corr = pairs[pairs['|r|'] >= 0.90].copy().reset_index(drop=True)

print(f"\nFeature pairs with |r| ≥ 0.90 : {len(high_corr)}")
print(f"\n{'Feature A':<28} {'Feature B':<28} {'Pearson r':>10}")
print("─" * 68)
for _, row in high_corr.iterrows():
    print(f"{row['Feature A']:<28} {row['Feature B']:<28} {row['Pearson r']:>10.4f}")

involved = Counter(high_corr['Feature A'].tolist() + high_corr['Feature B'].tolist())
inv_df = (pd.DataFrame.from_dict(involved, orient='index', columns=['high-corr pairs'])
            .sort_values('high-corr pairs', ascending=False))
print(f"\nFeatures most involved in |r| ≥ 0.90 pairs (top 12):")
print(inv_df.head(12).to_string())

# Figure 9: redundancy heatmap
geo_feats = [
    'mean radius',      'mean perimeter',      'mean area',
    'worst radius',     'worst perimeter',     'worst area',
    'radius error',     'perimeter error',     'area error',
    'mean compactness', 'mean concavity',      'mean concave points',
    'worst compactness','worst concavity',     'worst concave points',
]
corr_sub = X[geo_feats].corr()

fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(
    corr_sub,
    annot=True, fmt='.2f',
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.8, linecolor='white', ax=ax,
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
ax.tick_params(axis='y', rotation=0, labelsize=9)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'task2_04_redundancy_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# TASK 3 — K-Nearest Neighbours (KNN)
# =============================================================================
print("\n" + "=" * 60)
print("TASK 3 — K-Nearest Neighbours (KNN)")
print("=" * 60)

# ── 3.1 Sweep k = 1 … 20 ─────────────────────────────────────────────────────
k_values    = range(1, 21)
knn_results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    knn_results.append({'k': k, 'test_accuracy': acc})

knn_results  = pd.DataFrame(knn_results)
best_k       = 17   # selected — see discussion in notebook
best_knn_acc = knn_results.loc[knn_results['k'] == best_k, 'test_accuracy'].values[0]

print(f"{'k':<5} {'Test Accuracy':>14}")
print("─" * 21)
for _, row in knn_results.iterrows():
    marker = "  ← selected" if row['k'] == best_k else ""
    print(f"{int(row['k']):<5} {row['test_accuracy']:>14.4f}{marker}")
print(f"\nSelected k    : {best_k}")
print(f"Test accuracy : {best_knn_acc:.4f}")

# Figure 10: KNN accuracy vs k
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(
    knn_results['k'], knn_results['test_accuracy'],
    marker='o', linewidth=2.2, color=PALETTE[0],
    markerfacecolor=CLR_MAL, markeredgecolor='white',
    markeredgewidth=1.2, markersize=8, zorder=3
)
ax.axvline(best_k, color='crimson', linestyle='--', linewidth=1.8,
           label=f'Optimal k = {best_k}  (acc = {best_knn_acc:.4f})')
ax.fill_between(knn_results['k'], knn_results['test_accuracy'],
                alpha=0.08, color=PALETTE[0])
ax.set_xlabel('k (number of neighbours)', fontsize=11)
ax.set_ylabel('Test Accuracy', fontsize=11)
ax.set_title('KNN: Test Accuracy vs k', fontsize=13, fontweight='bold')
ax.set_xticks(knn_results['k'])
ax.set_ylim(knn_results['test_accuracy'].min() - 0.01,
            knn_results['test_accuracy'].max() + 0.015)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(fontsize=10)
ax.set_facecolor('#fafafa')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'task3_01_knn_accuracy_vs_k.png'), dpi=300, bbox_inches='tight')
plt.close()

# ── 3.2 Final KNN Model ───────────────────────────────────────────────────────
final_knn  = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
y_pred_knn = final_knn.predict(X_test_scaled)

print(f"\nFinal KNN model  —  k = {best_k}")
print("\nClassification report:")
print(classification_report(y_test, y_pred_knn,
                             target_names=['Malignant', 'Benign'], digits=4))

cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Confusion matrix:")
print(cm_knn)

# Figure 11: KNN confusion matrix
fig, ax = plt.subplots()
ax.grid(False)
ConfusionMatrixDisplay(confusion_matrix=cm_knn,
                       display_labels=['Malignant', 'Benign']
                       ).plot(cmap='Blues', values_format='d', ax=ax)
plt.title(f'KNN Confusion Matrix  (k = {best_k})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'task3_02_knn_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()


# =============================================================================
# TASK 4 — Logistic Regression
# =============================================================================
print("\n" + "=" * 60)
print("TASK 4 — Logistic Regression")
print("=" * 60)

# ── 4.1 Sweep C ───────────────────────────────────────────────────────────────
C_values   = np.logspace(-3, 3, 10)
lr_results = []

for C in C_values:
    model = LogisticRegression(C=C, solver='liblinear', max_iter=5000, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    lr_results.append({'C': C, 'test_accuracy': accuracy_score(y_test, y_pred)})

lr_results    = pd.DataFrame(lr_results)
best_accuracy = lr_results['test_accuracy'].max()
best_C        = lr_results.loc[lr_results['test_accuracy'] == best_accuracy, 'C'].min()

print(lr_results.to_string(index=False))
print(f"\nBest accuracy : {best_accuracy:.4f}")
print(f"Selected C    : {best_C:.6f}")

# Figure 12: LR accuracy vs C
plt.figure(figsize=(9, 5))
plt.plot(lr_results['C'], lr_results['test_accuracy'], marker='o', linewidth=2)
plt.axvline(best_C, color='red', linestyle='--', label=f'Best C = {best_C:.3g}')
plt.xscale('log')
plt.xlabel('C')
plt.ylabel('Test accuracy')
plt.title('Logistic Regression: Test Accuracy vs C')
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'task4_01_lr_accuracy_vs_C.png'), dpi=300, bbox_inches='tight')
plt.close()

# ── 4.2 Final Logistic Regression Model ──────────────────────────────────────
final_lr = LogisticRegression(C=best_C, solver='liblinear', max_iter=5000, random_state=42)
final_lr.fit(X_train_scaled, y_train)
y_pred_lr = final_lr.predict(X_test_scaled)

print("\nClassification report:")
print(classification_report(y_test, y_pred_lr,
                             target_names=['Malignant', 'Benign'], digits=4))

cm_lr = confusion_matrix(y_test, y_pred_lr)
print("Confusion matrix:")
print(cm_lr)

# Figure 13: LR confusion matrix
fig, ax = plt.subplots()
ax.grid(False)
ConfusionMatrixDisplay(confusion_matrix=cm_lr,
                       display_labels=['Malignant', 'Benign']
                       ).plot(cmap='Blues', values_format='d', ax=ax)
plt.title('Logistic Regression Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'task4_02_lr_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# ── 4.3 Coefficient Interpretation ───────────────────────────────────────────
coef_df = pd.DataFrame({
    'feature'               : X.columns,
    'coefficient_for_benign': final_lr.coef_[0],
})
coef_df['malignancy_weight'] = -coef_df['coefficient_for_benign']
coef_df['abs_coefficient']   =  coef_df['coefficient_for_benign'].abs()

top_malignant = coef_df.sort_values('malignancy_weight', ascending=False).head(10)
top_overall   = coef_df.sort_values('abs_coefficient',   ascending=False).head(15)

print("\nTop 10 features predicting malignancy:")
print(top_malignant[['feature', 'malignancy_weight', 'coefficient_for_benign']].to_string(index=False))
print("\nTop 15 largest absolute coefficients:")
print(top_overall[['feature', 'coefficient_for_benign', 'abs_coefficient']].to_string(index=False))


# =============================================================================
# TASK 5 — Support Vector Machine (SVM)
# =============================================================================
print("\n" + "=" * 60)
print("TASK 5 — Support Vector Machine (SVM)")
print("=" * 60)

# ── 5.1 Kernel Comparison: Linear, RBF, Polynomial ───────────────────────────
kernel_configs = [
    {'kernel': 'linear', 'label': 'Linear',     'C': 1, 'extra': {}},
    {'kernel': 'rbf',    'label': 'RBF',        'C': 1, 'extra': {'gamma': 'scale'}},
    {'kernel': 'poly',   'label': 'Polynomial', 'C': 1, 'extra': {'degree': 3, 'gamma': 'scale'}},
]
kernel_results = []

print(f"\n{'Kernel':<14} {'Accuracy':>10} {'AUC':>8} {'Train time (s)':>16}")
print("─" * 52)

for cfg in kernel_configs:
    svm = SVC(kernel=cfg['kernel'], C=cfg['C'],
              random_state=42, probability=True, **cfg['extra'])
    t0 = time.perf_counter()
    svm.fit(X_train_scaled, y_train)
    train_time = time.perf_counter() - t0

    y_pred  = svm.predict(X_test_scaled)
    y_proba = svm.predict_proba(X_test_scaled)[:, 1]
    acc     = accuracy_score(y_test, y_pred)
    auc     = roc_auc_score(y_test, y_proba)

    kernel_results.append({
        'Kernel'        : cfg['label'],
        'Accuracy'      : acc,
        'AUC'           : auc,
        'Train time (s)': round(train_time, 4),
        '_model'        : svm,
        '_y_pred'       : y_pred,
        '_y_proba'      : y_proba,
    })
    print(f"{cfg['label']:<14} {acc:>10.4f} {auc:>8.4f} {train_time:>14.4f}s")

display_cols = ['Kernel', 'Accuracy', 'AUC', 'Train time (s)']
kernel_df    = pd.DataFrame(kernel_results)[display_cols]
print("\nKernel comparison summary:")
print(kernel_df.to_string(index=False))

# Figure 14: kernel comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
kernels    = kernel_df['Kernel'].tolist()
accuracies = kernel_df['Accuracy'].tolist()
aucs       = kernel_df['AUC'].tolist()
bar_colors = PALETTE[:len(kernels)]

ax1 = axes[0]
bars = ax1.bar(kernels, accuracies, color=bar_colors, width=0.4,
               edgecolor='white', linewidth=1.8, alpha=0.90)
for bar, val in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_ylim(min(accuracies) - 0.02, 1.01)
ax1.set_ylabel('Test Accuracy', fontsize=11)
ax1.set_title('SVM Kernel Comparison: Accuracy', fontsize=12, fontweight='bold')
ax1.set_facecolor('#fafafa')

ax2 = axes[1]
bars2 = ax2.bar(kernels, aucs, color=bar_colors, width=0.4,
                edgecolor='white', linewidth=1.8, alpha=0.90)
for bar, val in zip(bars2, aucs):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
             f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_ylim(min(aucs) - 0.02, 1.01)
ax2.set_ylabel('ROC AUC', fontsize=11)
ax2.set_title('SVM Kernel Comparison: AUC', fontsize=12, fontweight='bold')
ax2.set_facecolor('#fafafa')

fig.suptitle('SVM Kernel Comparison (default C=1)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'task5_01_kernel_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()

# ── 5.2 RBF Kernel: C × Gamma Grid Search (5-fold CV on training set) ────────
C_values_rbf = [0.01, 0.1, 1, 10, 100]
gamma_values  = [0.001, 0.01, 0.1, 1, 10]

param_grid = {'C': C_values_rbf, 'gamma': gamma_values}
grid_cv = GridSearchCV(
    SVC(kernel='rbf', random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    return_train_score=False,
)

t0 = time.perf_counter()
grid_cv.fit(X_train_scaled, y_train)
search_time = time.perf_counter() - t0

# Reshape results: GridSearchCV iterates C outer, gamma inner
# → acc_grid[gi, ci] = CV accuracy for gamma_values[gi], C_values_rbf[ci]
scores_1d = grid_cv.cv_results_['mean_test_score']
times_1d  = grid_cv.cv_results_['mean_fit_time']
acc_grid  = scores_1d.reshape(len(C_values_rbf), len(gamma_values)).T
time_grid = times_1d.reshape(len(C_values_rbf), len(gamma_values)).T

best_idx   = np.unravel_index(np.argmax(acc_grid), acc_grid.shape)
best_gamma = gamma_values[best_idx[0]]
best_C_rbf = C_values_rbf[best_idx[1]]
best_acc   = acc_grid[best_idx]

print(f"\n5-fold CV grid search completed in {search_time:.4f} s  (training set only)")
print(f"\nCV accuracy grid  (rows = gamma, cols = C):")
print(f"{'C / gamma':<12}", end='')
print('  '.join([f'{g:>7}' for g in gamma_values]))
print("─" * 65)
for ci, C in enumerate(C_values_rbf):
    row_str = f"C = {C:<8}"
    for gi in range(len(gamma_values)):
        row_str += f"  {acc_grid[gi, ci]:>7.4f}"
    print(row_str)

print(f"\nBest RBF CV accuracy : {best_acc:.4f}")
print(f"Best C               : {best_C_rbf}")
print(f"Best gamma           : {best_gamma}")

# Figure 15: grid search heatmaps
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

ax_acc = axes[0]
sns.heatmap(
    acc_grid, ax=ax_acc,
    annot=True, fmt='.4f', cmap='YlOrRd',
    xticklabels=[str(c) for c in C_values_rbf],
    yticklabels=[str(g) for g in gamma_values],
    linewidths=0.5, linecolor='white',
    cbar_kws={'label': 'CV Accuracy', 'shrink': 0.85},
    annot_kws={'size': 9, 'weight': 'bold'},
)
ax_acc.add_patch(plt.Rectangle(
    (best_idx[1], best_idx[0]), 1, 1,
    fill=False, edgecolor='cyan', linewidth=3.5, zorder=5
))
ax_acc.set_xlabel('C (regularisation)', fontsize=11)
ax_acc.set_ylabel('Gamma (kernel bandwidth)', fontsize=11)
ax_acc.set_title('RBF SVM: CV Accuracy Grid (5-fold)\n(cyan border = best combination)',
                 fontsize=12, fontweight='bold')

ax_time = axes[1]
sns.heatmap(
    time_grid, ax=ax_time,
    annot=True, fmt='.4f', cmap='Blues',
    xticklabels=[str(c) for c in C_values_rbf],
    yticklabels=[str(g) for g in gamma_values],
    linewidths=0.5, linecolor='white',
    cbar_kws={'label': 'Mean CV Fit Time per Fold (s)', 'shrink': 0.85},
    annot_kws={'size': 9},
)
ax_time.set_xlabel('C (regularisation)', fontsize=11)
ax_time.set_ylabel('Gamma (kernel bandwidth)', fontsize=11)
ax_time.set_title('RBF SVM: Mean CV Fit Time per Fold (s)', fontsize=12, fontweight='bold')

fig.suptitle('RBF Kernel Hyperparameter Search: C × Gamma  (5-fold CV on training set)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'task5_02_rbf_grid_search.png'), dpi=300, bbox_inches='tight')
plt.close()

# Figure 16: effect of C and gamma on accuracy
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
acc_vs_C     = acc_grid[best_idx[0], :]
acc_vs_gamma = acc_grid[:, best_idx[1]]

ax_c = axes[0]
ax_c.plot([str(c) for c in C_values_rbf], acc_vs_C,
          marker='o', linewidth=2.2, color=PALETTE[0],
          markerfacecolor=CLR_MAL, markeredgecolor='white',
          markeredgewidth=1.2, markersize=9)
ax_c.set_xlabel(f'C  (gamma fixed at {best_gamma})', fontsize=11)
ax_c.set_ylabel('CV Accuracy', fontsize=11)
ax_c.set_title('RBF SVM: CV Accuracy vs C', fontsize=12, fontweight='bold')
ax_c.set_facecolor('#fafafa')
ax_c.set_ylim(min(acc_vs_C) - 0.02, max(acc_vs_C) + 0.015)
ax_c.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

ax_g = axes[1]
ax_g.plot([str(g) for g in gamma_values], acc_vs_gamma,
          marker='s', linewidth=2.2, color=PALETTE[2],
          markerfacecolor=PALETTE[1], markeredgecolor='white',
          markeredgewidth=1.2, markersize=9)
ax_g.set_xlabel(f'Gamma  (C fixed at {best_C_rbf})', fontsize=11)
ax_g.set_ylabel('CV Accuracy', fontsize=11)
ax_g.set_title('RBF SVM: CV Accuracy vs Gamma', fontsize=12, fontweight='bold')
ax_g.set_facecolor('#fafafa')
ax_g.set_ylim(min(acc_vs_gamma) - 0.05, max(acc_vs_gamma) + 0.015)
ax_g.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

fig.suptitle('RBF Kernel: Effect of C and Gamma on CV Accuracy  (5-fold CV)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'task5_03_rbf_C_gamma_effect.png'), dpi=300, bbox_inches='tight')
plt.close()

# ── 5.3 Final SVM Model ───────────────────────────────────────────────────────
final_svm = SVC(kernel='rbf', C=best_C_rbf, gamma=best_gamma,
                random_state=42, probability=True)
t0 = time.perf_counter()
final_svm.fit(X_train_scaled, y_train)
final_train_time = time.perf_counter() - t0

y_pred_svm  = final_svm.predict(X_test_scaled)
y_proba_svm = final_svm.predict_proba(X_test_scaled)[:, 1]

print(f"\nFinal SVM model — kernel=RBF, C={best_C_rbf}, gamma={best_gamma}")
print(f"Training time    : {final_train_time:.4f} s")
print("\nClassification report:")
print(classification_report(y_test, y_pred_svm,
                             target_names=['Malignant', 'Benign'], digits=4))

cm_svm = confusion_matrix(y_test, y_pred_svm)
print("Confusion matrix:")
print(cm_svm)
print(f"\nFinal test accuracy : {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Final AUC           : {roc_auc_score(y_test, y_proba_svm):.4f}")

# Figure 17: SVM confusion matrix
fig, ax = plt.subplots()
ax.grid(False)
ConfusionMatrixDisplay(confusion_matrix=cm_svm,
                       display_labels=['Malignant', 'Benign']
                       ).plot(cmap='Blues', values_format='d', ax=ax)
plt.title(
    f'SVM Confusion Matrix\n(RBF kernel, C={best_C_rbf}, gamma={best_gamma})',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, 'task5_04_svm_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

# ── 5.4 Full Kernel Summary Table ─────────────────────────────────────────────
tuned_rbf_row = {
    'Kernel'        : f'RBF (tuned C={best_C_rbf}, γ={best_gamma})',
    'Accuracy'      : accuracy_score(y_test, y_pred_svm),
    'AUC'           : roc_auc_score(y_test, y_proba_svm),
    'Train time (s)': round(final_train_time, 4),
}
summary_rows = pd.DataFrame(kernel_results)[display_cols].copy()
summary_rows = pd.concat([summary_rows, pd.DataFrame([tuned_rbf_row])], ignore_index=True)

print("\nFull SVM summary (including tuned RBF):")
print(summary_rows.to_string(index=False))

# =============================================================================
print(f"\n{'=' * 60}")
print(f"All tasks complete.  {len(os.listdir(IMG_DIR))} figures saved to: {IMG_DIR}")
print("=" * 60)
