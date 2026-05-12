#!/usr/bin/env python
# coding: utf-8

# # Assignment 2 - Supervised Learning Methods
# ## Task 1: Data Loading and Exploration
# **Dataset:** Breast Cancer Wisconsin (Diagnostic)
# **Goal:** Load the dataset, inspect its structure, analyse class balance,
# visualise feature distributions by class, and explore the correlation structure.

# ---
# ### 1.1 Imports and Library Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as mpl
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer

# Output directories
output_dir = '/Users/arriazui/Downloads/master/C1_S2/MACHINE_LEARNING/python_code'
img_dir = os.path.join(output_dir, 'images')
os.makedirs(img_dir, exist_ok=True)

# Reproducibility
np.random.seed(42)

# Colour palette
PALETTE     = ['#4A90D9', '#E8603C', '#2EAF7D', '#9B59B6', '#F39C12', '#C0392B', '#1ABC9C']
CLR_BEN     = '#4A90D9'    # benign     → steel blue
CLR_MAL     = '#E8603C'    # malignant  → vermilion
PALETTE_DIV = 'RdBu_r'
PALETTE_SEQ = 'YlOrRd'

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
# ### 1.2 Load Dataset
#
# The dataset is loaded directly from sklearn.datasets.
# Features and target are immediately separated into X and y.

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')   # 0 = malignant, 1 = benign

print(f"{'Samples:'} {X.shape[0]}")
print(f"{'Features:'} {X.shape[1]}")
print("Target encoding:", "0 = malignant  |  1 = benign")
print(f"{'Missing values'} {X.isnull().sum().sum()}")
print()
print(X.head())


# ---
# ### 1.3 Data Types and Summary Statistics

dtype_df = pd.DataFrame({
    'dtype'    : X.dtypes,
    'non-null' : X.notnull().sum(),
    'missing'  : X.isnull().sum(),
    'missing %': (X.isnull().sum() / len(X) * 100).round(2),
})
print(dtype_df)

print()
print(X.describe().T.round(4).to_string())


# ---
# ### 1.4 Class Balance

counts = y.value_counts().sort_index()
total  = len(y)
labels = ['Malignant (0)', 'Benign (1)']

print(f"\n{'Class':<20} {'Count':>8} {'Percentage':>12}")
for lbl, cnt in zip(labels, counts.values):
    print(f"{lbl:<20} {cnt:>8}   {cnt / total * 100:>8.1f}%")
print(f"{'Total':<20} {total:>8}   {'100.0%':>9}")
print(f"\nImbalance ratio (Benign : Malignant) = {counts[1] / counts[0]:.2f} : 1")
print(f"Minority class (malignant) share      = {counts[0] / total * 100:.1f}%")

# Class balance bar chart
counts_dict = {'Malignant': int(counts[0]), 'Benign': int(counts[1])}
total = sum(counts_dict.values())

clrs       = [CLR_MAL, CLR_BEN]
cls_labels = ['Malignant', 'Benign']
aligned_counts = [counts_dict[label] for label in cls_labels]

fig, ax = plt.subplots(figsize=(8, 5))
plot_labels = cls_labels[::-1]
plot_counts = aligned_counts[::-1]
plot_clrs   = clrs[::-1]

bars = ax.barh(plot_labels, plot_counts,
               color=plot_clrs, height=0.5,
               edgecolor='white', linewidth=1.8, alpha=0.92)

for bar, cnt in zip(bars, plot_counts):
    pct = cnt / total * 100
    ax.text(bar.get_width() + 3,
            bar.get_y() + bar.get_height() / 2,
            f'{cnt}  ({pct:.1f}%)',
            va='center', fontsize=11, fontweight='bold', color='#333333')

ax.set_xlabel('Number of Samples', fontsize=11)
ax.set_title('Sample Counts per Class', fontsize=13, fontweight='bold', pad=15)
ax.set_xlim(0, max(plot_counts) * 1.30)
ax.axvline(total / 2, color='grey', linestyle='--', linewidth=1.2,
           alpha=0.55, label='Perfect balance')
ax.legend(fontsize=9, loc='lower right')
ax.tick_params(axis='y', labelsize=11)
ax.set_facecolor('#fafafa')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task1_01_class_balance.png'), dpi=300, bbox_inches='tight')
plt.close()


# ---
# ### 1.5 Feature Distributions by Class

mean_feats  = [c for c in X.columns if c.startswith('mean')]
se_feats    = [c for c in X.columns if c.endswith('error')]
worst_feats = [c for c in X.columns if c.startswith('worst')]

short_lbl = (lambda feat:
    feat.replace('mean ', '').replace(' error', '').replace('worst ', '').title())

X_vis = X.copy()
X_vis['Class'] = y.map({0: 'Malignant', 1: 'Benign'})

print(f"Mean features  ({len(mean_feats)}) : {mean_feats}")
print(f"SE features    ({len(se_feats)}) : {se_feats}")
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
            medianprops  ={'color': 'black', 'linewidth': 2.5},
            whiskerprops ={'linewidth': 1.3,  'color': '#555555'},
            capprops     ={'linewidth': 2.0,  'color': '#555555'},
            flierprops   ={'marker': 'o', 'markersize': 3.5,
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
    plt.savefig(os.path.join(img_dir, save_name), dpi=300, bbox_inches='tight')
    plt.close()


plot_distributions(mean_feats, 'Mean Measurements', 'task1_02_dist_mean.png')
plot_distributions(se_feats, 'Standard Error (SE) Measurements', 'task1_03_dist_se.png')
plot_distributions(worst_feats, 'Worst (Largest Observed) Measurements', 'task1_04_dist_worst.png')


# ---
# ### 1.6 Correlation Analysis

corr = X.corr()

pairs = (
    corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        .stack()
        .reset_index()
)
pairs.columns  = ['Feature A', 'Feature B', 'Pearson r']
pairs['|r|']   = pairs['Pearson r'].abs()
pairs          = pairs.sort_values('|r|', ascending=False).reset_index(drop=True)

print(f"\nTotal feature pairs evaluated : {len(pairs)}")
print(f"Pairs with |r| >= 0.90         : {(pairs['|r|'] >= 0.90).sum()}")
print(f"Pairs with |r| >= 0.95         : {(pairs['|r|'] >= 0.95).sum()}")
print()
print("Top 15 most correlated pairs:")
display_cols = ['Feature A', 'Feature B', 'Pearson r']
print(pairs[display_cols].head(15).to_string(index=False))

# Correlation heatmap
fig, ax = plt.subplots(figsize=(17, 14))
mask = np.triu(np.ones_like(corr, dtype=bool))

hm = sns.heatmap(
    corr, mask=mask,
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    square=True,
    linewidths=0.35, linecolor='white',
    ax=ax,
    cbar_kws={
        'shrink'  : 0.65,
        'label'   : 'Pearson r',
        'pad'     : 0.02,
        'ticks'   : [-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
    }
)

feat_list = list(corr.columns)
for _, row in pairs[pairs['|r|'] >= 0.90].iterrows():
    i = feat_list.index(row['Feature B'])
    j = feat_list.index(row['Feature A'])
    ax.add_patch(plt.Rectangle(
        (j, i), 1, 1, fill=False,
        edgecolor='gold', linewidth=2.8, zorder=5
    ))

ax.set_title(
    'Feature Correlation Matrix — Breast Cancer Wisconsin (n=569)\n'
    'Gold border: |Pearson r| >= 0.90',
    fontsize=14, fontweight='bold', pad=16
)
ax.tick_params(axis='x', rotation=90, labelsize=7.5)
ax.tick_params(axis='y', rotation=0,  labelsize=7.5)

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task1_05_correlation_heatmap.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nTask 1 complete. Images saved to:", img_dir)
