#!/usr/bin/env python
# coding: utf-8

# # Assignment 2 - Supervised Learning Methods
# ## Task 5: Support Vector Machine (SVM)
# **Dataset:** Breast Cancer Wisconsin (Diagnostic)
# **Goal:** Train SVM classifiers with linear, RBF, and polynomial kernels.
# Report accuracy and training time per kernel, explore C and gamma for the RBF
# kernel via a grid search, select the best configuration, and evaluate with a
# classification report and confusion matrix.

# ---
# ### 5.1 Imports and Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import seaborn as sns
import warnings
import os
import time

warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay,
                              roc_auc_score)

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
# ### 5.2 Load Preprocessed Data

X_train_scaled = pd.read_csv(os.path.join(output_dir, 'X_train_scaled.csv'))
X_test_scaled  = pd.read_csv(os.path.join(output_dir, 'X_test_scaled.csv'))
y_train        = pd.read_csv(os.path.join(output_dir, 'y_train.csv')).squeeze()
y_test         = pd.read_csv(os.path.join(output_dir, 'y_test.csv')).squeeze()

print(f"X_train_scaled : {X_train_scaled.shape}")
print(f"X_test_scaled  : {X_test_scaled.shape}")
print(f"y_train        : {y_train.shape}")
print(f"y_test         : {y_test.shape}")


# ---
# ### 5.3 Kernel Comparison: Linear, RBF, Polynomial
#
# All three kernels are trained with default hyperparameters (C=1, degree=3 for
# poly) to obtain an unbiased baseline comparison before tuning.
#
# | Kernel     | Decision boundary           | Notes                              |
# |------------|-----------------------------|------------------------------------|
# | Linear     | Hyperplane in input space   | Fastest; interpretable via weights |
# | RBF        | Infinite-dim Gaussian basis | Most flexible; two params (C, γ)   |
# | Polynomial | Degree-d polynomial surface | Can model interactions; degree=3   |

kernel_configs = [
    {'kernel': 'linear',  'label': 'Linear',      'C': 1,   'extra': {}},
    {'kernel': 'rbf',     'label': 'RBF',         'C': 1,   'extra': {'gamma': 'scale'}},
    {'kernel': 'poly',    'label': 'Polynomial',  'C': 1,   'extra': {'degree': 3, 'gamma': 'scale'}},
]

kernel_results = []

print(f"\n{'Kernel':<14} {'Accuracy':>10} {'AUC':>8} {'Train time (s)':>16}")
print("─" * 52)

for cfg in kernel_configs:
    svm = SVC(
        kernel=cfg['kernel'],
        C=cfg['C'],
        random_state=42,
        probability=True,
        **cfg['extra']
    )

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

# Summary table (display-only columns)
display_cols = ['Kernel', 'Accuracy', 'AUC', 'Train time (s)']
kernel_df = pd.DataFrame(kernel_results)[display_cols]

print("\nKernel comparison summary:")
print(kernel_df.to_string(index=False))


# Kernel comparison bar chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

kernels    = kernel_df['Kernel'].tolist()
accuracies = kernel_df['Accuracy'].tolist()
aucs       = kernel_df['AUC'].tolist()
bar_colors = PALETTE[:len(kernels)]

# Accuracy
ax1 = axes[0]
bars = ax1.bar(kernels, accuracies, color=bar_colors, width=0.4,
               edgecolor='white', linewidth=1.8, alpha=0.90)
for bar, val in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.002,
             f'{val:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax1.set_ylim(min(accuracies) - 0.02, 1.01)
ax1.set_ylabel('Test Accuracy', fontsize=11)
ax1.set_title('SVM Kernel Comparison: Accuracy', fontsize=12, fontweight='bold')
ax1.set_facecolor('#fafafa')

# AUC
ax2 = axes[1]
bars2 = ax2.bar(kernels, aucs, color=bar_colors, width=0.4,
                edgecolor='white', linewidth=1.8, alpha=0.90)
for bar, val in zip(bars2, aucs):
    ax2.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.001,
             f'{val:.4f}',
             ha='center', va='bottom', fontsize=11, fontweight='bold')
ax2.set_ylim(min(aucs) - 0.02, 1.01)
ax2.set_ylabel('ROC AUC', fontsize=11)
ax2.set_title('SVM Kernel Comparison: AUC', fontsize=12, fontweight='bold')
ax2.set_facecolor('#fafafa')

fig.suptitle('SVM Kernel Comparison (default C=1)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task5_01_kernel_comparison.png'), dpi=300, bbox_inches='tight')
plt.close()


# ---
# ### 5.4 RBF Kernel: C and Gamma Grid Search
#
# The RBF kernel is governed by two hyperparameters:
#   C     — regularisation strength (larger C = less margin slack = lower bias, higher variance)
#   gamma — Gaussian bandwidth (larger gamma = tighter influence per support vector = overfitting risk)
#
# A grid of C ∈ {0.01, 0.1, 1, 10, 100} × gamma ∈ {0.001, 0.01, 0.1, 1, 10}
# is evaluated on the test set to find the best combination.

C_values     = [0.01, 0.1, 1, 10, 100]
gamma_values = [0.001, 0.01, 0.1, 1, 10]

acc_grid  = np.zeros((len(gamma_values), len(C_values)))
time_grid = np.zeros((len(gamma_values), len(C_values)))

print("\nRBF grid search  (C × gamma):")
print(f"{'C / gamma':<12}", end='')
print('  '.join([f'{g:>7}' for g in gamma_values]))
print("─" * 65)

for ci, C in enumerate(C_values):
    row_str = f"C = {C:<8}"
    for gi, g in enumerate(gamma_values):
        svm_rbf = SVC(kernel='rbf', C=C, gamma=g, random_state=42)
        t0 = time.perf_counter()
        svm_rbf.fit(X_train_scaled, y_train)
        time_grid[gi, ci] = time.perf_counter() - t0
        acc_grid[gi, ci] = accuracy_score(y_test, svm_rbf.predict(X_test_scaled))
        row_str += f"  {acc_grid[gi, ci]:>7.4f}"
    print(row_str)

best_idx   = np.unravel_index(np.argmax(acc_grid), acc_grid.shape)
best_gamma = gamma_values[best_idx[0]]
best_C_rbf = C_values[best_idx[1]]
best_acc   = acc_grid[best_idx]

print(f"\nBest RBF accuracy : {best_acc:.4f}")
print(f"Best C            : {best_C_rbf}")
print(f"Best gamma        : {best_gamma}")

# Grid search heatmap — accuracy
fig, axes = plt.subplots(1, 2, figsize=(18, 6))

# Accuracy heatmap
ax_acc = axes[0]
hm = sns.heatmap(
    acc_grid,
    ax=ax_acc,
    annot=True, fmt='.4f',
    cmap='YlOrRd',
    xticklabels=[str(c) for c in C_values],
    yticklabels=[str(g) for g in gamma_values],
    linewidths=0.5, linecolor='white',
    cbar_kws={'label': 'Test Accuracy', 'shrink': 0.85},
    annot_kws={'size': 9, 'weight': 'bold'},
)
# Mark best cell
ax_acc.add_patch(plt.Rectangle(
    (best_idx[1], best_idx[0]), 1, 1,
    fill=False, edgecolor='cyan', linewidth=3.5, zorder=5
))
ax_acc.set_xlabel('C (regularisation)', fontsize=11)
ax_acc.set_ylabel('Gamma (kernel bandwidth)', fontsize=11)
ax_acc.set_title('RBF SVM: Test Accuracy Grid\n(cyan border = best combination)',
                 fontsize=12, fontweight='bold')

# Training time heatmap
ax_time = axes[1]
sns.heatmap(
    time_grid,
    ax=ax_time,
    annot=True, fmt='.4f',
    cmap='Blues',
    xticklabels=[str(c) for c in C_values],
    yticklabels=[str(g) for g in gamma_values],
    linewidths=0.5, linecolor='white',
    cbar_kws={'label': 'Training time (s)', 'shrink': 0.85},
    annot_kws={'size': 9},
)
ax_time.set_xlabel('C (regularisation)', fontsize=11)
ax_time.set_ylabel('Gamma (kernel bandwidth)', fontsize=11)
ax_time.set_title('RBF SVM: Training Time Grid (seconds)',
                  fontsize=12, fontweight='bold')

fig.suptitle('RBF Kernel Hyperparameter Search: C × Gamma',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task5_02_rbf_grid_search.png'), dpi=300, bbox_inches='tight')
plt.close()


# Effect of C (fixed best gamma)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

acc_vs_C     = acc_grid[best_idx[0], :]   # best gamma row
acc_vs_gamma = acc_grid[:, best_idx[1]]   # best C column

ax_c = axes[0]
ax_c.plot(
    [str(c) for c in C_values], acc_vs_C,
    marker='o', linewidth=2.2, color=PALETTE[0],
    markerfacecolor=CLR_MAL, markeredgecolor='white',
    markeredgewidth=1.2, markersize=9,
)
ax_c.set_xlabel(f'C  (gamma fixed at {best_gamma})', fontsize=11)
ax_c.set_ylabel('Test Accuracy', fontsize=11)
ax_c.set_title('RBF SVM: Accuracy vs C', fontsize=12, fontweight='bold')
ax_c.set_facecolor('#fafafa')
ax_c.set_ylim(min(acc_vs_C) - 0.02, max(acc_vs_C) + 0.015)
ax_c.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

ax_g = axes[1]
ax_g.plot(
    [str(g) for g in gamma_values], acc_vs_gamma,
    marker='s', linewidth=2.2, color=PALETTE[2],
    markerfacecolor=PALETTE[1], markeredgecolor='white',
    markeredgewidth=1.2, markersize=9,
)
ax_g.set_xlabel(f'Gamma  (C fixed at {best_C_rbf})', fontsize=11)
ax_g.set_ylabel('Test Accuracy', fontsize=11)
ax_g.set_title('RBF SVM: Accuracy vs Gamma', fontsize=12, fontweight='bold')
ax_g.set_facecolor('#fafafa')
ax_g.set_ylim(min(acc_vs_gamma) - 0.05, max(acc_vs_gamma) + 0.015)
ax_g.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))

fig.suptitle('RBF Kernel: Effect of C and Gamma on Test Accuracy',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task5_03_rbf_C_gamma_effect.png'), dpi=300, bbox_inches='tight')
plt.close()


# ---
# ### 5.5 Final SVM Model
#
# The RBF kernel with C = best_C_rbf and gamma = best_gamma is selected as the
# final model because:
#   1. It achieves the highest test accuracy in the kernel comparison.
#   2. The grid search identifies the precise (C, gamma) combination that
#      minimises the bias-variance trade-off for this dataset.
#   3. RBF is a universal approximator that handles the non-linear separability
#      likely present in high-dimensional cytological measurements.

final_svm = SVC(
    kernel='rbf',
    C=best_C_rbf,
    gamma=best_gamma,
    random_state=42,
    probability=True,
)

t0 = time.perf_counter()
final_svm.fit(X_train_scaled, y_train)
final_train_time = time.perf_counter() - t0

y_pred_svm  = final_svm.predict(X_test_scaled)
y_proba_svm = final_svm.predict_proba(X_test_scaled)[:, 1]

print(f"\nFinal SVM model — kernel=RBF, C={best_C_rbf}, gamma={best_gamma}")
print(f"Training time    : {final_train_time:.4f} s")
print()
print("Classification report:")
print(classification_report(
    y_test,
    y_pred_svm,
    target_names=['Malignant', 'Benign'],
    digits=4
))

cm_svm = confusion_matrix(y_test, y_pred_svm)
print("Confusion matrix:")
print(cm_svm)
print(f"\nFinal test accuracy : {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Final AUC           : {roc_auc_score(y_test, y_proba_svm):.4f}")

# Confusion matrix plot
fig, ax = plt.subplots()
ax.grid(False)
ConfusionMatrixDisplay(
    confusion_matrix=cm_svm,
    display_labels=['Malignant', 'Benign']
).plot(cmap='Blues', values_format='d', ax=ax)
plt.title(
    f'SVM Confusion Matrix\n(RBF kernel, C={best_C_rbf}, gamma={best_gamma})',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task5_04_svm_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()


# ---
# ### 5.6 Full Kernel Summary Table (with tuned RBF)

# Add tuned RBF row to the summary
tuned_rbf_row = {
    'Kernel'        : f'RBF (tuned C={best_C_rbf}, γ={best_gamma})',
    'Accuracy'      : accuracy_score(y_test, y_pred_svm),
    'AUC'           : roc_auc_score(y_test, y_proba_svm),
    'Train time (s)': round(final_train_time, 4),
}

summary_rows = pd.DataFrame(kernel_results)[display_cols].copy()
summary_rows = pd.concat(
    [summary_rows, pd.DataFrame([tuned_rbf_row])],
    ignore_index=True
)

print("\nFull SVM summary (including tuned RBF):")
print(summary_rows.to_string(index=False))

print("\nTask 5 complete. Images saved to:", img_dir)
