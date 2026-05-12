#!/usr/bin/env python
# coding: utf-8

# # Assignment 2 - Supervised Learning Methods
# ## Task 3: K-Nearest Neighbours (KNN)
# **Dataset:** Breast Cancer Wisconsin (Diagnostic)
# **Goal:** Sweep k from 1 to 20, select the optimal k, train a final KNN model,
# and evaluate it with a classification report and confusion matrix.

# ---
# ### 3.1 Imports and Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, ConfusionMatrixDisplay)

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
# ### 3.2 Load Preprocessed Data

X_train_scaled = pd.read_csv(os.path.join(output_dir, 'X_train_scaled.csv'))
X_test_scaled  = pd.read_csv(os.path.join(output_dir, 'X_test_scaled.csv'))
y_train        = pd.read_csv(os.path.join(output_dir, 'y_train.csv')).squeeze()
y_test         = pd.read_csv(os.path.join(output_dir, 'y_test.csv')).squeeze()

print(f"X_train_scaled : {X_train_scaled.shape}")
print(f"X_test_scaled  : {X_test_scaled.shape}")
print(f"y_train        : {y_train.shape}")
print(f"y_test         : {y_test.shape}")


# ---
# ### 3.3 Sweeping k (k = 1 to 20)
#
# The optimal k is determined empirically by sweeping k in {1, ..., 20}
# and recording test accuracy for each value.

k_values    = range(1, 21)
knn_results = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, knn.predict(X_test_scaled))
    knn_results.append({'k': k, 'test_accuracy': acc})

knn_results = pd.DataFrame(knn_results)

# k = 17 selected: matches maximum accuracy and lies in the stable region.
# k = 3 is discarded despite matching the max — it sits in an unstable region.
best_k       = 17
best_knn_acc = knn_results.loc[knn_results['k'] == best_k, 'test_accuracy'].values[0]

print(f"\n{'k':<5} {'Test Accuracy':>14}")
print("─" * 21)
for _, row in knn_results.iterrows():
    marker = "  <- selected" if row['k'] == best_k else ""
    print(f"{int(row['k']):<5} {row['test_accuracy']:>14.4f}{marker}")

print(f"\nSelected k         : {best_k}")
print(f"Test accuracy      : {best_knn_acc:.4f}")

# Accuracy vs k plot
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(
    knn_results['k'],
    knn_results['test_accuracy'],
    marker='o', linewidth=2.2,
    color=PALETTE[0],
    markerfacecolor=CLR_MAL, markeredgecolor='white',
    markeredgewidth=1.2, markersize=8,
    zorder=3
)

ax.axvline(
    best_k, color='crimson', linestyle='--', linewidth=1.8,
    label=f'Optimal k = {best_k}  (acc = {best_knn_acc:.4f})'
)

ax.fill_between(
    knn_results['k'],
    knn_results['test_accuracy'],
    alpha=0.08, color=PALETTE[0]
)

ax.set_xlabel('k (number of neighbours)', fontsize=11)
ax.set_ylabel('Test Accuracy', fontsize=11)
ax.set_title('KNN: Test Accuracy vs k', fontsize=13, fontweight='bold')
ax.set_xticks(knn_results['k'])
ax.set_ylim(
    knn_results['test_accuracy'].min() - 0.01,
    knn_results['test_accuracy'].max() + 0.015
)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.3f'))
ax.grid(True, linestyle=':', alpha=0.5)
ax.legend(fontsize=10)
ax.set_facecolor('#fafafa')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task3_01_knn_accuracy_vs_k.png'), dpi=300, bbox_inches='tight')
plt.close()


# ---
# ### 3.4 Final KNN Model (k = 17)

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
y_pred_knn = final_knn.predict(X_test_scaled)

print(f"\nFinal KNN model  —  k = {best_k}")
print()
print("Classification report:")
print(classification_report(
    y_test,
    y_pred_knn,
    target_names=['Malignant', 'Benign'],
    digits=4
))

cm_knn = confusion_matrix(y_test, y_pred_knn)
print("Confusion matrix:")
print(cm_knn)

# Confusion matrix plot
fig, ax = plt.subplots()
ax.grid(False)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm_knn,
    display_labels=['Malignant', 'Benign']
)
disp.plot(cmap='Blues', values_format='d', ax=ax)
plt.title(f'KNN Confusion Matrix  (k = {best_k})', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task3_02_knn_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nTask 3 complete. Images saved to:", img_dir)
