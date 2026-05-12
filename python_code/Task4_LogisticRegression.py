#!/usr/bin/env python
# coding: utf-8

# # Assignment 2 - Supervised Learning Methods
# ## Task 4: Logistic Regression
# **Dataset:** Breast Cancer Wisconsin (Diagnostic)
# **Goal:** Sweep the regularisation parameter C, select the optimal value,
# train a final Logistic Regression model, evaluate it, and interpret the coefficients.

# ---
# ### 4.1 Imports and Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import warnings
import os

warnings.filterwarnings('ignore')

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
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
# ### 4.2 Load Preprocessed Data

X_train_scaled = pd.read_csv(os.path.join(output_dir, 'X_train_scaled.csv'))
X_test_scaled  = pd.read_csv(os.path.join(output_dir, 'X_test_scaled.csv'))
y_train        = pd.read_csv(os.path.join(output_dir, 'y_train.csv')).squeeze()
y_test         = pd.read_csv(os.path.join(output_dir, 'y_test.csv')).squeeze()

# Load original feature names for coefficient analysis
data = load_breast_cancer()
feature_names = data.feature_names

print(f"X_train_scaled : {X_train_scaled.shape}")
print(f"X_test_scaled  : {X_test_scaled.shape}")
print(f"y_train        : {y_train.shape}")
print(f"y_test         : {y_test.shape}")


# ---
# ### 4.3 Sweeping the Regularisation Parameter C
#
# The model is trained for each of 10 logarithmically spaced values of C
# from 10^-3 to 10^3. Smaller C = stronger regularisation.

C_values = np.logspace(-3, 3, 10)

lr_results = []

for C in C_values:
    model = LogisticRegression(
        C=C,
        solver='liblinear',
        max_iter=5000,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    lr_results.append({
        'C': C,
        'test_accuracy': accuracy_score(y_test, y_pred)
    })

lr_results = pd.DataFrame(lr_results)

best_accuracy = lr_results['test_accuracy'].max()
best_C        = lr_results.loc[lr_results['test_accuracy'] == best_accuracy, 'C'].min()

print(lr_results.to_string(index=False))
print(f"\nBest accuracy : {best_accuracy:.4f}")
print(f"Selected C    : {best_C:.6f}")

# Accuracy vs C plot
plt.figure(figsize=(9, 5))

plt.plot(
    lr_results['C'],
    lr_results['test_accuracy'],
    marker='o',
    linewidth=2,
    color=PALETTE[0],
    markerfacecolor=CLR_MAL, markeredgecolor='white',
    markeredgewidth=1.2, markersize=8,
)

plt.axvline(best_C, color='red', linestyle='--',
            label=f'Best C = {best_C:.3g}  (acc = {best_accuracy:.4f})')
plt.xscale('log')
plt.xlabel('C (regularisation strength, log scale)', fontsize=11)
plt.ylabel('Test Accuracy', fontsize=11)
plt.title('Logistic Regression: Test Accuracy vs C', fontsize=13, fontweight='bold')
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task4_01_lr_accuracy_vs_C.png'), dpi=300, bbox_inches='tight')
plt.close()


# ---
# ### 4.4 Final Logistic Regression Model (C = best_C)

final_lr = LogisticRegression(
    C=best_C,
    solver='liblinear',
    max_iter=5000,
    random_state=42
)

final_lr.fit(X_train_scaled, y_train)
y_pred_lr = final_lr.predict(X_test_scaled)

print("\nClassification report:")
print(classification_report(
    y_test,
    y_pred_lr,
    target_names=['Malignant', 'Benign'],
    digits=4
))

cm = confusion_matrix(y_test, y_pred_lr)
print("Confusion matrix:")
print(cm)

# Confusion matrix plot
fig, ax = plt.subplots()
ax.grid(False)
ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Malignant', 'Benign']
).plot(cmap='Blues', values_format='d', ax=ax)
plt.title('Logistic Regression Confusion Matrix', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task4_02_lr_confusion_matrix.png'), dpi=300, bbox_inches='tight')
plt.close()


# ---
# ### 4.5 Coefficient Interpretation
#
# Target encoding: 0 = malignant, 1 = benign.
# LogisticRegression coefficients point toward class 1 (benign).
# Negative coefficients therefore support malignancy.

coef_df = pd.DataFrame({
    'feature'               : feature_names,
    'coefficient_for_benign': final_lr.coef_[0]
})

coef_df['malignancy_weight'] = -coef_df['coefficient_for_benign']
coef_df['abs_coefficient']   =  coef_df['coefficient_for_benign'].abs()

top_malignant = coef_df.sort_values('malignancy_weight', ascending=False).head(10)
top_overall   = coef_df.sort_values('abs_coefficient',   ascending=False).head(15)

print("\nTop 10 features predicting malignancy (highest malignancy weight):")
print(top_malignant[['feature', 'malignancy_weight', 'coefficient_for_benign']].to_string(index=False))

print("\nTop 15 largest absolute coefficients:")
print(top_overall[['feature', 'coefficient_for_benign', 'abs_coefficient']].to_string(index=False))

# Coefficient bar chart
fig, ax = plt.subplots(figsize=(12, 7))

coef_sorted = coef_df.sort_values('coefficient_for_benign')
colors = [CLR_MAL if v < 0 else CLR_BEN for v in coef_sorted['coefficient_for_benign']]

ax.barh(
    coef_sorted['feature'],
    coef_sorted['coefficient_for_benign'],
    color=colors, height=0.65,
    edgecolor='white', linewidth=0.7, alpha=0.88
)

ax.axvline(0, color='#333333', linewidth=1.0)
ax.set_xlabel('Coefficient (positive = benign, negative = malignant)', fontsize=11)
ax.set_title(
    f'Logistic Regression Coefficients  (C = {best_C:.3g})\n'
    'Red = supports malignant prediction, Blue = supports benign prediction',
    fontsize=12, fontweight='bold'
)
ax.tick_params(axis='y', labelsize=8.5)
ax.set_facecolor('#fafafa')

plt.tight_layout()
plt.savefig(os.path.join(img_dir, 'task4_03_lr_coefficients.png'), dpi=300, bbox_inches='tight')
plt.close()

print("\nTask 4 complete. Images saved to:", img_dir)
