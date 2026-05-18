# Machine Learning — Session 6: Support Vector Machine (SVM)

**Date:** 2026-04-10 (Friday)
**Instructor:** Alexandra Gómez Villa
**Duration:** ~47 min

---

## 1. Overview

This session covers **Support Vector Machines (SVM)** — described as "probably the best classifier in the entire course." Before the deep learning era (pre-2012), SVM was at the core of every state-of-the-art ML pipeline.

---

## 2. Key Concepts

### 2.1 SVM: The Best First Choice

> **Rule of thumb:** If you have a supervised ML problem with features already extracted, **try SVM first**.

- Pre-2012, SVM was the **state-of-the-art classifier** in every domain
- After 2012 (deep learning era), DL outperforms SVM but requires massive data
- For **small to medium datasets** with good features, SVM is still the best

### 2.2 The SVM Pipeline

- Features extracted by domain experts → Proper feature selection → SVM model
- SVM solves a **quadratic optimization problem**
- Uses **support vectors** (critical data points near decision boundary)

### 2.3 Kernels

| Kernel | When to Use |
|--------|-------------|
| **Linear** | Data is linearly separable |
| **Polynomial** | First non-linear option to try |
| **RBF (Gaussian)** | Most common — handles complex boundaries |
| **Sigmoid** | Similar to neural network activation |
| **Pre-computed** | Custom kernel function |

- **Kernel trick:** Transform data from lower dimension to **very high dimension** where it becomes separable
- Mathematics of kernel trick is **outside the scope** — understanding the concept is sufficient

### 2.4 Important Hyperparameters

| Parameter | Purpose | Typical Range |
|-----------|---------|---------------|
| **C** (slack) | Controls margin flexibility / misclassification cost | 1–10 (start with 1–10 loop) |
| **gamma** (RBF kernel) | Feature scaling in kernel | Often defaults work |
| **degree** (polynomial kernel) | Complexity of polynomial | 2–7 |

### 2.5 Hyperparameter Tuning Strategy

- Use a **for loop** with cross-validation for C values (1 to 10)
- For more thorough search: nested loops for **C × kernel** combinations
- **Do not** guess ranges — let the data tell you through cross-validation
- For large datasets (millions), subset the data for tuning

---

## 3. Session Notebook

- SVM implementation with scikit-learn
- Includes a **face recognition example** (more realistic case)
- Available on Campus Virtual

---

## 4. Session Timeline

| Section | Time | Content |
|---------|------|---------|
| **I. Introduction** | 0:05–2:16 | Pre-2012 dominance, first model to try |
| **II. Bias-Variance in SVM** | 2:16–38:39 | Maximal margin classifier, kernels, C parameter |
| **III. Kernel Details** | 38:39–41:23 | Options in scikit-learn (linear, poly, RBF, sigmoid) |
| **IV. Q&A — C Parameter** | 41:23–45:35 | How to choose C (for loop + CV) |
| **V. Key Concepts Summary** | 45:35–47:10 | SVM takeaways |
| **VI. Wrap-up & Notebook** | 47:10–47:56 | Notebook with face recognition example |

---

## 5. Key Takeaways

- **SVM** = best classifier for supervised problems with good features
- **Kernels** enable non-linear classification by transforming features to higher dimensions
- **C** controls the margin: low C = wider margin (more tolerant of misclassifications)
- **Always tune C and kernel** with cross-validation (for loop approach)
- SVM dominated ML pre-2012 and is still excellent for medical/small datasets
- Notebook with face recognition example is available

---

*Sources: Transcript `2026_04_10_Support Vector Machine_6.md`*
