# Machine Learning — Session 4: Supervised Learning — Linear Regression

**Date:** 2026-03-13 (Friday)
**Instructor:** Alexandra Gómez Villa
**Duration:** ~52 min

---

## 1. Overview

This session begins **Block 2: Supervised Learning**. It covers the definition of supervised learning, the two types (regression vs classification), **linear regression** as the most basic regression model, and the critical **bias-variance tradeoff**.

---

## 2. Key Concepts

### 2.1 Supervised Learning vs Unsupervised Learning

- **Unsupervised:** No target labels — find patterns/groups in data
- **Supervised:** Have target labels/ground truth — model learns to predict output from input

**Two types of supervised learning:**
| Type | Target | Example |
|------|--------|---------|
| **Regression** | Continuous value | Predicting price, temperature |
| **Classification** | Discrete class | Disease vs healthy |

### 2.2 Linear Regression

- The most basic regression model
- Fits a **line** (or hyperplane) to the data: $y = wx + b$
- Two solution methods:
  1. **Least Squares** — closed-form solution
  2. **Gradient Descent** — iterative optimization

### 2.3 Bias-Variance Tradeoff

| Error Source | Problem | Description |
|-------------|---------|-------------|
| **Bias** | Underfitting | Model is too simple, doesn't capture data patterns |
| **Variance** | Overfitting | Model memorizes training data, fails on new data |

**The goal:** Find the right balance between bias and variance.

### 2.4 Training/Test Split

- Split data: ~70% training, ~30% test
- **Training set:** Adjust model parameters
- **Test set:** Evaluate generalization (simulates real-world deployment)
- If test performance is low → **overfitting** (high variance)

### 2.5 Cross-Validation (K-Fold)

**Purpose:** Test how well the model generalizes (independent of which samples are chosen for training).

**How it works:**
1. Split data into **K folds** (e.g., K=5)
2. Train on K-1 folds, test on the remaining fold
3. Rotate and repeat (K times)
4. Report **average performance** across all folds

**Benefits:**
- More robust evaluation than a single train/test split
- Helps select hyperparameters that balance bias and variance
- Each sample is used for both training and testing

**Limitations:**
- Computationally expensive (train K times per hyperparameter)
- Not feasible for deep learning (training takes days/months)
- Still feasible for **small/medium medical datasets**

### 2.6 Hyperparameter Tuning with Cross-Validation

- For each hyperparameter value (e.g., learning rate = 1, 2, 3...):
  - Run K-fold cross-validation
  - Record average accuracy
- The best value is the one with highest average accuracy
- Multiple nested loops can explore combinations (learning rate × kernel, etc.)

---

## 3. Session Notebook

- Available on Campus Virtual
- Practice: linear regression with gradient descent and least squares
- Explore how the line changes with different learning rates

---

## 4. Session Timeline

| Section | Time | Content |
|---------|------|---------|
| **I. Supervised Learning Intro** | 0:03–2:17 | Definition, regression vs classification |
| **II. Linear Regression** | 2:17–41:44 | Least squares, gradient descent |
| **III. Bias-Variance Tradeoff** | 41:44–45:33 | Underfitting (bias) vs overfitting (variance) |
| **IV. Cross-Validation** | 45:33–52:00 | K-fold CV, hyperparameter tuning |
| **V. Wrap-up** | 52:00–52:51 | Notebook, assignment reminder |

---

## 5. Key Takeaways

- **Supervised learning** needs labeled data (target/ground truth)
- **Linear regression** = simplest regression model (line fitting)
- **Bias** = underfitting; **Variance** = overfitting — find the balance
- **Cross-validation** is the best way to evaluate generalization on small datasets
- Assignment 1 reminder: deadline **March 30**

---

*Sources: Transcript `2026_03_13_Supervised Learning: Linear Regression_4.md`*
