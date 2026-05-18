# Machine Learning — Session 8: Neural Networks 2

**Date:** 2026-04-17 (Friday)
**Instructor:** Alexandra Gómez Villa
**Duration:** ~1h 05min

---

## 1. Overview

This second neural network session covers **activation functions, loss functions, optimizers, and regularization**. It also presents the **ImageNet challenge** as the pivotal moment for deep learning (2012), and announces **Assignment 2**.

---

## 2. Key Concepts

### 2.1 Activation Functions (Deep Dive)

| Function | Use Case |
|----------|----------|
| **ReLU** (default in scikit-learn) | Most common — fast, avoids vanishing gradient |
| **Sigmoid** | Binary classification output |
| **Tanh** | Older, similar to sigmoid but centered at 0 |

### 2.2 Loss Functions

| Task | Loss Function |
|------|---------------|
| **Binary classification** | Binary cross-entropy |
| **Multi-class classification** | Categorical cross-entropy |
| **Regression** | Mean Squared Error (MSE) |

### 2.3 Optimizers

| Optimizer | Description |
|-----------|-------------|
| **SGD** (Stochastic Gradient Descent) | Basic optimizer |
| **Adam** (default in scikit-learn) | Adaptive — most popular, works well generally |
| **Momentum** | Helps escape local minima |
| **Learning rate schedules** | Adjust learning rate during training |

### 2.4 Regularization

| Method | Description |
|--------|-------------|
| **L2** (Ridge) | Default in scikit-learn, penalizes large weights |
| **L1** (Lasso) | Can produce sparse weights |
| **Dropout** | Key for deep learning — randomly drops neurons during training |
| **Early stopping** | Stop training when validation performance stops improving |

> **Note:** Dropout is critical for deep learning but not common in small MLPs (scikit-learn doesn't implement it)

### 2.5 scikit-learn MLP Parameters

- Default activation: `relu`
- Default optimizer: `adam`
- Default learning rate: `0.001`
- Hyperparameters: batch size, momentum, beta1, beta2, max iterations

### 2.6 Deep Learning Threshold

- **Shallow:** 1–2 layers
- **Deep:** >2 layers (>4 layers for "truly deep")
- **ResNet (2016):** Up to 100+ layers

### 2.7 The ImageNet Moment (2012)

- **ImageNet:** 1 million images, 1,000 classes
- Pre-2012: incremental accuracy improvements year over year
- **2012 (AlexNet):** Huge accuracy jump — deep learning breakthrough
- **2015:** Surpassed **human error rate**
- Result: Deep learning now dominates ALL competitions

---

## 3. Assignment 2 Announcement

| Feature | Detail |
|---------|--------|
| **Topic** | Supervised Learning (classification) |
| **Dataset** | Breast Cancer Wisconsin |
| **Methods** | KNN, Logistic Regression, SVM, Neural Network |
| **Similar structure** | Same as Assignment 1 (notebook + report) |
| **Deadline** | May 13, 2026 |
| **Grades** | Assignment 1 grades to be posted next Tuesday |

---

## 4. Session Timeline

| Section | Time | Content |
|---------|------|---------|
| **I. Recap + Outline** | 0:04–1:33 | Neural networks overview |
| **II. Activation Functions** | 1:33–20:00 | ReLU, sigmoid, tanh |
| **III. Loss Functions** | 20:00–30:00 | Cross-entropy, MSE |
| **IV. Optimizers** | 30:00–45:00 | SGD, Adam, momentum |
| **V. Regularization** | 45:00–57:55 | L1, L2, dropout, early stopping |
| **VI. scikit-learn MLP** | 57:55–59:35 | Parameter walkthrough |
| **VII. ImageNet Story** | 59:35–1:02:15 | 2012 breakthrough, human-level accuracy |
| **VIII. Practical Advice** | 1:02:15–1:05:00 | Hyperparameter selection, Assignment 2 news |

---

## 5. Key Takeaways

- **ReLU** is the default activation; **Adam** is the default optimizer
- **Dropout** is essential for deep networks but not in simple MLPs
- **Never** tune hyperparameters on training data — use validation/CV
- ImageNet 2012 = deep learning revolution
- Assignment 2 (Supervised Learning) deadline: **May 13**
- No practice session — review the notebook from Session 7

---

*Sources: Transcript `2026_04_17_Neural Networks 2_8.md`*
