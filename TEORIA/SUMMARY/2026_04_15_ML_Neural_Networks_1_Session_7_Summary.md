# Machine Learning — Session 7: Neural Networks 1

**Date:** 2026-04-15 (Wednesday)
**Instructor:** Alexandra Gómez Villa
**Duration:** ~47 min

---

## 1. Overview

This session introduces **neural networks (MLP — Multi-Layer Perceptron)** — the core architecture behind deep learning. The session covers the basic structure, forward pass, activation functions (sigmoid creating non-linear decision boundaries), and backpropagation.

---

## 2. Key Concepts

### 2.1 Why Neural Networks Matter

- Neural networks are the **foundation of deep learning**
- They matter **only because of deep learning**
- For standard ML problems, **SVM is still better** than a small neural network
- Deep learning becomes powerful with **many layers** (deep architectures)

### 2.2 Neural Network Structure (MLP)

**Components:**
1. **Input layer** — receives features
2. **Hidden layers** — intermediate processing (can be one or many)
3. **Output layer** — produces prediction (classification or regression)

**How it works:**
- Each neuron: weighted sum of inputs + bias → activation function
- Non-linear activation functions (e.g., sigmoid) enable **non-linear decision boundaries**
- Multiple layers allow learning of **hierarchical features**

### 2.3 Activation Functions

- **Sigmoid** — squashes output to (0,1), good for probabilities
- Non-linear activations are essential for learning complex patterns

### 2.4 Backpropagation

- The algorithm for training neural networks
- Computes gradients of the loss with respect to each weight
- Given the course scope, the instructor **does not require** implementing backpropagation from scratch
- In practice, use **scikit-learn, PyTorch, or TensorFlow** implementations

### 2.5 Binary vs Multi-Class Classification

- **Binary:** Single output neuron (0 or 1)
- **Multi-class:** Multiple output neurons (one per class)
- Neural networks can also be used for **regression** (continuous output)

### 2.6 Importance of Understanding Neural Networks

- Essential foundation for the **Deep Learning course**
- Even if not the best for small ML problems, understanding neural networks is critical
- They form the basis of **all modern AI** (GPT, image recognition, etc.)

---

## 3. Session Notebook

- Backpropagation code is **provided** (not student-implemented)
- Focus on using **scikit-learn MLP** for classification
- Practice: train one-layer → multi-layer → predict classes
- Dataset is the same as decision trees session

---

## 4. Session Timeline

| Section | Time | Content |
|---------|------|---------|
| **I. Recap & Continuation** | 0:14–0:51 | Reviewing previous concepts, bias term |
| **II. Q&A — Weight/Bias Mixup** | 40:48–42:14 | Clarifying slide error (0.7 vs 0.8) |
| **III. Q&A — Sigmoid & Boundaries** | 42:17–43:09 | How sigmoid creates non-linear boundaries |
| **IV. Notebook & Backpropagation** | 43:20–46:08 | Using scikit-learn, not implementing from scratch |
| **V. SVM vs Neural Networks** | 46:12–47:10 | SVM still better for standard ML |
| **VI. Wrap-up** | 47:10–47:36 | Homework: review notebook |

---

## 5. Key Takeaways

- **Neural networks** = input → hidden layers → output (with non-linear activations)
- **Sigmoid** enables non-linear decision boundaries
- **Backpropagation** is the training algorithm (use libraries, don't implement)
- Neural networks are **important for deep learning**, not for standard small-data ML
- **SVM > Neural Network** for typical ML problems in this course
- Notebook for practice is available on Campus Virtual

---

*Sources: Transcript `2026_04_15_Neural Networks_7.md`*
