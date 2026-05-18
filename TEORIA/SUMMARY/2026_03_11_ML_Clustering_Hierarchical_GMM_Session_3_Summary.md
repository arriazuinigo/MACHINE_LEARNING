# Machine Learning — Session 3: Clustering — Hierarchical & GMM

**Date:** 2026-03-11 (Wednesday)
**Instructor:** Alexandra Gómez Villa
**Duration:** ~56 min

---

## 1. Overview

This session concludes the **Unsupervised Learning block** (Block 1 of 4). It covers **Hierarchical Clustering** and **Gaussian Mixture Models (GMM)**. The session also introduces **Assignment 1**.

---

## 2. Key Concepts

### 2.1 Hierarchical Clustering

- Builds a **tree-like structure** (dendrogram) of clusters
- Two approaches:
  - **Agglomerative** (bottom-up): Each point starts as its own cluster, merge closest pairs
  - **Divisive** (top-down): Start with one cluster, recursively split
- Good for **nested/structured data** where hierarchy matters

### 2.2 Gaussian Mixture Models (GMM)

- Assumes data is generated from a **mixture of Gaussian distributions**
- Each cluster is represented by a Gaussian with its own mean and covariance
- Soft clustering: each sample has a **probability of belonging to each cluster**
- Similar to K-Means but:
  - Handles **non-spherical clusters**
  - Provides **probabilistic assignments** (not hard labels)
  - Useful for **high-dimensional spaces**

### 2.3 Clustering Algorithm Selection Guide

| Algorithm | Best For |
|-----------|----------|
| **K-Means** | Compact, spherical clusters (start here) |
| **DBSCAN / HDBSCAN** | Arbitrary shapes, handles noise |
| **Hierarchical Clustering** | Nested/hierarchical structure |
| **Gaussian Mixture Models** | High-dimensional, probabilistic |

### 2.4 Clustering Evaluation Metrics

| Type | Metrics | Description |
|------|---------|-------------|
| **Internal** (no labels) | Silhouette score, Davies-Bouldin index | Quality based on cluster cohesion/separation |
| **External** (need labels) | ARI, Normalized Mutual Information | Agreement with ground truth |
| **Visual** | T-SNE, PCA, UMAP | Dimensionality reduction for visualization |

### 2.5 Practical Advice

- **Always start with K-Means** (simplest)
- Experiment with **multiple algorithms**
- Use **silhouette score** and **visual inspection** (PCA, T-SNE)
- Compute both internal metrics (silhouette) and external metrics (ARI, NMI)

---

## 3. Assignment 1 Details

- **Dataset:** Heart Disease UCI (processed.cleveland.data)
- **Focus:** Clustering without using the target labels
- **Tasks:**
  1. Data loading, EDA (distributions, box plots, correlations)
  2. Preprocessing: encode categorical, normalize, PCA, feature selection
  3. K-Means clustering (elbow method, silhouette)
  4. GMM clustering (BIC/AIC)
  5. Hierarchical clustering (dendrogram)
  6. Evaluation with ARI
  7. Written report (max 2,500 words)
- **Deadline:** March 30, 2026
- **Team format:** 4–5 members
- **Deliverable:** Jupyter Notebook + PDF report (zipped)

### Key Report Advice

- **Coding is important**, but **analysis/conclusions matter more**
- Divide tasks among team members, have a **joint review session**
- Report should focus on: which model is better, what observations in data
- No coding questions (not a coding course) — ML questions are OK

---

## 4. Session Timeline

| Section | Time | Content |
|---------|------|---------|
| **I. Introduction** | 0:31–1:25 | Completing Unsupervised Learning block |
| **II. Hierarchical Clustering** | 1:25–1:07 | Strategies, dendrograms |
| **III. Gaussian Mixture Models** | 1:07–47:26 | GMM concepts, probabilistic clustering |
| **IV. Algorithm Selection Guide** | 47:26–49:34 | When to use each algorithm |
| **V. Assignment 1** | 49:46–56:57 | Dataset, tasks, rubric, advice |

---

## 5. Key Takeaways

- **Hierarchical clustering** produces dendrograms (nested structure)
- **GMM** provides soft (probabilistic) clustering for non-spherical data
- Always **experiment with multiple algorithms**
- **Silhouette score** (internal) + **ARI** (external) for evaluation
- Assignment 1 deadline: **March 30**
- Report quality matters more than code

---

*Sources: Transcript `2026_03_11_Clustering: Hierarchical and GMM_3.md`*
