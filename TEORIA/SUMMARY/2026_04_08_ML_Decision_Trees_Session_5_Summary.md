# Machine Learning — Session 5: Decision Trees

**Date:** 2026-04-08 (Wednesday)
**Instructor:** Alexandra Gómez Villa
**Duration:** ~52 min

---

## 1. Overview

This session covers **decision trees** — a supervised learning method for both classification and regression. It also provides a brief introduction to **Random Forest** (ensemble of decision trees), with the **Microsoft Kinect** case study as a real-world example.

---

## 2. Key Concepts

### 2.1 Decision Tree Basics

- Tree structure: **internal nodes** = features, **branches** = decisions, **leaves** = predictions
- **Goal:** Find the best splits to maximize purity at each node
- Can be used for both **classification** and **regression**

### 2.2 Splitting Criteria

| Criterion | Description |
|-----------|-------------|
| **Information Gain** | Based on entropy reduction |
| **Gini Impurity** | Probability of incorrect classification |
| **Variance Reduction** | For regression trees |

### 2.3 Stopping Criteria & Pruning

- **Depth** (maximum tree depth) is the main stopping criterion
- **Pruning** = making the tree shallower to prevent overfitting
- The instructor recommends letting the library (scikit-learn) manage pruning

### 2.4 Random Forest (Ensemble Method)

- **Trains multiple decision trees** on different subsets of data
- Each tree is a different "expert"
- Combines decisions via **majority vote** (classification) or averaging (regression)
- One of the most successful models in **data competitions**

### 2.5 Case Study: Microsoft Kinect (2007)

- Microsoft Kinect: an **edge device** that captures video and predicts body skeleton
- Uses a **depth map** (distance of surface to sensor) — darker = closer
- How it works:
  1. Collect dataset: millions of people in front of sensor
  2. Humans manually annotate joint positions in the depth images
  3. Train a **Random Forest of 3 trees, depth 20**
  4. Trained on **1 million images** using a **1,000-core cluster**
  5. Runs at **200 frames per second** on the device
- This was a **paradigm shift** from Nintendo Wii (accelerometer-based) to controller-free interaction

---

## 3. Session Notebook

- Python notebook on Campus Virtual for practicing decision trees with scikit-learn
- Short and straightforward

---

## 4. Session Timeline

| Section | Time | Content |
|---------|------|---------|
| **I. Welcome Back** | 0:18–0:53 | After Easter vacation, grading update |
| **II. Decision Trees** | 0:53–44:19 | Splitting criteria, purity, depth, pruning |
| **III. Random Forest Intro** | 44:19–48:09 | Ensemble methods, majority voting |
| **IV. Kinect Case Study** | 48:13–52:10 | Depth maps, skeleton prediction, 200 fps |
| **V. Wrap-up** | 52:10–52:52 | Notebook, questions |

---

## 5. Key Takeaways

- **Decision trees** split data based on features to increase purity
- **Information gain** and **Gini impurity** are common splitting criteria
- **Random Forest** = ensemble of trees → more robust than single tree
- Kinect uses Random Forest for **real-time skeleton tracking** (200 fps)
- Decision trees/Random Forest are **extremely powerful** despite their simplicity
- Notebook available on Campus Virtual

---

*Sources: Transcript `2026_04_08_Decision Trees_5.md`*
