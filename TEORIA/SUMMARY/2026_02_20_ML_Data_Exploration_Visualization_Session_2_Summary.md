# Machine Learning — Session 2: Data Exploration & Visualization

**Date:** 2026-02-20 (Friday)
**Instructor:** Alexandra Gómez Villa
**Duration:** ~57 min

---

## 1. Overview

This session covers the **basic machine learning pipeline** steps of **data collection, preparation, and exploratory data analysis (EDA)**. The instructor demonstrates these concepts using a **fruit classification** example (apples, oranges, lemons, mandarins).

---

## 2. Key Concepts

### 2.1 The ML Pipeline (Data Collection Phase)

- **Data collection** is typically handled by domain experts (doctors, nurses), not ML engineers
- When starting a project, the ML expert may need to **guide domain experts** on what data is useful
- Often, a dataset already exists and work begins at data preparation

### 2.2 Fruit Classifier Example

**Dataset variables:**
- Fruit label, fruit name, mass, width, color score
- Main focus: **mass** and **width** as features

### 2.3 Visualization Techniques

| Technique | Purpose |
|-----------|---------|
| **Box plots** | Identify outliers per class |
| **Scatter plots** | Distribution of features |
| **Histograms** | Distribution of individual variables |
| **Correlation heatmaps** | Feature correlations |

- Outliers can be detected visually (e.g., some fruit classes had outliers in box plots)
- Good **visualization skills** and **basic statistics** are essential

---

## 3. Notebook & Tools

### 3.1 Python Notebooks

- Each theory session has a **corresponding Python notebook** on Campus Virtual
- Students can work:
  - **Locally** with VS Code
  - **Online** with Google Colab (free resource from Google)
- The fruit dataset is downloaded from **GitHub** (free, publicly available)

### 3.2 Google Colab Demo

- Colab is an **online Python notebook** environment
- Runs on a **virtual machine in the cloud**
- Useful for learning, not for big projects
- Students need a Google account
- Data must be uploaded to Colab's file system

### 3.3 Session Notebook

- The notebook explores fruit dataset features
- Students are encouraged to explore visualizations, draw conclusions
- Focus on understanding feature behavior

---

## 4. Session Timeline

| Section | Time | Content |
|---------|------|---------|
| **I. ML Pipeline Overview** | 0:03–1:50 | Data collection roles, pipeline phases |
| **II. Fruit Classifier Example** | 1:50–50:09 | Visualization techniques, box plots, scatter plots |
| **III. Notebook & Tools** | 50:09–56:51 | Python notebooks, Google Colab demo, fruit dataset |
| **IV. Wrap-up** | 56:51–57:40 | Invitation to explore notebook, good weekend |

---

## 5. Key Takeaways

- **Data preparation** is often the most time-consuming part of ML
- **Visualization** is critical for understanding data before modeling
- Outliers, class separability, and feature distributions should be examined early
- Python notebooks (local or Google Colab) are provided for each session
- Session notebooks are practical, not overly complex

---

*Sources: Transcript `2026_02_20_Data_exploration_and_visualization_2.md`*
