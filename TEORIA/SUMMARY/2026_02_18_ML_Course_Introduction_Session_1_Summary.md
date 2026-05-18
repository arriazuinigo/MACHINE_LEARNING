# Machine Learning - Session 1: Course Introduction Summary

**Date:** 2026-02-18 (Wednesday)
**Instructor:** Alexandra Gómez Villa
**Duration:** ~52 min 33s (0:19 – 52:33)
**Subject:** Machine Learning (Foundational Course)

> ⚠️ **Note:** The transcript contains a significant gap between **0:50** and **44:48** (~44 minutes missing). The core syllabus content, course structure overview, detailed AI/AI-vs-ML-vs-DL breakdown, schedule, and evaluation criteria were likely delivered during this interval but were not captured in the transcript.

---

## Course Evaluation & Assessment Structure (ML 2025-26)

### Overall Grade Breakdown

| Component | Weight | Deadline | Deliverables | Details |
|-----------|--------|----------|-------------|---------|
| **Assignment 1 — Unsupervised Learning** | **25%** | **30 March 2026** (23:59 CET) | Jupyter Notebook (.ipynb) + Written Report (.pdf) as `.zip` | Clustering patient data (Heart Disease UCI). K-Means, GMM, Hierarchical Clustering. **Team submission** (zip name: `TeamXX_UnsupervisedLearning.zip`) |
| **Assignment 2 — Supervised Learning** | **25%** | **13 May 2026** (23:59 CET) | Jupyter Notebook (.ipynb) + Written Report (.pdf) as `.zip` | Breast Cancer Wisconsin classification. KNN, Logistic Regression, SVM, Neural Network. **Team submission** (zip name: `TeamXX_SupervisedLearning.zip`) |
| **Other components** | **50%** | — | — | Not detailed in available transcripts |
| **Total** | **100%** | — | — | Grade split: 50% from assignments + 50% from other components |

> **Note:** Only 50% of the grade (2 assignments at 25% each) is documented in the available materials. The remaining 50% may consist of additional deliverables, exams, or participation not captured in these transcripts.

### Assignments Detail

| Feature | Assignment 1: Unsupervised Learning | Assignment 2: Supervised Learning |
|---------|-------------------------------------|----------------------------------|
| **Weight** | 25% of final grade | 25% of final grade |
| **Grading** | Out of 100 points (rubric-based) | Out of 100 points (rubric-based) |
| **Deadline** | 30 March 2026 (23:59 CET) | 13 May 2026 (23:59 CET) |
| **Dataset** | Heart Disease UCI (processed.cleveland.data) — 303 patients, 13 features | Breast Cancer Wisconsin (sklearn) — 569 patients, 30 features |
| **Methods covered** | K-Means, GMM, Hierarchical Clustering | KNN, Logistic Regression, SVM (linear/RBF/poly), Neural Network (MLP) |
| **Team format** | 4–5 members per group | Same teams (presumably) |
| **Deliverables** | Jupyter Notebook + Written Report (1,500–2,500 words) | Jupyter Notebook + Written Report (1,500–2,500 words) |

### Assignment 1 Grading Rubric (100 pts total → 25% of final grade)

| Section | Points | Criteria |
|---------|--------|----------|
| **Task 1: Data Exploration** | 10 | Loading & shape (2), Distributions (4), Correlation analysis (4) |
| **Task 2: Preprocessing** | 15 | Encoding & scaling (4), PCA (6), Feature selection (5) |
| **Task 3: K-Means** | 15 | Elbow & silhouette (6), Final model & viz (5), Cluster characterisation (4) |
| **Task 4: GMM** | 15 | BIC/AIC selection (6), Visualization (5), Comparison w/ K-Means (4) |
| **Task 5: Hierarchical Clustering** | 15 | Dendrogram (6), Cut & viz (5), Structural comparison (4) |
| **Task 6: Evaluation** | 15 | ARI scores (9), Confusion matrix & discussion (6) |
| **Task 7: Written Report** | 15 | Methodology (4), Algorithm comparison (5), Reflection (4), Clarity (2) |
| **TOTAL** | **100** | |

### Assignment 2 Grading Rubric (100 pts total → 25% of final grade)

| Section | Points | Criteria |
|---------|--------|----------|
| **Task 1: Data Exploration** | 10 | Loading & shape (2), Distributions (4), Correlation analysis (4) |
| **Task 2: Preprocessing** | 10 | Scaling & split (5), Feature analysis (5) |
| **Task 3: KNN** | 15 | k sweep (6), Final model (5), Interpretation (4) |
| **Task 4: Logistic Regression** | 15 | Regularisation sweep (6), Final model (5), Coefficient analysis (4) |
| **Task 5: SVM** | 15 | Kernel comparison (7), Final model (5), Hyperparameter effect (3) |
| **Task 6: Neural Network** | 15 | Architecture exploration (6), Learning curves (5), Final model (4) |
| **Task 7: Model Comparison** | 10 | ROC curves (5), Summary table & discussion (5) |
| **Task 8: Written Report** | 10 | Methodology (3), Algorithm comparison (4), Reflection (3) |
| **TOTAL** | **100** | |

### Course Calendar — Sessions by Topic

| Week/Block | Date | Topic | Type |
|------------|------|-------|------|
| **Block 1: Unsupervised Learning** | | | |
| Session 1 | Feb 18 (Wed) | **Course Introduction** — AI/ML/DL overview, course structure, why ML in healthcare | Theory |
| Session 2 | Feb 20 (Fri) | Data Exploration & Visualization — ML pipeline, data collection, EDA, Gaussian distributions | Theory + Notebook |
| Session 3 | Mar 11 (Wed) | Clustering: Hierarchical & GMM — Assignment 1 released | Theory |
| Session 4 | Mar 13 (Fri) | Supervised Learning: Linear Regression — Bias-variance tradeoff; Assignment 1 deadline: **Mar 30** | Theory |
| **Block 2: Supervised Learning** | | | |
| Session 5 | Apr 8 (Wed) | Decision Trees & Random Forest intro — Microsoft Kinect case study | Theory + Notebook |
| Session 6 | Apr 10 (Fri) | Support Vector Machine — Kernels, C, gamma | Theory + Notebook |
| Session 7 | Apr 15 (Wed) | Neural Networks 1 — MLP, activation functions, training | Theory |
| Session 8 | Apr 17 (Fri) | Neural Networks 2 — Regularization, loss functions, optimizers; Assignment 2 announced | Theory |
| **Block 3: Time Series** | | | |
| Session 9 | Apr 22 (Wed) | Time Series Introduction | Theory |
| Session 10 | Apr 24 (Fri) | ARIMA Models | Theory |
| **Block 4: Advanced Topics** | | | |
| Session 11 | Apr 29 (Wed) | Markov Chains; Grades for Assignment 1 posted | Theory |
| Session 12 | May 8 (Fri) | Hidden Markov Models; Assignment 2 deadline: **May 13** | Theory |

> **Note:** Each session has an accompanying Python notebook on the virtual campus for practical work.

---

## Session Structure Overview

| Section | Time Range | Duration | % of Captured | Description |
|---------|-----------|----------|---------------|-------------|
| **I. Course Welcome & Introduction** | 0:19 – 0:50 | ~31 s | ~2% | Welcome, ML as foundational course, relation to Deep Learning & NLP |
| *[GAP — Core content missing]* | *0:50 – 44:48* | *~44 min* | — | *Likely covered: Syllabus details, AI/ML/DL hierarchy, course structure, evaluation, schedule, why healthcare needs ML* |
| **II. AI vs ML vs Deep Learning** | 44:48 – 46:34 | ~1 min 46 s | ~11% | AI hierarchy; DL successes (ChatGPT, NLP, generative models) but requires huge data |
| **III. ML in Healthcare Challenges** | 46:34 – 48:35 | ~2 min 1 s | ~13% | Privacy, limited public datasets, high-risk nature of medicine |
| **IV. Course Focus: ML + DL as Feature Extractors** | 48:35 – 49:58 | ~1 min 23 s | ~9% | Focus on ML; using pre-trained DL models as feature extractors (e.g., 512-vector) |
| **V. Q&A — Histogram of Oriented Gradients (HOG)** | 49:58 – 51:40 | ~1 min 42 s | ~11% | José Pablo asks about melanoma detection method; HOG for pedestrian detection |
| **VI. Wrap-up & Session 2 Preview** | 51:40 – 52:33 | ~53 s | ~6% | Friday: Gaussian distributions, data description; notebook provided |

**Total captured content:** ~8 min 44 s (due to ~44 min gap)

---

## Detailed Summary

### I. Course Welcome & Introduction (0:19 – 0:50)

- **Welcome:** Alexandra Gómez Villa welcomes students to the Machine Learning course.
- **Foundational course:** ML is described as one of the main courses/topics students will use throughout their careers in data.
- **Connection to other courses:** The tools learned here will be used in **Introduction to Deep Learning** and **Natural Language Processing**.

---

### II. AI vs ML vs Deep Learning (44:48 – 46:34)

#### The Hierarchy of AI

```
Artificial Intelligence (AI)
  └── Machine Learning (ML) ← Our course focus
        └── Deep Learning (DL)
```

- **Machine Learning** is a subfield of AI — using data and models to learn about data distributions.
- **Deep Learning** is a subfield of ML.

#### Why Deep Learning Dominates Today

- Most successful AI applications use DL: **ChatGPT**, **NLP systems**, **generative models** (text-to-image), **predictive economics**.
- DL automatically learns features and solves problems more effectively.
- **BUT** DL requires **huge amounts of data** to work — this is a critical limitation.

---

### III. ML in Healthcare Challenges (46:34 – 48:35)

Three main reasons why DL has limited impact in medicine:

1. **Privacy:** Patient data is protected; cannot be disclosed to third parties or companies.
2. **Limited public datasets:** Due to privacy constraints, there are no large public medical datasets.
3. **High-risk domain:** Mistakes in healthcare have serious consequences (e.g., misdiagnosing lung cancer on an X-ray). Unlike generating a fun image with a "third eye" that can be laughed off, medical errors are unacceptable.

> **Current approach in medicine:** Most secure medical applications use a **mix of ML and DL** — not pure DL.

---

### IV. Course Focus: ML + DL as Feature Extractors (48:35 – 49:58)

- **Primary focus:** Machine Learning (not training DL models from scratch).
- **How DL is used:** Pre-trained DL models ("foundation models") serve as **feature extractors**:
  - E.g., an image goes through a DL model → outputs a **512-dimensional vector**.
  - This vector is then used to train traditional ML models.
- **Last part of the course:** Will cover how to extract features from DL models and use them.
- **Goal:** Prepare students for the Deep Learning course (if taken later).

---

### V. Q&A — Histogram of Oriented Gradients (HOG) (49:58 – 51:40)

- **Question from José Pablo Soriano Torres:** What method was used for melanoma detection mentioned earlier?

- **Answer — HOG (Histogram of Oriented Gradients):**
  - Was the **state-of-the-art** before the deep learning era (pre-2012).
  - Originally used for **pedestrian detection** in videos and images.
  - Very effective for detecting **shapes** in images (e.g., melanoma lesions).
  - Useful in medical imaging when you have specific shape patterns to detect.

---

### VI. Wrap-up & Session 2 Preview (51:40 – 52:33)

- **Session 2 (Friday, Feb 20):** Will begin with the basics:
  - **Describing data** — data points, Gaussian distributions.
  - Fundamental concepts that form the base for everything learned later.
  - A **Python notebook** will be provided for hands-on practice with numbers and visualizations.
- **Office hours:** Instructor is available via email for questions.
- **Student request (Ecem Onat Cakmak):** Asked for presentations to be uploaded before class for note-taking. Instructor agreed to upload them just before class.

---

## Key Concepts Introduced

| Concept | Description |
|---------|-------------|
| **AI / ML / DL Hierarchy** | AI ⊃ ML ⊃ DL — deep learning is a subset of machine learning |
| **Deep Learning limitations** | Requires massive data, which is often unavailable in healthcare due to privacy |
| **Healthcare ML challenges** | Privacy laws, limited public datasets, high-risk consequences of errors |
| **DL as feature extractor** | Pre-trained DL models produce feature vectors for ML models |
| **HOG (Histogram of Oriented Gradients)** | Pre-DL state-of-the-art for shape detection in images (pedestrians, melanoma) |
| **Foundational course** | ML skills transfer to Deep Learning and NLP courses |
| **Mixed ML/DL approach** | Most secure medical applications blend traditional ML with pre-trained DL features |

---

## Preparation for Session 2

Session 2 (Feb 20 — Data Exploration & Visualization) will cover:
- The **basic machine learning pipeline** (data collection → preparation → feature selection → model training)
- **Data exploration techniques**: histograms, box plots, scatter plots, correlation heatmaps
- **Gaussian distributions** as a foundation
- Practical work with a **Python notebook** available on the virtual campus

---

## Academic Integrity Notes

- **Teamwork:** 4–5 members per group. Tasks can be divided, but joint review session required before submission.
- **Allowed libraries:** scikit-learn, NumPy, pandas, matplotlib, seaborn, scipy.
- **AI tools:** May be used for syntax assistance, but **analytical decisions** and **written report** must be the students' own work. Must indicate if AI tools were used.
- **Key requirement:** Notebook must run **end-to-end without errors** as a minimum for full marks.

---

*Generated from transcript: `2026_02_18_Course_intro_1.md` and assignment PDFs `MHEDAS_Assignment1_UnsupervisedLearning.pdf` & `MHEDAS_Assignment2_SupervisedLearning-2.pdf`*

> **Data completeness note:** ~44 minutes of the lecture (the syllabus and detailed course structure portion) were missing from the transcript. The evaluation information was reconstructed from the official assignment PDFs found in the course materials.
