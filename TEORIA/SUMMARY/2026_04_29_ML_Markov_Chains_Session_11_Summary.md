# Machine Learning — Session 11: Markov Chains

**Date:** 2026-04-29 (Wednesday)
**Instructor:** Alexandra Gómez Villa
**Duration:** ~47 min

---

## 1. Overview

This session begins **Block 4: Advanced Topics** with **Markov Chains** — a model for sequential systems with discrete states. This is a foundation topic leading to **Hidden Markov Models** (next session). The session also discusses **Assignment 1 grades**.

---

## 2. Key Concepts

### 2.1 Assignment 1 Grade Comments

- Overall results were **very good**
- **Feedback:** Many submissions had excessive text (pages of analysis)
- **Key advice:** Quality over quantity — the report should highlight **key observations**, not everything
- Brevity is important: clearly communicate the most important findings

### 2.2 Markov Chains

**Definition:** A model for sequence systems using a **discrete set of states**.
- **States:** Must be explicitly defined (e.g., healthy, critical, deceased)
- **Transition matrix:** Probabilities of moving from one state to another
- **Memoryless property:** Next state depends **only on the current state** (not on the full history)

### 2.3 Markov Chain vs ARIMA

| Model | Predicts | Output |
|-------|----------|--------|
| **ARIMA** | Values of the signal | Continuous number (e.g., heart rate value) |
| **Markov Chain** | State transitions | Discrete class (e.g., healthy/critical) |

### 2.4 Limitations of Markov Chains

- States must be **explicitly defined in advance** — which is often unrealistic
- Real-world states depend on **multiple variables** (heart rate, oxygen, hormones, etc.)
- Example: Classifying a patient as "stable" or "critical" requires:
  - Heart rate range
  - Blood oxygen concentration
  - Hormone levels
  - Other lab values
- In practice, defining states purely from observable variables is **difficult or impossible**

### 2.5 Bridge to Hidden Markov Models (HMM)

- **Key insight:** States are usually **not directly observable**
- The natural next step: assume states are **hidden** (latent)
- HMMs use Markov chain dynamics (transition matrix) to **infer hidden states** from observations
- This will be covered in the next session

---

## 3. Notebook Availability

- **No Python notebook** for Markov Chains
- Markov Chains are treated as **foundational theory** for HMMs
- The HMM session will include practical work

---

## 4. Session Timeline

| Section | Time | Content |
|---------|------|---------|
| **I. Assignment 1 Feedback** | 0:15–1:30 | Grades, advice on conciseness |
| **II. Markov Chains** | 1:30–38:27 | States, transition matrices, memoryless property |
| **III. Limitations** | 38:27–40:02 | Difficulty of defining states in real problems |
| **IV. Preview: HMMs** | 40:02–42:33 | Hidden states as generalization |
| **V. Wrap-up** | 42:33–47:36 | Summary, no notebook, technical issues with recording |

---

## 5. Key Takeaways

- **Markov Chains** model discrete state transitions with the memoryless property
- **Transition matrix** encodes the dynamics between states
- The main **limitation**: states must be explicitly defined (often unrealistic)
- Assignment 1 feedback: **be concise** in written reports
- **HMMs** (next session) address the state definition problem by treating states as hidden

---

*Sources: Transcript `2026_04_29_Markov Chains_11.md`*
