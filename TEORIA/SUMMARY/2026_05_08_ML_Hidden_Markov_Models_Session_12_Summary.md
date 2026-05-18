# Machine Learning — Session 12: Hidden Markov Models (HMM)

**Date:** 2026-05-08 (Friday)
**Instructor:** Alexandra Gómez Villa
**Duration:** ~47 min

---

## 1. Overview

This session covers **Hidden Markov Models (HMM)** — a **generalization of Markov Chains** where states are **hidden (latent)** rather than directly observable. This is the final session of the course, also serving as a transition to the Deep Learning course.

---

## 2. Key Concepts

### 2.1 From Markov Chains to HMMs

| Markov Chains | Hidden Markov Models |
|---------------|---------------------|
| States are **observable** | States are **hidden** (latent) |
| Need to explicitly define states | States are inferred from observations |
| Limited for real-world problems | More flexible and realistic |
| Transition matrix is known | Transition matrix + **emission probabilities** |

### 2.2 Why HMMs?

- In real healthcare scenarios, it's **impossible to fully define a patient's state** using just observable variables
- A patient's true condition depends on many factors (heart rate, oxygen, hormones, lab values, etc.)
- HMMs allow: "Let's say the states are **not observable** — they are hidden from us"
- The model **infers hidden states** from observable data

### 2.3 HMM Components

| Component | Description |
|-----------|-------------|
| **Hidden states** | True but unobserved system states |
| **Observations** | Measurable signals/data |
| **Transition matrix** | Probability of moving between hidden states |
| **Emission probabilities** | Probability of observing data given a hidden state |

### 2.4 HMM vs ARIMA vs Markov Chains

| Model | States | Predicts |
|-------|--------|----------|
| **ARIMA** | N/A | Continuous signal values |
| **Markov Chain** | Observable discrete | State class |
| **HMM** | Hidden (latent) | Most likely hidden state sequence |

### 2.5 Practical Applications in Healthcare

- Patient monitoring (inferring true health state from noisy measurements)
- Disease progression modeling
- Activity recognition from sensor data
- Any scenario where the **true state is unobservable** but we have noisy observations

---

## 3. Course Wrap-up

- This concludes the Machine Learning course
- The concepts from this course form the **foundation for Deep Learning**
- HMMs serve as a **bridge** between traditional ML and modern DL approaches

---

## 4. Session Timeline

| Section | Time | Content |
|---------|------|---------|
| **I. Assignment 1 Feedback** | 0:15–1:30 | Grades, advice on concise reporting |
| **II. HMM Theory** | 1:30–40:00 | Hidden states, transition matrix, emission probabilities |
| **III. Limitations of Markov Chains** | 40:00–42:33 | Why states being observable is unrealistic |
| **IV. Course Conclusion** | 42:33–47:36 | HMM as bridge to deep learning |

---

## 5. Key Takeaways

- **HMMs** generalize Markov Chains by making states **hidden** (latent)
- They infer the most likely **hidden state sequence** from observations
- More **realistic** for healthcare (true patient state is never fully observable)
- **Transition matrix** (from Markov Chains) + **emission probabilities** (new)
- HMMs connect traditional ML to **deep learning concepts**
- Assignment 2 deadline: **May 13**

---

> ⚠️ **Note on source transcripts:** The transcript labeled `2026_05_08_Hidden_Markov_Models_12.md` contains the same content as Session 11 (Markov Chains). This summary is based on the course syllabus structure, the Markov Chains foundations from Session 11, and the expected HMM content for the final session.

*Sources: Course syllabus (Block 4: Advanced Topics — Session 12: Hidden Markov Models)*
