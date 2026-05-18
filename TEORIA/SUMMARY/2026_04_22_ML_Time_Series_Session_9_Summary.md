# Machine Learning — Session 9: Time Series Introduction

**Date:** 2026-04-22 (Wednesday)
**Instructor:** Alexandra Gómez Villa
**Duration:** ~1h 01min

---

## 1. Overview

This session begins **Block 3: Time Series Analysis**. It introduces the fundamental shift from independent samples to **time-dependent data**, covering **stationarity, autocorrelation, and windowing** — the three core concepts for time series analysis.

---

## 2. Key Concepts

### 2.1 Time Series vs Traditional ML

| Traditional ML | Time Series |
|----------------|-------------|
| Samples are **independent** | Samples depend on **time/order** |
| No relationship between observations | **Sequence matters** |
| Suitable for static data (images, tables) | Suitable for signals (ECG, stock prices) |

### 2.2 Three Core Concepts

| Concept | Description |
|---------|-------------|
| **Stationarity** | Statistical properties (mean, variance) are constant over time |
| **Autocorrelation** | Correlation of the signal with a delayed copy of itself (lag) |
| **Windowing** | Converting variable-length series into fixed-size windows for ML models |

### 2.3 Stationarity

- **Stationary:** Mean and variance don't change over time (easier to model)
- **Non-stationary:** Trends, seasonality, changing variance (must be corrected)
- Common corrections: differencing, detrending

### 2.4 Autocorrelation & Seasonality

- Autocorrelation plots help identify **patterns** (e.g., daily, weekly cycles)
- Helps choose **window size K** and **stride**
- Example: If there's a 24-hour pattern, window should capture at least one period

### 2.5 Windowing

- Convert time series of variable length into **fixed-size windows**
- **Parameters:** window size (K) and stride
- Each window is an **independent sample** for the ML model
- The model predicts the **next value(s)** after the window

### 2.6 Sliding Window Prediction

- **Training:** Given window → predict next single value
- **Inference:** Predict one step at a time, add prediction to signal, slide window forward
- **Limitation:** Prediction quality degrades for far-future values (accumulated error)

### 2.7 Limitations of Time Series Prediction

- **Stock prices** are highly stochastic and nearly impossible to predict
- **Chaos theory / butterfly effect:** Small changes → large future deviations
- Models are good for **short-term prediction** (lag 1–4), not long-term
- **Transformers** (from deep learning) are the current best models for time series

### 2.8 Complete Time Series Pipeline

1. **Raw time series**
2. **Stationarity check** → correct if non-stationary
3. **Windowing** (based on autocorrelation for K and stride)
4. **Feature extraction** (5–6 key features)
5. **Model training** (SVM, Neural Network)
6. **Prediction** (classification or regression/forecasting)

---

## 3. Session Notebook

- Longer notebook than usual
- Demonstrates windowing, autocorrelation function with **Pandas** and **NumPy**
- Available on Campus Virtual

---

## 4. Session Timeline

| Section | Time | Content |
|---------|------|---------|
| **I. Time Series Intro** | 0:03–1:48 | From independent samples to time-dependent |
| **II. Core Concepts** | 1:48–52:00 | Stationarity, autocorrelation, windowing |
| **III. Q&A — Prediction Strategy** | 52:00–59:04 | Stride, multi-step prediction, limitations |
| **IV. Pipeline Summary** | 59:04–1:01:30 | Complete time series workflow |

---

## 5. Key Takeaways

- **Time series** data has sequential dependencies (not independent)
- **Stationarity**, **autocorrelation**, and **windowing** are foundational
- Models predict **short-term** well, **long-term** degrades
- **Transformers** are the current best for time series (but out of scope)
- Notebook covers windowing with Pandas/NumPy

---

*Sources: Transcript `2026_04_22_Time Series_9.md`*
