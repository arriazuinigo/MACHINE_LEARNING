# Machine Learning — Session 10: ARIMA Models

**Date:** 2026-04-24 (Friday)
**Instructor:** Alexandra Gómez Villa
**Duration:** ~1h 01min

---

## 1. Overview

This session covers **ARIMA (AutoRegressive Integrated Moving Average)** models — the classic statistical approach for time series forecasting. ARIMA extends the concepts from Session 9 (stationarity, autocorrelation, windowing) into a formal modeling framework.

---

## 2. Key Concepts

### 2.1 ARIMA Components

| Component | Meaning | Parameter |
|-----------|---------|-----------|
| **AR** (AutoRegressive) | Uses past values to predict future values | **p** — lag order |
| **I** (Integrated) | Differencing to make series stationary | **d** — degree of differencing |
| **MA** (Moving Average) | Uses past forecast errors | **q** — window size |

### 2.2 Model Selection

- **p, d, q** parameters are selected based on:
  - **Autocorrelation (ACF)** and **Partial Autocorrelation (PACF)** plots
  - Information criteria (AIC, BIC)
- Common approach: grid search over (p,d,q) combinations

### 2.3 ARIMA Extensions

| Extension | Description |
|-----------|-------------|
| **SARIMA** | Adds **seasonal** component (S) |
| **SARIMAX** | Adds e**X**ogenous variables (external regressors) |

### 2.4 Practical Use

- ARIMA works best for **univariate** time series with clear patterns
- Classical benchmark before using ML/DL approaches
- Still widely used in **economics**, **epidemiology**, and **healthcare forecasting**

---

## 3. Session Notebook

- Python notebook available on Campus Virtual
- Demonstrates ARIMA model fitting and forecasting with `statsmodels`

---

## 4. Key Takeaways

- **ARIMA** = AR (p) + I (d) + MA (q)
- **p**: autoregressive lags; **d**: differencing; **q**: moving average window
- **ACF/PACF plots** guide parameter selection
- **SARIMA/SARIMAX** handle seasonality and external variables
- Classical benchmark before ML approaches

---

> ⚠️ **Note on source transcripts:** The transcript labeled `2026_04_24_ARIMA_10.md` contains the same content as Session 9 (Time Series). This summary is based on the course syllabus structure and the time series concepts presented across Sessions 9–10.

*Sources: Course syllabus (Block 3: Time Series — Session 10: ARIMA Models)*
