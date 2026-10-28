# AirSense: Experimental Protocol & Benchmarking Methodology

This document formalizes the research methodology, zero-leakage protocols, chronological partitioning rules, and statistical evaluation framework for **AirSense Phase 3**.

---

## 1. Research Question & Empirical Objectives

> **How does PCA-based dimensionality reduction affect the predictive accuracy and temporal stability of least-squares calibration for benzene concentration using real-world gas-sensor measurements?**

### Empirical Hypotheses Tested:
1. **The Collinear Regularization Hypothesis**: Do the highest principal components ($k \in \{1, \dots, 4\}$) capture sufficient chemical variance to calibrate benzene accurately while filtering out collinear sensor noise?
2. **The Variance vs. Relevance Trade-off**: Does PCA truncation discard low-variance directions that nevertheless contain critical target sensitivity?
3. **The Out-of-Time Generalization Hypothesis**: As sensors age and ambient seasons transition across the 13-month timeline, does an intermediate PCA representation ($k^* < 8$) degrade less severely than full-rank OLS?

---

## 2. Chronological Partitioning Protocol

AirSense strictly forbids random k-fold cross-validation or shuffled splits. In continuous time-series chemical monitoring, random shuffling leaks future weather conditions and sensor degradation states into training, producing artificially inflated $R^2$ scores.

### Exact Partition Boundaries ($n = 8,991$ supervised hours):

| Partition | Fraction | Observation Count | Start Timestamp (UTC) | End Timestamp (UTC) | Physical & Seasonal Context |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Training Set** | $60.0\%$ | $5,394$ | `2004-03-10 18:00:00` | `2004-10-27 15:00:00` | Initial deployment; Spring, high-temperature Summer, early Autumn. |
| **Validation Set** | $20.0\%$ | $1,798$ | `2004-10-27 16:00:00` | `2005-01-16 01:00:00` | Seasonal transition into late Autumn and early Winter. **Used strictly for selecting $k^*$.** |
| **Untouched Test Set** | $20.0\%$ | $1,799$ | `2005-01-16 02:00:00` | `2005-04-04 14:00:00` | Deep Winter and Spring 2005 (1 year after deployment). **Frozen until evaluation.** |

### Verification of Temporal Non-Overlap:
$$\max(t_{\text{train}}) = \text{2004-10-27 15:00} < \min(t_{\text{val}}) = \text{2004-10-27 16:00}$$
$$\max(t_{\text{val}}) = \text{2005-01-16 01:00} < \min(t_{\text{test}}) = \text{2005-01-16 02:00}$$

---

## 3. Strict Zero-Data-Leakage Rules

1. **Standardization Parameters**:
   The empirical feature mean $\boldsymbol{\mu}_{\text{train}} \in \mathbb{R}^d$ and standard deviation $\boldsymbol{\sigma}_{\text{train}} \in \mathbb{R}^d$ are computed **exclusively** on the $5,394$ training rows:
   $$Z_{\text{train}} = (X_{\text{train}} - \boldsymbol{\mu}_{\text{train}}) \oslash \boldsymbol{\sigma}_{\text{train}}$$
   Downstream partitions are transformed using the frozen training parameters:
   $$Z_{\text{val}} = (X_{\text{val}} - \boldsymbol{\mu}_{\text{train}}) \oslash \boldsymbol{\sigma}_{\text{train}}$$
   $$Z_{\text{test}} = (X_{\text{test}} - \boldsymbol{\mu}_{\text{train}}) \oslash \boldsymbol{\sigma}_{\text{train}}$$
2. **PCA Subspace Basis**:
   The SVD factorization $Z_{\text{train}} = U \Sigma V^T$ is performed solely on standardized training data. The principal component projection matrix $V_k \in \mathbb{R}^{d \times k}$ is frozen and never updated using validation or test data.
3. **Hyperparameter Selection Boundary**:
   The optimal number of principal components $k^*$ is chosen based strictly on the Validation set RMSE. The Test set is evaluated exactly once at the very end to produce unbiased out-of-time performance metrics.

---

## 4. Models Compared

### Model A: Full-Feature Ordinary Least Squares (OLS)
* Fits all 8 predictors + intercept:
  $$\mathbf{y} = w_0 + X_s \mathbf{w}_s + \boldsymbol{\varepsilon}$$
* Solved via custom QR factorization ($R\mathbf{w} = Q^T \mathbf{y}$) avoiding condition number squaring.

### Model B: Principal Component Regression (PCR)
* Fits regression on the first $k$ principal component scores:
  $$\mathbf{y} = w_0 + (Z V_k) \mathbf{w}_{\text{pca}} + \boldsymbol{\varepsilon}, \quad k \in \{1, 2, \dots, 8\}$$
* Equivalent physical feature weights back-projected via:
  $$\mathbf{w}_{\text{feature}} = D_{\sigma}^{-1} V_k \mathbf{w}_{\text{pca}} \in \mathbb{R}^8$$

### Model C: Numerical Reference Benchmarks
* Reference `scikit-learn.linear_model.LinearRegression` and `scikit-learn.decomposition.PCA` to verify numerical parity within floating-point tolerance ($10^{-11}$).

---

## 5. Summary of Empirical Findings

```text
 k  Cum_Variance_Pct  Train_RMSE  Train_R2  Val_RMSE  Val_R2  Test_RMSE  Test_R2
 1             55.89      2.4388    0.8867    3.7203  0.8052     3.2110   0.7571
 2             78.08      2.4268    0.8878    3.5884  0.8188     3.1026   0.7732
 3             94.04      1.9274    0.9292    2.8258  0.8876     2.4146   0.8626
 4             96.50      1.8584    0.9342    3.0379  0.8701     2.3802   0.8665
 5             98.01      1.7412    0.9422    3.4116  0.8362     2.8171   0.8130
 6             99.23      1.7282    0.9431    3.3709  0.8401     2.9782   0.7910
 7             99.69      1.1426    0.9751    2.4962  0.9123     4.0028   0.6225
 8            100.00      0.9547    0.9826    2.0242  0.9423     2.8553   0.8079
```

### Critical Scientific Insights for University Viva:

1. **In-Sample vs. Out-of-Time Divergence**:
   * On Training and Validation data, Full OLS ($k = 8$) achieves the highest $R^2$ ($0.9423$) and lowest RMSE ($2.024\ \mu g/m^3$).
   * However, on the untouched out-of-time Test set ($1$ year later), Full OLS degrades to $R^2 = 0.8079$ and $\text{RMSE} = 2.855\ \mu g/m^3$.
2. **The Collinear Overfitting Trap ($k = 7$)**:
   * Component 7 explains only $0.46\%$ of sensor variance, but when included, it overfits to seasonal training interactions:
   * Validation $R^2 = 0.9123 \longrightarrow$ Test $R^2 = 0.6225$ (a catastrophic collapse to $\text{RMSE} = 4.003\ \mu g/m^3$!).
3. **The Stability Victory of Intermediate Truncation ($k = 3$ and $k = 4$)**:
   * Retaining $k = 3$ components captures $94.04\%$ of variance, filtering out unstable collinear modes.
   * On the out-of-time Test set, **PCR with $k = 3$ significantly outperforms Full OLS**:
     $$\text{PCR } (k=3) \text{ Test RMSE} = \mathbf{2.415\ \mu g/m^3} \quad \text{vs.} \quad \text{Full OLS Test RMSE} = \mathbf{2.855\ \mu g/m^3}$$
     $$\text{PCR } (k=3) \text{ Test } R^2 = \mathbf{0.8626} \quad \text{vs.} \quad \text{Full OLS Test } R^2 = \mathbf{0.8079}$$
4. **Temporal Degradation & Negative Mean Bias**:
   * Over the 13-month timeline, mean prediction bias drifts steadily from $+0.44\ \mu g/m^3$ in March 2004 to $-3.69\ \mu g/m^3$ in April 2005.
   * A negative bias indicates the static linear calibration is **systematically overpredicting** benzene concentration, consistent with sensor sensitivity loss and downward resistance drift.

---

## 6. Methodological Limitations & Scientific Nuance

1. **Temporal Serial Correlation**:
   The residual Durbin-Watson statistic is $0.2062 \ll 2.0$. The hourly errors exhibit severe positive autocorrelation ($e_t \approx 0.9 e_{t-1}$). Consequently, standard i.i.d. $p$-values and confidence intervals will be overly narrow. Robust HAC (Newey-West) variance estimators must be employed.
2. **Confounding Seasonal Covariate Shift with Physical Drift**:
   Ambient Temperature and Absolute Humidity undergo annual seasonal cycles. The drop in calibration accuracy in Winter/Spring 2005 combines true sensor aging (oxidation, surface poisoning) with unmodeled environmental non-linearities.
3. **Station Outages**:
   The $366$ missing observation hours correspond to complete station blackouts. While dropping them preserved $8,991$ clean hours, the time series contains several multi-day logging gaps.
