# AirSense: Scientific, Mathematical & Operational Limitations

This document provides a transparent, scientifically rigorous catalog of the assumptions, boundary conditions, and limitations of the AirSense engine. An undergraduate research viva is evaluated not only on what a model accomplishes, but on whether the researcher understands where their methodology encounters physical and mathematical boundaries.

---

## 1. Physical Sensor Aging vs. Ambient Environmental Confounding

### 1.1 The Drift Attribution Dilemma
In Section 6 of the Streamlit application and in `scripts/run_experiment.py`, the empirical calibration residuals demonstrate a progressive negative shift in Mean Bias Error (MBE):
- **March 2004**: Mean Bias $= \mathbf{+0.44\ \mu g/m^3}$
- **July 2004**: Mean Bias $= +0.02\ \mu g/m^3$
- **November 2004**: Mean Bias $= -0.89\ \mu g/m^3$
- **February 2005**: Mean Bias $= -2.48\ \mu g/m^3$
- **April 2005**: Mean Bias $= \mathbf{-3.69\ \mu g/m^3}$
- **Net Bias Shift**: **$-4.13\ \mu g/m^3$** over 13 contiguous months.

### 1.2 Why Sensor Drift Cannot Be Claimed as Conclusively Proven
In solid-state physics, metal-oxide (MOX) sensors undergo physical aging due to:
1. Irreversible sintering and grain boundary restructuring of the $\text{SnO}_2 / \text{WO}_3$ semiconductor lattice.
2. Contamination and poisoning of surface adsorption sites by volatile sulfur compounds and siloxanes.
3. Gradual thermal degradation of internal micro-heating elements.

However, in an **uncontrolled outdoor field deployment**, ambient meteorology undergoes massive seasonal oscillations:
- Ambient temperature swings from winter lows of $0^\circ\text{C}$ to summer highs of $40^\circ\text{C}$.
- Absolute humidity ($AH$) alters atmospheric water vapor density ($0.3$ to $2.2\ g/m^3$). Water molecules competitively adsorb onto semiconductor active sites, displacing oxygen ions ($\text{O}^-, \text{O}^{2-}$) and mimicking reducing gas signals.
- Winter atmospheric thermal inversions trap urban pollutants near ground level, radically changing pollutant cross-ratios.

**Scientific Boundary**:  
Because no periodic zero/span calibration was conducted in a sealed environmental chamber using certified reference gas cylinders, **it is impossible to mathematically decouple physical sensor aging from unmodeled non-linear climate confounding**. The empirical data strictly demonstrates **changing error dynamics and temporal non-stationarity**, not isolated sensor drift.

---

## 2. Temporal Autocorrelation & The Failure of Classical i.i.d. Assumptions

### 2.1 The Durbin-Watson Diagnostic
Classical Gauss-Markov regression theory assumes that error disturbances are independent and identically distributed (i.i.d.):
$$\mathbf{e} \sim \mathcal{N}(\mathbf{0}, \sigma^2 I_n) \implies \text{Cov}(e_t, e_{t-k}) = 0 \quad \text{for } k \ne 0$$
Evaluating the Durbin-Watson statistic on AirSense hourly residuals yields:
$$DW = \frac{\sum_{t=2}^N (e_t - e_{t-1})^2}{\sum_{t=1}^N e_t^2} = \mathbf{0.2062} \ll 2.0$$

### 2.2 Mathematical Consequence
A Durbin-Watson statistic of $0.2062$ proves extreme positive serial autocorrelation:
$$\hat{\rho} \approx 1 - \frac{DW}{2} = 1 - \frac{0.2062}{2} \approx \mathbf{0.8969}$$
Successive hourly errors are heavily dependent ($e_t \approx 0.897 e_{t-1} + \nu_t$).
- **Implication**: When errors are positively autocorrelated, the effective sample size is far smaller than the nominal sample size ($N_{\text{eff}} \ll 8,991$).
- **Risk**: Standard ordinary least squares (OLS) standard error formulas underestimate parameter uncertainty by a factor of $\sqrt{\frac{1+\rho}{1-\rho}} \approx \sqrt{\frac{1.897}{0.103}} \approx 4.29\times$.
- **Recommendation**: In operational deployments, confidence intervals and hypothesis tests must use **Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC)** standard errors.

---

## 3. Power-Law Kinetics vs. First-Order Linear Models

### 3.1 Non-Linear Chemisorption
Metal-oxide gas sensors operate through surface chemisorption kinetics described by the **Wolkenstein adsorption isotherm** and the empirical power-law response equation:
$$R_s = R_0 (1 + K \cdot C)^{-\alpha} \quad \text{or} \quad \ln(R_s) \approx \ln(R_0) - \alpha \ln(C)$$
where $R_s$ is sensor resistance, $C$ is target gas concentration, and $\alpha \in [0.3, 0.8]$ is a sensor-specific sensitivity exponent.

### 3.2 Linear Model Approximation
AirSense implements linear least-squares calibration:
$$\hat{y}_t = w_0 + \sum_{j=1}^d w_j x_{tj}$$
- **Strengths**: Linear models are globally convex, numerically stable, have unique closed-form solutions via QR/SVD, and provide interpretable physical sensitivities ($\mathbf{w}_{\text{feature}} = D_\sigma^{-1} V_k \mathbf{w}_{\text{pca}}$).
- **Limitation**: A linear plane cannot capture the logarithmic saturation of metal-oxide sensors at high concentrations ($C_6H_6 > 30\ \mu g/m^3$). This causes residual heteroscedasticity: error variance increases at extreme pollution peaks.

---

## 4. Missingness Architecture & Coincident Station Dropouts

### 4.1 Coincident Outage Mechanism
In the raw UCI dataset ($9,357$ hourly observations), exactly $366$ rows contain the `-200` missing sentinel simultaneously across:
- All 5 metal-oxide sensors (`PT08.S1` to `PT08.S5`)
- All 3 weather channels (`T`, `RH`, `AH`)
- The certified reference target (`C6H6(GT)`)

These $366$ coincident rows represent station-wide power failures, electrical breaker trips, or telemetry modem communication dropouts.
- **Consequence**: Because missingness is completely coincident between features and target, dropping rows with missing target values ($9,357 - 366 = 8,991$) leaves a feature matrix with **zero missing values**.
- **Limitation**: While convenient for supervised learning, this means AirSense did not need to deploy complex multiple imputation algorithms (e.g. MICE or matrix completion). In field deployments with asynchronous single-sensor channel dropouts, missing value imputation would be required.

### 4.2 Defective Reference Analyzer
The reference analyzer for non-methanic hydrocarbons (`NMHC(GT)`) failed after May 2004, leaving $8,443$ missing values ($90.23\%$). It cannot serve as a target or predictor, illustrating the fragility of certified field instrumentation.

---

## 5. Static Calibration vs. Dynamic In-Situ Tracking

### 5.1 Static Model Failure
The AirSense calibration models (both Full OLS and PCR) are **static models**: their coefficients $\mathbf{w}$ are fitted once on historical training data (March to October 2004) and held fixed.
- By Month 12 (February 2005), test RMSE degrades to $2.82\ \mu g/m^3$ with an MBE of $-2.63\ \mu g/m^3$.
- By Month 14 (April 2005), test RMSE collapses to $3.82\ \mu g/m^3$ with an MBE of $-3.69\ \mu g/m^3$.

### 5.2 Operational Requirement
In commercial environmental sensor networks, static calibrations must be augmented with:
1. **Recursive Least Squares (RLS)** with exponential forgetting factors ($\lambda \approx 0.995$) to continuously adapt baseline offsets.
2. **Periodic Transfer Calibration**: Deploying a roving reference vehicle every 3–6 months to update sensor scaling parameters.
3. **Automated CUSUM Alarms**: Triggering recalibration alerts when residual excursions exceed decision boundary $H$.

---

## 6. Scope, Ethics & Regulatory Compliance Notice

AirSense is an educational scientific-computing engine engineered for the university syllabus *Mathematical Foundations for AI and Data Science*.
- It demonstrates linear algebra, spectral theory, numerical optimization, and inferential statistics.
- It is **not certified for regulatory air-quality compliance or life-safety monitoring** under EU Directive 2008/50/EC or US EPA standards.
