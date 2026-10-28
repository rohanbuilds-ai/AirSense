# AirSense: Statistical Assumptions, Diagnostics & Limitations

This document provides a critical examination of the statistical assumptions underlying linear calibration, hypothesis testing, and drift analysis in the **AirSense Project**, documenting violated assumptions, remedial strategies, and causal boundaries.

---

## 1. Critique of Parametric Assumptions

Standard statistical inference tools ($t$-tests, ANOVA, OLS standard errors, MLE) rely on classical Gauss-Markov and parametric assumptions:
1. **Normality of Errors**: $\varepsilon_i \overset{\text{i.i.d.}}{\sim} \mathcal{N}(0, \sigma^2)$.
2. **Homoscedasticity**: $\text{Var}(\varepsilon_i \mid X) = \sigma^2$ (constant error variance across predictor space).
3. **Serial Independence**: $\text{Cov}(\varepsilon_t, \varepsilon_s) = 0 \quad \forall t \ne s$.

---

## 2. Empirical Diagnostics on Real AirSense Data

### Diagnostic 1: Normality of Physical Signals and Residuals
* **Empirical Finding**:
  * Ground-truth benzene $C_6H_6(GT)$ exhibits severe right-skewness ($\text{skew} = 1.62$, $\text{kurtosis} = 3.84$). The D'Agostino-Pearson normality test yields $p < 10^{-100}$, definitively rejecting normality.
  * Chemical sensor responses (`PT08.S1` to `PT08.S5`) are physically constrained to non-negative resistance values and follow asymmetric log-normal or gamma-like shapes.
  * OLS residuals $e_t$ show fat tails and skewness under heavy traffic spikes.
* **Methodological Implication**:
  * Large-sample inference relies on the **Central Limit Theorem** for the sample mean ($\bar{x}_n \approx \mathcal{N}$ as $n \to \infty$).
  * For small-sample tests, nonparametric alternatives or robust Welch formulations must be used rather than assuming exact Gaussianity.

---

### Diagnostic 2: Equality of Variances (Homoscedasticity)
* **Empirical Finding**:
  * Levene's test across chronological quarters yields $p < 10^{-20}$, demonstrating significant heteroscedasticity across seasons.
  * Residual variance in deep Winter ($\sigma_e \approx 2.8\ \mu g/m^3$) is substantially larger than in Summer ($\sigma_e \approx 0.8\ \mu g/m^3$).
* **Methodological Implication**:
  * Classical Student's two-sample $t$-tests (which assume $\sigma_1^2 = \sigma_2^2$) are invalid.
  * We enforce **Welch's $t$-test** with Welch-Satterthwaite adjusted degrees of freedom for all group comparisons.

---

### Diagnostic 3: Temporal Serial Correlation (The Critical Violation)
* **Empirical Finding**:
  * The Durbin-Watson statistic for hourly residuals is:
    $$DW = 0.2062 \ll 2.0$$
  * An hourly residual $e_t$ has an autocorrelation of $\rho_1 \approx 0.89$ with the preceding hour $e_{t-1}$.
* **Consequences of Serial Correlation**:
  1. **Artificially Deflated Standard Errors**: Standard formulas assume independent information per row. In reality, $n$ serially correlated hourly observations carry an *effective sample size* $n_{\text{eff}} \ll n$:
     $$n_{\text{eff}} \approx n \frac{1 - \rho_1}{1 + \rho_1} \approx 8991 \times \frac{0.11}{1.89} \approx 523$$
  2. **Inflated Type I Error Rates**: Naive $t$-tests on regression coefficients will produce microscopic $p$-values ($p < 10^{-16}$) simply due to sample size and autocorrelation, leading to false claims of statistical significance.
  3. **Remedial Strategy**: In our inference lab, we demonstrate that inference on time series requires **subsampling**, **block bootstrapping**, or **Newey-West (HAC)** variance estimators.

---

## 3. Observational Data & Non-Causality

> [!CAUTION]
> **Correlation vs. Causation Guardrail for University Viva**:
> AirSense is built on an **observational field dataset**. No physical variables were experimentally controlled or randomized.

1. **No Causal Inferences**:
   A high regression coefficient between `PT08.S2(NMHC)` and $C_6H_6(GT)$ ($w = 0.82$) reflects empirical chemical co-occurrence in automotive exhaust (gasoline combustion releases both benzene and non-methane hydrocarbons simultaneously). It does **not** prove that the sensor *causes* benzene concentration.
2. **Environmental Confounding**:
   Metal-oxide sensors possess physical semiconductor bandgap sensitivity to ambient water vapor ($AH$) and temperature ($T$). Changes in model predictions across seasons are a mixture of:
   * Ambient covariate shift (weather variations).
   * Traffic volume fluctuations (seasonal driving patterns).
   * Genuine sensor aging / surface oxidation.
   Attributing performance drops strictly to "sensor drift" without experimental laboratory re-calibration would be scientifically unfounded.

---

## 4. Summary Table of Assumptions vs. AirSense Reality

| Classical Assumption | Required by Method | AirSense Status | Impact if Ignored | AirSense Mitigation |
| :--- | :--- | :--- | :--- | :--- |
| **Normality** | $t$-tests, ANOVA, MLE | **Violated** (Right-skewed) | Minor for $n \ge 30$ (CLT applies); severe for small $n$ | CLT simulations; report nonparametric diagnostics |
| **Homoscedasticity** | Student's $t$, Classical ANOVA | **Violated** (Winter variance $3\times$ Summer) | Inflated false positive rate | Welch's $t$-test; report Levene diagnostics |
| **Independence** | All classical inference | **Severely Violated** ($DW = 0.206$) | Massive underestimation of parameter uncertainty | Documented temporal autocorrelation; decorrelation subsampling |
| **Linearity** | OLS Calibration | **Approximation** ($R^2 \approx 0.81$) | Bias in extreme concentration spikes ($> 40\ \mu g/m^3$) | Residual vs Fitted diagnostic plots |
| **Causality** | Never assumed | **Purely Observational** | False causal claims | Explicit observational disclaimer in documentation |
