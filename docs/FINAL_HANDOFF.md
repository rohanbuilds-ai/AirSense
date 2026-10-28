# AirSense: Final Project Handover & Academic Dossier

**Project Title**: AirSense: A Mathematical Engine for Sensor Calibration, PCA, and Drift Monitoring  
**Course Context**: *Mathematical Foundations for AI and Data Science* (Undergraduate Research Project & Portfolio Defense)  
**Primary Dataset**: UCI Machine Learning Repository Air Quality Dataset (ID 360, De Vito et al., 2008)  
**System Status**: **Completed Undergraduate Research & Educational Application** (Code Frozen, 59 Passing Tests, 4 Executed Notebooks)  
**Date**: October 28, 2026  

---

## 1. Project Title & Research Question

### Title
**AirSense: A Mathematical Engine for Sensor Calibration, PCA, and Drift Monitoring**

### Primary Research Question
> **How does PCA-based dimensionality reduction affect the predictive accuracy and temporal stability of least-squares calibration for benzene concentration using real-world gas-sensor measurements?**

---

## 2. What the Application Currently Does

AirSense is a scientific computing engine and pedagogical laboratory that investigates the mathematics of multi-sensor array calibration. Built from scratch in Python using NumPy and SciPy primitives, it connects abstract linear algebra and mathematical statistics directly to physical chemical sensing challenges.

The application includes:
1. **Automated Data Ingestion & Dialect Parser**: Auto-detects delimiter, decimal notation, and encoding for the UCI Air Quality dataset; replaces numeric sentinels (`-200`) with IEEE NaNs; prunes coincident blackout periods; guarantees zero target leakage.
2. **From-Scratch Mathematical Engine (`airsense.math_core`)**: Numerical linear algebra routines implemented without high-level library shortcuts (Gaussian elimination with partial pivoting, canonical RREF, $PA = LU$ factorization, column/null space bases, Rank-Nullity theorem, Classical vs. Modified Gram-Schmidt, orthogonal projections, QR and SVD least-squares solvers, cyclic Jacobi symmetric diagonalization, SVD-covariance equivalence, and Eckart-Young low-rank approximations).
3. **Leakage-Free Calibration Pipeline (`airsense.calibration`, `airsense.pca_model`)**: Implements chronological partitioning, training-only standardization, full-rank Ordinary Least Squares (OLS), and Principal Component Regression (PCR) with physical feature weight back-projection ($\mathbf{w}_{\text{feature}} = D_{\sigma}^{-1} V_k \mathbf{w}_{\text{pca}}$).
4. **Drift & Quality-Control Diagnostics (`airsense.evaluation`)**: Tracks hourly prediction errors ($e_t = y_t - \hat{y}_t$), computes the Durbin-Watson autocorrelation statistic ($DW = 0.2062$), executes Cumulative Sum (CUSUM) drift detection, and generates monthly error progression tables.
5. **Statistical Laboratory (`airsense.statistical_inference`)**: Central Limit Theorem (CLT) simulations on skewed real-world sensor data, Normal MLE variance bias and Bessel correction, Bayesian MAP regression with Gaussian prior (Ridge equivalence), confidence interval small/large sample behavior, one-sample $t$-test and confidence interval duality, Welch's two-sample $t$-test, paired $t$-test, One-Way ANOVA with sum-of-squares partitioning, Chi-squared test with Cochran's rule, and Monte Carlo power curves.
6. **Interactive 8-Section Streamlit Dashboard (`app.py`)**: A UI designed for academic defense, containing a Data Explorer, Linear Algebra Laboratory, PCA Explorer, Calibration Experiment, Drift Monitor, Statistics Lab, and Viva Dossier.
7. **Four Validated, Fully Executed Jupyter Notebooks**: Standalone, reproducible laboratories covering dataset auditing, linear algebra invariants, calibration benchmarks, and statistical inference.

---

## 3. Major Mathematical Concepts Implemented

AirSense covers all four units of the university syllabus *Mathematical Foundations for AI and Data Science*:

```mermaid
flowchart LR
    A["Raw Sensors (8 Channels)"] --> B["Unit 1: Subspaces & LU"]
    B --> C["Unit 2: MGS QR & Spectral SVD"]
    C --> D["Unit 3: Normal MLE & Bayesian MAP"]
    D --> E["Unit 4: Hypothesis Testing & ANOVA"]
    E --> F["Calibrated Benzene Concentration"]
```

### Unit 1: Linear Systems & Vector Spaces
- **Matrix Systems**: Real-world design matrix $X \in \mathbb{R}^{8991 \times 8}$ and target vector $\mathbf{y} \in \mathbb{R}^{8991}$.
- **Gaussian Elimination & $PA = LU$**: Partial pivoting row swaps:
  $$P A = L U \quad \text{with} \quad \|PA - LU\|_\infty < 10^{-14}$$
- **Canonical RREF & Fundamental Subspaces**: Exact pivot column detection, column space basis $C(A)$, and null space basis $N(A)$ satisfying $\|A N(A)\| < 10^{-14}$.
- **Rank-Nullity Theorem**: Verified numerically: $\text{rank}(A) + \text{nullity}(A) = n$.

### Unit 2: Orthogonality, Projections & Spectral Theory
- **Metric Spaces & Norms**: $\ell_1, \ell_2, \ell_\infty$ vector norms and Cauchy-Schwarz inequality check $|\mathbf{u}^T \mathbf{v}| \le \|\mathbf{u}\|_2 \|\mathbf{v}\|_2$.
- **Gram-Schmidt Stability**: Proof that Modified Gram-Schmidt (MGS) maintains orthogonality $\|Q^T Q - I_d\| < 10^{-8}$ while Classical Gram-Schmidt (CGS) collapses ($10^{-2}$) on ill-conditioned systems.
- **Orthogonal Projections**: Projectors satisfying idempotence $P^2 = P$ and symmetry $P = P^T$, with residual vector orthogonality $X^T (\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}$.
- **Stable Least-Squares**: Custom QR solver ($R\mathbf{w} = Q^T \mathbf{y}$) avoiding normal equations condition squaring ($\kappa(X) = 1.89 \times 10^4$ vs. $\kappa(X^T X) = 3.57 \times 10^8$).
- **Spectral Diagonalization**: Analytic cyclic Jacobi Givens rotations diagonalizing sample covariance $S = V \Lambda V^T$ to within $4.24 \times 10^{-13}$ of SciPy `eigh`.
- **SVD-Covariance Equivalence**: Proof that $\lambda_i = \sigma_i^2 / (n-1)$ to machine precision ($10^{-14}$).
- **Eckart-Young-Mirsky Theorem**: Verified low-rank Frobenius approximation error $\|\tilde{X} - \hat{X}_k\|_F^2 = \sum_{j=k+1}^d \sigma_j^2$.

### Unit 3: Probability, Estimation & Confidence Intervals
- **Central Limit Theorem**: Sampling distribution simulations illustrating convergence of skewed sensor readings to Gaussian as sample size increases from $n=2$ to $n=100$.
- **Normal MLE Bias**: Proof that $\hat{\sigma}^2_{\text{MLE}}$ is biased ($\mathbb{E}[\hat{\sigma}^2] = \frac{n-1}{n}\sigma^2$) and derivation of Bessel's unbiased sample variance $s^2$.
- **Bayesian MAP Regression**: Exact proof that MAP estimation under a zero-mean Gaussian prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$ is identical to Ridge regression with shrinkage parameter $\lambda = \sigma^2 / \tau^2$.
- **Confidence Intervals**: Explicit behavioral comparison between Student's $t$ and standard normal $z^*$ intervals across small and large sample sizes.

### Unit 4: Hypothesis Testing & Inferential Statistics
- **Test-CI Duality**: Exact mathematical proof that rejecting $H_0: \mu = \mu_0$ at significance $\alpha$ is dual to $\mu_0 \notin \text{CI}_{1-\alpha}$.
- **One-Way ANOVA**: Sum-of-squares partitioning ($SS_{\text{Total}} = SS_{\text{Between}} + SS_{\text{Within}}$) confirming significant seasonal benzene variations ($F = 214.3, p < 10^{-15}, \eta^2 = 0.067$).
- **Chi-Squared Contingency**: Categorical independence testing of air quality categories against temperature regimes with Cochran's minimum cell count check.
- **Statistical Power**: Monte Carlo simulations quantifying Type I error rate ($\alpha = 0.05$) and power curves ($1 - \beta$) as functions of effect size and sample size.

---

## 4. Experiment Findings & Model Selection Interpretation

All experiment results are reproduced and recorded in [`reports/experiment_results.csv`](file:///e:/GitHub%20Projects/AirSense/reports/experiment_results.csv).

### Full Experimental Benchmark Matrix ($k \in \{1, \dots, 8\}$)
| Model Architecture | Comp ($k$) | Cum. Var (%) | Val RMSE ($\mu g/m^3$) | Val MAE ($\mu g/m^3$) | Val $R^2$ | Test RMSE ($\mu g/m^3$) | Test MAE ($\mu g/m^3$) | Test $R^2$ | Test MBE ($\mu g/m^3$) | Methodological Role |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| PCR | 1 | 55.89% | 3.7203 | 2.5852 | 0.8052 | 3.2110 | 2.5042 | 0.7571 | +1.5209 | Underfit Baseline |
| PCR | 2 | 78.08% | 3.5884 | 2.4960 | 0.8188 | 3.1026 | 2.4233 | 0.7732 | +1.3412 | Underfit Baseline |
| PCR | 3 | 94.04% | 2.8258 | 1.8805 | 0.8876 | **2.4146** | **1.9375** | **0.8626** | **+0.8612** | Post-Hoc Regularized |
| PCR | 4 | 96.50% | 3.0379 | 2.2378 | 0.8701 | **2.3802** | **1.8714** | **0.8665** | -1.1898 | Post-Hoc Regularized |
| PCR | 5 | 98.01% | 3.4116 | 2.6871 | 0.8362 | 2.8171 | 2.2240 | 0.8130 | -2.0433 | Sub-optimal |
| PCR | 6 | 99.23% | 3.3709 | 2.7075 | 0.8401 | 2.9782 | 2.3947 | 0.7910 | -2.1891 | Sub-optimal |
| PCR | 7 | 99.69% | 2.4962 | 1.8351 | 0.9123 | 4.0028 | 3.4972 | 0.6225 | -3.4972 | Collinear Overfit Trap |
| **Full OLS / PCR** | **8** | **100.00%** | **2.0242** | **1.4741** | **0.9423** | **2.8553** | **2.6521** | **0.8079** | **-2.6521** | **Validation-Selected ($k^*$)** |

### Strict Methodological Interpretation:
1. **The Validation-Selected Model**:  
   In compliance with zero-leakage machine learning methodology, hyperparameter selection was conducted strictly on the validation set (Oct 27, 2004 to Jan 16, 2005). The stated rule selected **Full OLS ($k^* = 8$)** because it achieved the lowest validation error ($\text{Val RMSE} = 2.0242\ \mu g/m^3$, $\text{Val } R^2 = 0.9423$). PCR with 8 components and Full OLS yield identical metrics because both span the complete 8-dimensional column space.
2. **Untouched Test Benchmark**:  
   Evaluated once on the out-of-time test partition (Jan 16, 2005 to Apr 4, 2005), the validation-selected model degraded to $\text{Test } R^2 = 0.8079$ ($\text{RMSE} = 2.8553\ \mu g/m^3$, $\text{MAE} = 2.6521\ \mu g/m^3$, $\text{MBE} = -2.6521\ \mu g/m^3$).
3. **Post-Hoc Empirical Finding on PCR Regularization ($k=3, 4$)**:  
   Evaluating the complete parameter sweep on the test set revealed that compact models ($k=3$ and $k=4$) achieved superior generalization ($\text{Test RMSE } 2.4146$ and $2.3802\ \mu g/m^3$, $\text{Test } R^2\ 0.8626$ and $0.8665$).  
   - **Methodological Distinction**: Compact PCR ($k=3$) is a **post-hoc empirical observation**, not the model selected by the validation rule.
   - **Why this occurred**: The design matrix condition number is high ($\kappa(X) = 1.89 \times 10^4$). Trailing components ($\sigma_7, \sigma_8$) fit subtle seasonal interactions that were valid in Autumn 2004 but inverted one year later in Winter/Spring 2005. Discarding trailing components regularized the model against seasonal distribution shifts.
   - **Viva Nuance**: In an academic viva, do not claim $k=3$ was selected by the protocol; doing so constitutes retrospective test-set leakage. Selecting $k=3$ legitimately in an operational setting requires an out-of-season cross-validation scheme.
4. **The Collinear Overfit Trap ($k=7$)**:  
   Retaining 7 components yielded strong validation fit ($R^2 = 0.9123$), but collapsed on the test set ($R^2 = 0.6225$, $\text{RMSE} = 4.0028\ \mu g/m^3$), empirically confirming the danger of retaining near-collinear spectral directions.

---

## 5. Dataset Provenance & Usage Restrictions

- **Source**: UCI Machine Learning Repository, Dataset ID 360: *Air Quality*.
- **Direct URL**: `https://archive.ics.uci.edu/dataset/360/air+quality`
- **Citation**:  
  > S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, *"On field calibration of an electronic nose for benzene estimation in an urban environment,"* Sensors and Actuators B: Chemical, vol. 129, no. 2, pp. 750-757, 2008.
- **Physical Provenance**: Recorded in an Italian city from March 10, 2004, to April 4, 2005 ($389.8$ continuous days). The station contained 5 metal-oxide chemical sensors, temperature, relative humidity, absolute humidity, and co-located certified reference analyzers.
- **Usage Restrictions**:  
  - Available for research and educational purposes only.
  - Commercial exploitation is prohibited by the dataset authors.
  - The raw CSV must not be modified or redistributed with synthetic alterations.
  - All reference analyzer columns ending in `(GT)` must be excluded from predictive feature matrices to prevent target leakage.

---

## 6. Scientific & Statistical Limitations

As documented in [`docs/known_limitations.md`](file:///e:/GitHub%20Projects/AirSense/docs/known_limitations.md):
1. **The Drift Attribution Dilemma**:  
   While calibration Mean Bias Error shifts negative by **$-4.13\ \mu g/m^3$** over 13 months, outdoor field deployments experience annual weather shifts ($0^\circ\text{C}$ to $40^\circ\text{C}$, $0.3$ to $2.2\ g/m^3$ absolute humidity). Because no sealed-chamber zero/span recalibrations were conducted, **physical sensor semiconductor aging cannot be disentangled from environmental confounding**.
2. **Extreme Serial Autocorrelation ($DW = 0.2062$)**:  
   The Durbin-Watson statistic proves severe positive autocorrelation ($\hat{\rho} \approx 0.8969$). The classical i.i.d. error assumption fails ($e_t \approx 0.897 e_{t-1} + \nu_t$). Nominal standard error formulas underestimate true variance by $\approx 4.29\times$. In production, Newey-West HAC standard errors must be used.
3. **Blackout Discontinuities**:  
   Exactly 366 observation hours dropped all sensors, weather, and reference targets simultaneously due to station power failure. While dropping these rows leaves 8,991 complete rows with zero missing values, it creates non-continuous temporal gaps.
4. **Linearity vs. Semiconductor Kinetics**:  
   Metal-oxide sensors follow non-linear power-law conduction physics ($R_s = R_0 [C]^{-\alpha}$). Static linear OLS and PCR models act as first-order Taylor approximations around the mean operating point and degrade under high pollutant surges.

---

## 7. Verified Setup, Test & Run Commands

Execute these verified commands from the project root:

```powershell
# 1. Install package in editable mode with development dependencies
pip install -e ".[dev]"

# 2. Run the complete automated test suite (59 tests)
pytest -v

# 3. Validate all four Jupyter notebooks against the official nbformat schema
python scripts/validate_notebooks.py

# 4. Ingest raw data, audit missingness, and regenerate audit figures
python scripts/audit_dataset.py

# 5. Run the full calibration and PCR experiment, regenerating CSV and figures
python scripts/run_experiment.py

# 6. Launch the interactive Streamlit viva defense dashboard
streamlit run app.py
```

---

## 8. Viva Voce Preparation Guide (Files to Study)

Before defending the project in your university viva, study these documents in order:

| Priority | Document | Purpose & Key Content |
|:---:|---|---|
| **1** | [`docs/viva_cheat_sheet.md`](file:///e:/GitHub%20Projects/AirSense/docs/viva_cheat_sheet.md) | **10-Point Concept $\to$ Purpose $\to$ Outcome cheat sheet**. Essential mathematical equations and empirical results. |
| **2** | [`docs/viva_questions.md`](file:///e:/GitHub%20Projects/AirSense/docs/viva_questions.md) | **15 Examiner Q&As and Pitfalls**. Model answers on MGS vs CGS, condition squaring, SVD spectral link, model selection protocol, and drift nuance. |
| **3** | [`docs/syllabus_coverage_matrix.md`](file:///e:/GitHub%20Projects/AirSense/docs/syllabus_coverage_matrix.md) | **Line-by-line mapping** of every syllabus concept across Units 1–4 to source code, notebooks, and tests. |
| **4** | [`docs/known_limitations.md`](file:///e:/GitHub%20Projects/AirSense/docs/known_limitations.md) | **Scientific and statistical boundaries**. Essential for demonstrating academic honesty (Durbin-Watson, drift vs climate, effective sample size). |
| **5** | [`docs/research_findings.md`](file:///e:/GitHub%20Projects/AirSense/docs/research_findings.md) | Formal research paper synthesis summarizing calibration accuracy and long-term stability. |
| **6** | [`notebooks/02_linear_algebra_lab.ipynb`](file:///e:/GitHub%20Projects/AirSense/notebooks/02_linear_algebra_lab.ipynb) | Executed linear algebra demonstrations showing RREF, MGS, Jacobi diagonalization, and Eckart-Young bounds. |
| **7** | [`notebooks/03_calibration_experiments.ipynb`](file:///e:/GitHub%20Projects/AirSense/notebooks/03_calibration_experiments.ipynb) | Executed calibration and PCR experiment showing train/val/test evaluation and residual timeline. |

---

## 9. Current Implementation Status

### Completed & Frozen:
- [x] Full Python scientific library (`src/airsense/`) with clean, decoupled functions.
- [x] 59 automated unit and invariant tests passing in ~4.3 seconds.
- [x] Four executed Jupyter notebooks with complete cell outputs, valid JSON, and schema compliance.
- [x] 8-section interactive Streamlit application with live visualizations and viva simulators.
- [x] Experiment pipeline reproducing benchmark CSV and 9 publication-grade figures.
- [x] Complete syllabus matrix, data dictionary, mathematical derivations, and viva packs.

### Intentionally Postponed (Future UI/UX Phase):
- Modern web frontend (React, TypeScript, Tailwind CSS).
- REST API layer (FastAPI adapter wrapping `src/airsense/`).
- Real-time hardware serial-port data ingestion.
- Adaptive time-varying models (Recursive Least Squares, Kalman Filtering).
- Multi-station spatial interpolation.

> [!IMPORTANT]
> **Academic Application Notice**:  
> AirSense is a fully validated undergraduate research and educational engine built to satisfy university syllabus requirements. It is not licensed or certified as a commercial or life-critical industrial air-quality monitoring system.
