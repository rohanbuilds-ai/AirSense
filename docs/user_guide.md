# AirSense: Comprehensive User & Operational Guide

Welcome to the **AirSense User Guide**. This document provides instructions for installing, launching, and operating the AirSense Streamlit interface, executing automated verification tests, and reproducing scientific experiments via the command line.

---

## 1. System Requirements & Installation

### 1.1 Prerequisites
- **Python**: Version 3.10, 3.11, 3.12, or 3.13.
- **Operating System**: Windows, macOS, or Linux.
- **Memory**: Minimum 4 GB RAM (8 GB recommended for interactive plotting).

### 1.2 Installation Steps
Clone the repository and install AirSense in editable mode with all dependencies:

```bash
# Clone the project repository
git clone https://github.com/username/AirSense.git
cd AirSense

# Install package and dependencies in editable mode
pip install -e ".[dev]"
```

Alternatively, install using `requirements.txt`:
```bash
pip install -r requirements.txt
pip install pytest
```

---

## 2. Launching the Interactive Streamlit Interface

Run the application directly from the repository root:

```bash
streamlit run app.py
```

Streamlit will launch a local web server (default: `http://localhost:8501`) and open the application in your browser.

---

## 3. Application Section Walkthrough

The application is structured into **8 modular workspaces**, selectable from the sidebar navigation:

```
├── 🏛️ 1. Overview & Provenance
├── 🔍 2. Data Explorer & Missingness
├── 📐 3. Linear-Algebra Laboratory
├── 🌐 4. PCA Explorer & Low-Rank
├── 🔬 5. Sensor Calibration Experiment
├── 📉 6. Sensor Drift & Error Dynamics
├── 📊 7. Probability & Statistics Lab
└── 🎓 8. Methodology & Viva Dossier
```

---

### Section 1: 🏛️ Overview & Scientific Provenance
- **Purpose**: Understand the environmental monitoring challenge, sensor physics, dataset provenance, and modeling boundary conditions.
- **What to explore**:
  - **Problem Statement**: Why low-cost metal-oxide semiconductor (MOX) sensors suffer from chemical cross-sensitivity and baseline aging.
  - **Dataset Specifications**: Summary of the 5 gas sensors (`PT08.S1`–`PT08.S5`), 3 weather covariates (`T`, `RH`, `AH`), and certified reference analyzers (`C6H6(GT)`).
  - **Computational Pipeline**: Architecture flowchart tracking data from raw CSV parsing to QR least-squares solving and CUSUM drift detection.
  - **Scientific Limitations**: Why ambient weather swings and power outages require careful modeling caveats.
  - **Attribution & Licensing**: Citation of the original De Vito et al. (2008) research paper.

---

### Section 2: 🔍 Data Explorer & Missingness Audit
- **Purpose**: Inspect raw field observations, analyze missingness patterns, and examine the certified reference target.
- **Key Features**:
  - **Observation Table**: View the full 9,357-hour dataset or the 8,991-hour supervised subset. Adjust the sample size slider to browse raw values.
  - **Missingness Audit**: Inspect the percentage of missing values per column. Note the **coincident missingness finding**: exactly 366 rows dropped all 8 features and target simultaneously due to power failures.
  - **Reference Concentration**: Analyze the distribution of certified reference benzene ($C_6H_6(GT)$), its positive skewness, and extreme pollution peaks.
  - **Data-Leakage Audit**: Verify that all ground-truth analyzer columns ending in `(GT)` are excluded from the predictor matrix.

---

### Section 3: 📐 Linear-Algebra Laboratory
- **Purpose**: Verify from-scratch numerical algorithms using pure NumPy primitives on deterministic test matrices and real sensor subsets.
- **Interactive Controls**:
  - **Gaussian Elimination & RREF**: Input any custom $3 \times 3$ or $4 \times 4$ matrix or select presets (Invertible, Singular, Sensor Correlation). Observe the step-by-step LU decomposition ($PA = LU$) and pivot column detection in RREF.
  - **Fundamental Subspaces & Rank-Nullity**: View the computed basis for Column Space $C(A)$ and Null Space $N(A)$, verifying $\text{rank}(A) + \text{nullity}(A) = n$.
  - **Gram-Schmidt Stability (CGS vs. MGS)**: Adjust the **Collinearity Perturbation Slider** ($\epsilon = 10^{-1}$ to $10^{-10}$) to see Classical Gram-Schmidt catastrophically lose orthogonality ($\|Q^T Q - I\|_F \approx 10^{-2}$), while Modified Gram-Schmidt preserves it ($\approx 10^{-9}$).
  - **Orthogonal Projections**: Verify symmetry ($P = P^T$), idempotence ($P^2 = P$), and residual orthogonality ($X^T \mathbf{e} = \mathbf{0}$).
  - **Condition Squaring Demonstration**: Compare QR and SVD least-squares solvers against the Normal Equations to understand why condition squaring ($\kappa(X^T X) = [\kappa(X)]^2$) loses 8 decimal digits.

---

### Section 4: 🌐 PCA Explorer & Low-Rank Reconstruction
- **Purpose**: Analyze the spectral decomposition of the sensor covariance matrix and verify low-rank reconstruction bounds.
- **Interactive Controls**:
  - **Scree Plot**: Examine the individual and cumulative explained variance across the 8 principal components.
  - **Eigenvalues vs. Singular Values**: Confirm the theoretical equivalence $\lambda_i = \sigma_i^2 / (n-1)$ to within floating-point tolerance ($< 10^{-14}$).
  - **Projected Coordinates**: View 2D scatter plots of observations projected onto $(PC_1, PC_2)$, colored by certified benzene concentration.
  - **Eckart-Young-Mirsky Bounds**: Adjust the truncation rank $k \in [1, 7]$ to verify that empirical Frobenius reconstruction error equals the theoretical singular value tail sum:
    $$\|\tilde{X} - \hat{X}_k\|_F^2 = \sum_{j=k+1}^d \sigma_j^2$$

---

### Section 5: 🔬 Sensor Calibration Experiment
- **Purpose**: Evaluate the core research question: *How does PCA dimensionality reduction affect calibration accuracy?*
- **Interactive Controls**:
  - **Feature Selector**: Check/uncheck individual chemical sensors and meteorological variables to test restricted models (e.g., chemical sensors only vs. chemical + meteorological).
  - **Model Selector**: Switch between **Principal Component Regression (PCR)** and **Full-Feature Least-Squares (OLS)**.
  - **Component Slider ($k$)**: Select the number of retained principal components ($1 \le k \le d$).
  - **Partition Selector**: Toggle between **Test Set (Out-of-Time)**, **Validation Set**, and **Training Set**.
  - **Metrics & Visualizations**: Review live RMSE, MAE, $R^2$, and Mean Bias Error. Compare predictions against the 1:1 reference line and inspect residual error histograms.
  - **Back-Projected Physical Weights**: Inspect physical sensitivities $\mathbf{w}_{\text{feature}} = D_\sigma^{-1} V_k \mathbf{w}_{\text{pca}}$ in physical engineering units ($\mu g/m^3$ per sensor unit).

---

### Section 6: 📉 Sensor Drift & Error Dynamics
- **Purpose**: Diagnose temporal non-stationarity and sensor error behavior over the 13-month field campaign.
- **Interactive Controls**:
  - **Monthly Bias Progression**: Observe how Mean Bias Error shifts from $+0.44\ \mu g/m^3$ (March 2004) to **$-3.69\ \mu g/m^3$** (April 2005) as sensors lose baseline resistance.
  - **CUSUM Drift Alarms**: Adjust the **Slack Allowance ($\delta$)** and **Alarm Limit ($H$)** to detect cumulative excursions. Note persistent alarm trips in the final quarter (Spring 2005).
  - **Autocorrelation & Serial Dependence**: Examine the Durbin-Watson statistic ($0.2062 \ll 2.0$), confirming positive hourly error correlation.
  - **Environmental Interactions**: Examine scatter plots of residuals against Temperature ($T$) and Absolute Humidity ($AH$).
  - **Scientific Qualification Notice**: Understand why changing error behavior cannot be claimed as isolated physical drift without controlled laboratory recalibration.

---

### Section 7: 📊 Probability & Statistics Laboratory
- **Purpose**: Hands-on exploration of probability, estimation, hypothesis testing, and statistical power.
- **Visual Separation**: Every tab is explicitly labeled with a banner distinguishing **`[SIMULATION]`** from **`[EMPIRICAL]`** observations.
- **Interactive Controls**:
  - **Central Limit Theorem `[SIMULATION]`**: Select synthetic population shapes (Exponential, Uniform, Bimodal) and adjust sample size $n \in [2, 100]$ to observe convergence to $\mathcal{N}(\mu, \sigma^2/n)$.
  - **Normal MLE Bias `[SIMULATION]`**: Adjust small sample size $n \in [3, 25]$ to demonstrate Bessel's variance bias $\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$.
  - **CI & Test Duality `[EMPIRICAL]`**: Input a hypothesized mean $\mu_0$ and significance level $\alpha$ to verify that $H_0: \mu = \mu_0$ is rejected if and only if $\mu_0$ lies outside the $(1 - \alpha)$ confidence interval.
  - **Seasonal ANOVA `[EMPIRICAL]`**: Partition total variance into between-season and within-season components, testing seasonal differences with the $F$-statistic and $\eta^2$ effect size.
  - **Statistical Power Simulator `[SIMULATION]`**: Vary effect size (Cohen's $d$), sample size $n$, and significance level $\alpha$ to visualize Type I ($\alpha$), Type II ($\beta$), and Power ($1 - \beta$).

---

### Section 8: 🎓 Methodology & Viva Defense Dossier
- **Purpose**: Prepare for technical interviews, oral exams, and university viva presentations.
- **Key Resources**:
  - **The 10-Point Viva Triad**: Structured **Concept $\to$ Purpose $\to$ Outcome** cheat sheet covering all 10 linear algebra and statistical units.
  - **Derivations & Tolerances**: Mathematical proofs for condition number squaring and physical weight back-projection.
  - **Assumption Diagnostics**: Reality check evaluating Gauss-Markov regression assumptions against real sensor data.
  - **Examiner Q&A Simulator**: Common examiner questions with model answers.

---

## 4. Running the Verification Test Suite

AirSense includes **55 automated unit tests** verifying mathematical invariants:

```bash
# Run the complete test suite
pytest -v
```

### Test Coverage Highlights:
- `test_math_core.py`: LU factorization identity ($\|PA - LU\| < 10^{-12}$), Rank-Nullity theorem ($\text{rank} + \text{nullity} = n$), Gram-Schmidt MGS stability, projection idempotence ($P^2 = P$) and symmetry ($P = P^T$), cyclic Jacobi spectral decomposition, and Eckart-Young reconstruction error.
- `test_calibration.py`: Chronological partition monotonicity, leakage-free standardizer, QR least-squares parity with scikit-learn, and PCR pipeline.
- `test_data_loader.py`: CSV dialect detection, sentinel (-200) conversion, timestamp monotonicity, and schema validation.
- `test_statistical_inference.py`: CLT convergence, MLE variance bias, CI duality, Welch $t$-test, ANOVA $SS$-partitioning, and power curves.
- `test_ui.py`: Cached dataset loaders, CUSUM trip logic, and view renderer callable checks.

---

## 5. Command-Line Experiment Execution

To reproduce the data audit and experimental benchmark results without opening the web interface:

```bash
# 1. Run Data Ingestion & Audit Script
python scripts/audit_dataset.py

# 2. Run Calibration Benchmark Across k=1..8
python scripts/run_experiment.py
```

Generated artifacts are stored in:
- Cleaned CSVs: `data/processed/air_quality_cleaned.csv` and `air_quality_supervised.csv`
- Benchmark Table: `reports/experiment_results.csv`
- Audit Figures: `reports/figures/01_missingness_audit.png` through `09_monthly_error_progression.png`

---

## 6. Troubleshooting & FAQs

### Q1: `ModuleNotFoundError: No module named 'airsense'`
Run `pip install -e .` in the root directory where `pyproject.toml` is located. This registers the `airsense` package in your Python environment.

### Q2: Streamlit shows a blank page or port conflict
Specify a custom port:
```bash
streamlit run app.py --server.port 8502
```

### Q3: Why does `app.py` take a few seconds on first launch?
On first launch, `@st.cache_data` loads and processes the 9,357-row dataset and caches it in memory. Subsequent tab navigation is instantaneous.
