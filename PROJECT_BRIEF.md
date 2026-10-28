# AirSense: A Mathematical Engine for Sensor Calibration, PCA, and Drift Monitoring

## 1. Project Overview & Objective

**AirSense** is a mathematically rigorous, research-oriented Python application designed for an undergraduate *Mathematical Foundations for AI and Data Science* mini-project and portfolio piece. 

The application solves a real-world scientific computing problem: **calibrating low-cost chemical gas sensor arrays to predict reference benzene ($C_6H_6$) concentrations**, evaluating the mathematical implications of **Principal Component Analysis (PCA)** on calibration accuracy and temporal stability, and monitoring **sensor drift** over a contiguous 1-year recording timeline.

### Pedagogical Standard
Every mathematical component is built following the university viva standard:
$$\text{Concept} \longrightarrow \text{Purpose} \longrightarrow \text{Outcome}$$
Every component features:
1. **Mathematical Explanation**: Rigorous formulation from first principles.
2. **From-Scratch Implementation**: Explicit numerical implementation (NumPy/SciPy primitives).
3. **Reference Verification**: Parity checks and numerical tolerance benchmarks against industry-standard libraries (`numpy.linalg`, `scipy.linalg`, `sklearn`).
4. **Demonstrable Artifact**: Numerical output, matrix condition diagnostics, or interactive visualization.

---

## 2. Core Research Question

> **How does PCA-based dimensionality reduction affect the predictive accuracy and temporal stability of least-squares calibration for benzene concentration using real-world gas-sensor measurements?**

### Sub-Hypotheses & Investigation Plan
1. **Variance vs. Relevance**: PCA selects orthogonal directions maximizing feature variance $\text{Var}(X\mathbf{v}_i)$. Does top variance in metal-oxide sensor readings correspond to target benzene sensitivity, or does it capture environmental confounders (Temperature, Humidity)?
2. **Reconstruction vs. Calibration**: How does the reconstruction error $\|\tilde{X} - \hat{X}_k\|_F$ scale with $k$, and does an optimal truncation $k^* < d$ regularize collinear sensors or induce harmful bias?
3. **Temporal Stability & Drift**: Does calibration on low-dimensional principal subspaces degrade more or less severely than full-rank Ordinary Least Squares (OLS) as sensors age and seasons change across the 1-year timeline?
4. **Residual Behavior**: Are residual shifts driven by true sensor degradation/poisoning, unmodeled non-linearities, or seasonal covariate shift?

---

## 3. Dataset Specification

- **Source**: UCI Machine Learning Repository — [Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
- **Nature**: 9,358 hourly field recordings (March 2004 to April 2005) from an array of 5 metal-oxide sensors deployed in an Italian city, alongside a certified reference analyzer.
- **Physical Sensor Predictors (8 features)**:
  - `PT08.S1(CO)`: Tin oxide sensor (carbon monoxide targeted)
  - `PT08.S2(NMHC)`: Titania sensor (non-methane hydrocarbons targeted)
  - `PT08.S3(NOx)`: Tungsten oxide sensor (nitrogen oxides targeted)
  - `PT08.S4(NO2)`: Tungsten oxide sensor (nitrogen dioxide targeted)
  - `PT08.S5(O3)`: Indium oxide sensor (ozone targeted)
  - `T`: Ambient Temperature (°C)
  - `RH`: Relative Humidity (%)
  - `AH`: Absolute Humidity ($g/m^3$)
- **Primary Prediction Target**:
  - `C6H6(GT)`: True reference Benzene concentration in $\mu g/m^3$.
- **Excluded Columns**:
  - All certified reference analyzer measurements ending in `(GT)`: `CO(GT)`, `NMHC(GT)`, `NOx(GT)`, `NO2(GT)`.
  - Date and Time are converted into a unified ISO-8601 UTC timestamp and preserved strictly for **chronological ordering**, time-series tracking, and out-of-time evaluation, never fed as raw linear features.
- **Data Hygiene & Missing Value Sentinel**:
  - Missing readings are documented by UCI as `-200`. These must be converted to `NaN`.
  - Rows missing the target `C6H6(GT)` are excluded from supervised evaluation.
  - Missing feature values must be audited (missingness patterns, contiguous gaps) and imputed strictly using statistics estimated solely on the training partition.

---

## 4. Academic Workflow & Mathematical Mapping

The syllabus requires 10 core stages plus a dedicated inference laboratory:

| Stage | Mathematical Concept | Algorithmic Implementation | Verification & Output |
| :--- | :--- | :--- | :--- |
| **1. Data Loading & Matrix Representation** | Matrix spaces $\mathbb{R}^{n \times d}$, design matrix $X = [\mathbf{1} \mid X_s]$, row/column indexing, machine precision $\epsilon$ | Custom data loader, array structuring, design matrix builder | Shape assertions, memory layout, IEEE 754 precision checks |
| **2. Elimination, RREF & LU Decomposition** | Elementary row operations, Gaussian elimination with partial pivoting, $PA = LU$, forward/back substitution | `gaussian_elimination()`, `rref()`, `lu_factorize()`, `lu_solve()` | Scipy `scipy.linalg.lu`, numerical residual $\|PA - LU\|_\infty < 10^{-12}$ |
| **3. Vector Spaces & Subspaces** | Four fundamental subspaces: $C(A), N(A), C(A^T), N(A^T)$, Rank-Nullity Theorem $\dim(C(A)) + \dim(N(A)) = d$, linear independence | `matrix_rank()`, `null_space_basis()`, `column_space_basis()`, condition check | Parity with `numpy.linalg.matrix_rank`, dimension theorems verified |
| **4. Linear Transformations** | Linear maps $T: \mathbb{R}^d \to \mathbb{R}^m$, change of basis $P^{-1}AP$, centering and scaling transforms | Transformation matrix operators, feature standardizers | Geometric distortion, invertibility check $\det(A) \ne 0$ |
| **5. Norms, Orthogonality & Gram–Schmidt** | Vector $\ell_1, \ell_2, \ell_\infty$ norms, inner products $\langle \mathbf{u}, \mathbf{v} \rangle$, cosine angles, Classical vs. Modified Gram–Schmidt (MGS), $A = QR$ | `vector_norm()`, `gram_schmidt_modified()`, `qr_factorize()` | Orthogonality verification $\|Q^T Q - I\|_F < 10^{-12}$, comparison with `scipy.linalg.qr` |
| **6. Projections & Least Squares** | Orthogonal projection operator $P = X(X^T X)^{-1}X^T$, normal equations $X^T X \mathbf{w} = X^T \mathbf{y}$, residual orthogonality $X^T \mathbf{e} = \mathbf{0}$, QR & SVD solvers | `project_onto_subspace()`, `solve_ols_normal_eq()`, `solve_ols_qr()`, `solve_ridge()` | Residual orthogonality checks, equivalence with `scipy.linalg.lstsq` |
| **7. Covariance & Positive Semi-Definiteness** | Sample covariance matrix $\Sigma = \frac{1}{n-1}\tilde{X}^T \tilde{X}$, quadratic forms $\mathbf{z}^T \Sigma \mathbf{z} \ge 0$, Cholesky decomposition $\Sigma = L L^T$ | `empirical_covariance()`, `is_positive_semidefinite()`, `cholesky_factorize()` | Parity with `numpy.cov`, Cholesky verification $\|L L^T - \Sigma\|_F < 10^{-12}$ |
| **8. Eigenvalues & Symmetric Diagonalization** | Spectral Theorem for symmetric matrices $S = V \Lambda V^T$, characteristic equation, Power Iteration, Rayleigh Quotient Iteration | `power_iteration()`, `eigen_symmetric_jacobi()` or QR algorithm | Parity with `scipy.linalg.eigh`, orthogonality $V^T V = I$, $\|AV - V\Lambda\| < 10^{-11}$ |
| **9. SVD & Principal Component Analysis** | SVD: $\tilde{X} = U \Sigma V^T$, singular values $\sigma_i = \sqrt{(n-1)\lambda_i}$, Eckart-Young-Mirsky Low-Rank Approximation, Scree plot, explained variance | `svd_thin()`, `pca_fit_transform()`, `reconstruction_error()` | Parity with `scipy.linalg.svd` and `sklearn.decomposition.PCA`, variance ratio verification |
| **10. Regression, Uncertainty & Calibration** | Principal Component Regression (PCR) vs OLS vs Ridge, bias-variance trade-off, time-series evaluation, CUSUM drift detection | `pcr_model()`, `evaluate_metrics()`, `cusum_drift_monitor()` | $R^2$, MAE, RMSE, Durbin-Watson statistic, rolling temporal residual stability |
| **Separate Lab: Probability & Inference** | Maximum Likelihood Estimation (MLE), Likelihood Ratio tests, Confidence vs Prediction intervals, Bayesian-Ridge equivalence | `mle_gaussian_regression()`, `confidence_prediction_intervals()`, `hypothesis_tests()` | Synthetic + holdout experiments, formal p-values, parameter uncertainty bounds |

---

## 5. Methodological & Statistical Guardrails

1. **Zero-Data-Leakage Chronological Protocol**:
   - Strict chronological ordering: **Training Set** (first 60%), **Validation/Calibration Set** (next 20%), **Untouched Test Set** (final 20%).
   - All standardizers, imputers, PCA bases, and model hyperparameters are strictly fitted on the **Training Set only** and applied downstream to Validation and Test sets.
   - The test set is evaluated exactly once for final reporting; no hyperparameter tuning or feature selection is permitted on the test set.
2. **Missingness Policy**:
   - UCI Air Quality uses `-200` to denote missing values.
   - All `-200` occurrences in features and targets are converted to `np.nan`.
   - Rows where the prediction target `C6H6(GT)` is missing are pruned for supervised training and evaluation.
   - For missing features, an audit of gap length is conducted. Short missing bursts are documented, and a training-fitted univariate or iterative median imputer is used with zero test-set leakage.
3. **Temporal Autocorrelation & Inference Integrity**:
   - Sensor data exhibit high serial correlation; standard OLS i.i.d. standard errors are biased downward.
   - Hypothesis tests, inference intervals, and residual analyses will account for autocorrelation using **Newey-West (HAC) variance estimators** and block bootstrap methods.
4. **Honest Scientific Claims**:
   - Residual divergence over time will not be prematurely labeled as "sensor drift" without distinguishing between seasonal ambient shifts (temperature/humidity dependence), reference analyzer anomalies, and genuine loss of sensor sensitivity.

---

## 6. Software Architecture & Engineering Standards

- **Language**: Python 3.11+ (running on Python 3.13)
- **Core Dependencies**: NumPy, SciPy, pandas, scikit-learn, Matplotlib, seaborn, pytest, Streamlit
- **Modularity**: Clean separation of mathematical primitives (`src/airsense/math/`), data processing (`src/airsense/data/`), calibration models (`src/airsense/models/`), drift analysis (`src/airsense/drift/`), and UI visualization (`src/airsense/ui/`).
- **Testing Rigor**: Every mathematical algorithm has unit tests verifying:
  - Theoretical invariants (e.g., $Q^T Q = I$, $X^T \mathbf{e} = \mathbf{0}$, $\text{trace}(\Sigma) = \sum \lambda_i$).
  - Edge cases: Singular matrices, rank-deficient systems, collinear inputs, zero vectors.
  - Numerical parity against NumPy/SciPy reference standards within defined tolerances ($\epsilon = 10^{-10}$).
- **Reproducibility**: Environment defined via `pyproject.toml` / `requirements.txt`, automated test suites via `pytest`, and scripted data acquisition pipelines (no raw proprietary or modified data checked into git).
