# AirSense: Interactive Calibration Workspace Documentation

**Workspace**: `/calibration` — Sensor Calibration & PCA Regression  
**Date**: October 29, 2026  
**Status**: **Production Ready & Fully Implemented**  
**Integration**: Connected to FastAPI `POST /api/v1/calibration/run` and `GET /api/v1/calibration/benchmark-matrix`  
**Execution Engine**: Direct Python execution via `src/airsense/calibration.py`, `src/airsense/pca_model.py`, and `src/airsense/math_core.py`  

---

## 1. Scientific Overview & Problem Formulation

In low-cost ambient air quality monitoring, solid-state metal-oxide semiconductor (MOX) sensors produce raw resistance / conductivity voltages that co-vary with target atmospheric pollutants (benzene, carbon monoxide, nitrogen oxides, ozone) and ambient weather variables (temperature, relative humidity, absolute humidity).

The primary objective of the **AirSense Calibration Laboratory** is to learn a mapping $f: \mathbb{R}^d \to \mathbb{R}$ translating raw sensor signals $\mathbf{x}_t \in \mathbb{R}^d$ into certified benzene concentrations $y_t \equiv \text{C}_6\text{H}_6(\text{GT}) \in \mathbb{R}$ [$\mu\text{g/m}^3$]:

$$\hat{y}_t = w_0 + \sum_{j=1}^d w_j x_{tj}$$

Crucially, raw MOX sensor responses exhibit severe cross-sensitivities and inter-sensor collinearities ($r > 0.85$). Standard Ordinary Least Squares (OLS) suffers from ill-conditioning and overfits to transient seasonal weather regimes. The workspace enables researchers and evaluators to interactively compare Full OLS with Principal Component Regression (PCR) across arbitrary predictor subsets, observing the effect of dimensionality reduction on validation error and physical feature sensitivity.

---

## 2. Zero-Leakage Chronological Protocol Architecture

The workspace enforces strict out-of-time chronological boundaries to ensure rigorous academic evaluation without future information leakage:

| Partition | Timeframe | Duration | Role in Workspace |
| :--- | :--- | :--- | :--- |
| **Training (60%)** | Mar 10, 2004 18:00 – Oct 27, 2004 15:00 | **5,394 Hours** | Feature mean $\boldsymbol{\mu}$, standard deviation $\mathbf{D}_{\sigma}$, SVD components $\mathbf{V}_k$, and regression weights $\mathbf{w}$ are fitted **strictly** on this partition. |
| **Validation (20%)** | Oct 27, 2004 16:00 – Jan 16, 2005 01:00 | **1,798 Hours** | Used dynamically for interactive model selection, feature subset evaluation, and component tuning ($k$). Scatter, timeline, and residual plots render validation observations. |
| **Untouched Test (20%)** | Jan 16, 2005 02:00 – Apr 04, 2005 14:00 | **1,799 Hours** | **Completely sealed during interactive tuning**. Displayed only as a historical locked reference matrix to prevent data snooping. |

> [!IMPORTANT]
> **Academic Viva Rule**: Interactive experimentation evaluates strictly on Training and Validation partitions. The final test partition is frozen to maintain academic integrity and prevent target leakage.

---

## 3. Mathematical Foundations & Implementation

### 3.1 QR Factorization vs. Normal Equations
Standard linear regression estimates parameters by minimizing the residual sum of squares:
$$\min_{\mathbf{w}} \|\mathbf{X}\mathbf{w} - \mathbf{y}\|_2^2$$

Forming the normal equations $\mathbf{X}^T\mathbf{X}\mathbf{w} = \mathbf{X}^T\mathbf{y}$ squares the spectral condition number:
$$\kappa(\mathbf{X}^T\mathbf{X}) = \kappa(\mathbf{X})^2$$

When $\kappa(\mathbf{X}) \approx 10^4$ (typical for collinear sensor arrays), $\kappa(\mathbf{X}^T\mathbf{X}) \approx 10^8$, resulting in catastrophic floating-point cancellation in IEEE 754 double precision.

AirSense utilizes **Modified Gram-Schmidt QR Factorization**:
$$\mathbf{X} = \mathbf{Q}\mathbf{R}, \quad \mathbf{Q}^T\mathbf{Q} = \mathbf{I}, \quad \mathbf{R} \in \mathbb{R}^{d \times d} \text{ (upper triangular)}$$
$$\mathbf{R}\mathbf{w} = \mathbf{Q}^T\mathbf{y}$$

Because $\mathbf{Q}$ is orthogonal, the condition number of the triangular system satisfies $\kappa(\mathbf{R}) = \kappa(\mathbf{X})$, preserving full numerical precision.

### 3.2 5-Step Principal Component Regression (PCR)
1. **Feature Standardization**:
   $$\mathbf{Z} = (\mathbf{X} - \mathbf{1}\boldsymbol{\mu}^T)\mathbf{D}_{\sigma}^{-1}, \quad \mu_j = \frac{1}{n}\sum_{i=1}^n X_{ij}, \quad \sigma_j = \sqrt{\frac{1}{n-1}\sum_{i=1}^n (X_{ij} - \mu_j)^2}$$
2. **Thin Singular Value Decomposition (SVD)**:
   $$\mathbf{Z} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T, \quad \mathbf{V} \in \mathbb{R}^{d \times d}$$
3. **Subspace Coordinate Projection**:
   $$\mathbf{T}_k = \mathbf{Z}\mathbf{V}_k \in \mathbb{R}^{n \times k}, \quad 1 \le k \le d$$
4. **Least-Squares Regression on Orthogonal Scores**:
   $$\hat{\mathbf{y}} = w_0\mathbf{1} + \mathbf{T}_k \mathbf{w}_{\text{pca}}, \quad \mathbf{w}_{\text{pca}} = (\mathbf{T}_k^T\mathbf{T}_k)^{-1}\mathbf{T}_k^T(\mathbf{y} - \bar{y}\mathbf{1})$$
5. **Physical Weight Back-Projection**:
   To translate latent principal weights back into physical sensor resistance units ($\mu\text{g/m}^3$ per raw unit):
   $$\hat{y} = w_0 + (\mathbf{x} - \boldsymbol{\mu})^T \mathbf{D}_{\sigma}^{-1}\mathbf{V}_k \mathbf{w}_{\text{pca}} = \left(w_0 - \boldsymbol{\mu}^T \mathbf{w}_{\text{feature}}\right) + \mathbf{x}^T \mathbf{w}_{\text{feature}}$$
   $$\mathbf{w}_{\text{feature}} = \mathbf{D}_{\sigma}^{-1}\mathbf{V}_k \mathbf{w}_{\text{pca}} \in \mathbb{R}^d$$

---

## 4. REST API Endpoint Specifications

### 4.1 `POST /api/v1/calibration/run`
Executes user-configured calibration model on the backend Python engine.

**Request Schema (`CalibrationRunRequest`)**:
```json
{
  "model_type": "PCR",
  "selected_features": [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH"
  ],
  "k_components": 3
}
```

**Response Schema (`CalibrationRunResponse`)**:
```json
{
  "experiment_id": "exp_pcr_8f_k3",
  "model_type": "PCR",
  "selected_features": ["PT08.S1(CO)", "PT08.S2(NMHC)", ...],
  "k_components": 3,
  "feature_count": 8,
  "train_metrics": {
    "rmse": 1.9274,
    "mae": 1.3412,
    "r2": 0.9292,
    "mbe": 0.0000
  },
  "val_metrics": {
    "rmse": 2.8258,
    "mae": 1.8805,
    "r2": 0.8876,
    "mbe": -0.4215
  },
  "feature_sensitivities": [
    {"feature": "PT08.S1(CO)", "weight": 0.003412, "unit": "ug/m3 per sensor_unit"},
    {"feature": "PT08.S2(NMHC)", "weight": 0.018945, "unit": "ug/m3 per sensor_unit"}
  ],
  "intercept": 10.0827,
  "pca_variance": [
    {"k": 1, "variance_pct": 55.89, "cum_variance_pct": 55.89},
    {"k": 2, "variance_pct": 22.19, "cum_variance_pct": 78.08},
    {"k": 3, "variance_pct": 15.96, "cum_variance_pct": 94.04}
  ],
  "scatter_sample": [
    {"timestamp": "2004-10-27 16:00", "actual": 11.2, "predicted": 10.8, "residual": 0.4}
  ],
  "timeline_sample": [
    {"timestamp": "2004-10-27 16:00", "actual": 11.2, "predicted": 10.8}
  ],
  "residuals_distribution": [
    {"bin_start": -8.5, "bin_end": -7.5, "bin_center": -8.0, "count": 12, "density": 0.0067, "normal_density": 0.0051}
  ],
  "split_info": {
    "training_rows": 5394,
    "validation_rows": 1798,
    "test_partition": "FROZEN (1,799 hours out-of-time benchmark strictly preserved)"
  },
  "warnings": [],
  "scientific_notice": {
    "validation_protocol": "Model weights fitted strictly on 5,394 training hours and evaluated on 1,798 validation hours.",
    "test_freeze_rule": "Final test partition (1,799 hours) remains completely sealed during interactive tuning."
  }
}
```

### 4.2 `GET /api/v1/calibration/benchmark-matrix`
Returns the 8-configuration historical benchmark matrix ($k = 1 \dots 8$) matching `reports/experiment_results.csv`.

---

## 5. Frontend Interactive Features

1. **Model Architecture Selector**: Toggle between Full OLS and PCR (SVD).
2. **Principal Components Slider**: Clamped between $1$ and $d$ (active features count) with quick preset pills:
   - `k=1 (Coarse)`
   - `k=3 (Regularized)`
   - `k=d (Full Rank)`
3. **Predictor Feature Matrix**: Individual toggle checkboxes for all 8 predictors, grouped into Metal-Oxide Sensors (MOX) and Environmental Variables (Env). Includes "Select All" and "Sensors Only" shortcuts.
4. **Validation Performance KPI Strip**: Real-time cards displaying Validation RMSE, MAE, $R^2$, and Mean Bias Error (MBE), each paired with training metrics to detect overfitting.
5. **Interactive Visualization Tabs**:
   - **Measured vs. Predicted**: 180-point validation scatter plot with $1:1$ reference line.
   - **Validation Timeline**: Dual line trajectory comparing certified reference benzene with model output.
   - **Residual Distribution**: 20-bin histogram of validation errors with fitted Gaussian normal curve.
   - **Feature Sensitivities**: Horizontal bar chart and attribution table displaying back-projected physical weights.
   - **PCA Scree Plot**: Component variance percentages and cumulative variance curve.
6. **Frozen Benchmark Reference Matrix**: Tabular view of historical out-of-time performance for $k = 1 \dots 8$.
7. **Mathematical Formulations Drawer**: Collapsible technical guide covering QR stability, 5-step PCR derivation, Concept $\to$ Purpose $\to$ Outcome triad, and university viva defense questions.

---

## 6. Verification and Validation

- **Automated Tests**: 14 tests in `tests/test_api.py` (and 73 total in pytest suite) verifying:
  - Baseline 8-feature OLS validation RMSE ($2.0242 \pm 0.001$).
  - PCR $k=3$ validation RMSE ($2.8258 \pm 0.001$) and cumulative variance ($94.04\% \pm 0.1\%$).
  - Custom 3-feature subset fitting (`PT08.S1`, `PT08.S2`, `T`).
  - Validation guards: $k > d$ error (HTTP 400), empty feature set error (HTTP 400), unsupported model type error (HTTP 400), reference analyzer target leakage rejection (HTTP 400).
- **TypeScript & Vite Build**: `npm run build` passes with zero type errors (`dist/assets/index-*.js`, gzip: 197 kB).
