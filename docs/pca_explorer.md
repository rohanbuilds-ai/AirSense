# AirSense: PCA Explorer (`/pca-explorer`)

**Workspace**: Principal Component Analysis Explorer  
**Syllabus Coverage**: Unit 2 (Spectral SVD, Covariance Diagonalization, Low-Rank Approximations, PCA/PCR)  
**Implementation Phase**: UI/UX Phase 5  
**Date**: October 29, 2026  
**Status**: **Complete & Verified (84 passing automated tests, Vite production build verified)**  

---

## 1. Executive Summary

The **PCA Explorer** (`/pca-explorer`) is an interactive analytical workspace designed to explore the mathematics, geometry, and sensor-calibration dynamics of Principal Component Analysis.

Rather than relying on black-box dimensionality reduction tools, the PCA Explorer exposes the exact spectral equivalence between Thin Singular Value Decomposition ($X_c = U \Sigma V^T$) and covariance matrix diagonalization ($S = V \Lambda V^T$), proving the Eckart-Young-Mirsky low-rank approximation theorem directly on empirical gas sensor readings from the UCI Air Quality dataset.

All calculations are executed in-memory via the FastAPI backend gateway calling `src/airsense/math_core.py` and `src/airsense/pca_model.py`. The interface strictly prevents target leakage and temporal data snooping by fitting standardizers and SVD transformations exclusively on the 5,394-hour training partition (March–October 2004). Out-of-time validation observations (1,798 hours) are projected using frozen training parameters.

---

## 2. Interactive Analytical Sections

The workspace is organized into five coordinated sections:

### 1. Scree Plot & Explained Variance
- **Interactive Scree Plot**: Recharts composed bar and line chart showing:
  - Individual variance explained ratio: $\lambda_i / \sum_{j=1}^d \lambda_j \times 100\%$.
  - Cumulative variance curve.
  - Standard reference threshold lines (80%, 90%, 95%).
  - Retained components indicator ($k$). Clickable bars allow instant switching of $k$.
- **Spectral Variance Table**: Detailed metrics for each component:
  - Eigenvalue $\lambda_i$.
  - Singular value $\sigma_i$.
  - Individual and cumulative variance percentages.
  - Active retention status (`Retained` vs `Discarded`).
- **Kaiser-Guttman & Elbow Criterion Interpretation**: Explains why components with $\lambda_i > 1.0$ (typically the first 3 components, capturing $\sim 94\%$ of variance) contain true physical chemical signal, whereas components with $\lambda_i \ll 1.0$ capture noise.

### 2. Component Loadings Matrix
- **Diverging Matrix Visualizer**: Full $d \times d$ matrix of principal axes (columns of $V$):
  - Diverging color scale: Emerald/Teal tints for positive loadings, Rose/Amber tints for negative loadings.
  - Exact numerical loading values $v_{ij} \in [-1, 1]$.
  - Hover tooltips showing exact loadings and variance contributions $v_{ij}^2$.
- **Selected Component Inspector**:
  - Horizontal bar chart of directional loading coefficients $v_i$.
  - Feature contribution breakdown table: $v_{ij}^2 \times 100\%$ where $\sum_{i=1}^d v_{ij}^2 = 1.0$ (Unit Norm Invariant).
  - Chemical interpretation: e.g. PC1 captures general combustion pollution intensity with co-moving positive loadings on `PT08.S1`, `PT08.S2`, and `PT08.S4`, while PC2 captures seasonal temperature-humidity trade-offs.

### 3. Principal Component Score Explorer (2D Scatter)
- **Arbitrary Component Pair Projection**: Users can select any pair of principal components for the X and Y axes (e.g. PC1 vs PC2, PC1 vs PC3, PC2 vs PC3).
- **Subsampled Time-Series Observation Scatter**:
  - 280 representative hourly observations evenly stratified across the selected partition.
  - Continuous color mapping by certified reference Benzene `C6H6(GT)` concentration (teal = low, amber = moderate, red = high).
  - **Critical Caveat**: The target analyzer is treated strictly as an external visual annotation; it is never used in feature scaling or SVD fitting. Visible gradients along PC1 confirm that the dominant principal axis aligns strongly with true ambient hydrocarbon pollution without supervision.

### 4. Reconstruction Lab & Error Analysis
- **Dynamic Low-Rank Approximation**: User varies $k$ ($1 \le k \le d$) to observe how the reduced representation reconstructs the original feature matrix:
  $$\hat{Z}_k = T_k V_k^T = \sum_{j=1}^k t_j v_j^T, \quad \hat{X}_k = \mu_{\text{train}} + \hat{Z}_k \cdot \sigma_{\text{train}}$$
- **Eckart-Young Invariant Check**:
  - Empirical Frobenius reconstruction error: $\|Z - \hat{Z}_k\|_F$.
  - Theoretical bound: $\sqrt{\sum_{j=k+1}^d \sigma_j^2}$.
  - Machine precision parity: empirical error matches the theoretical singular value tail sum within $< 10^{-14}$.
- **48-Hour Continuous Sensor Reconstruction**:
  - Plots original physical sensor readings, rank-$k$ reconstructed signal, and reconstruction residuals across 48 continuous hours for any selected sensor channel.
- **Per-Feature Reconstruction Accuracy Table**: Reports physical RMSE, MAE, and relative error % for all sensor channels.

### 5. Mathematical Transparency & Pedagogical Derivations
- **SVD to Covariance Spectral Equivalence**:
  $$\lambda_i = \frac{\sigma_i^2}{n - 1} \iff \sigma_i = \sqrt{(n - 1)\lambda_i}$$
  Thin SVD of centered data $X_c = U \Sigma V^T$ directly diagonalizes the sample covariance matrix $S = \frac{1}{n-1} X_c^T X_c = V \frac{\Sigma^2}{n-1} V^T$.
- **The Eckart-Young-Mirsky Theorem**:
  $$\min_{\operatorname{rank}(B) \le k} \|X_c - B\|_F^2 = \|X_c - \hat{X}_k\|_F^2 = \sum_{j=k+1}^d \sigma_j^2$$
- **Uncorrelated Principal Scores**:
  $$\frac{1}{n - 1} T^T T = \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_k)$$
  The projected coordinates are completely decorrelated, eliminating multi-collinearity without discarding physical features.
- **Unsupervised Nature Caveat**: Retaining more variance does not guarantee higher calibration accuracy. Dropping the smallest 5 components removes sensor cross-sensitivity and thermal noise, enabling PCR ($k=3$) to outperform full OLS on out-of-time test data (RMSE 2.41 vs 2.85).

---

## 3. Preprocessing & Scaling Comparison

| Dimension | Standardized (Correlation PCA) | Unscaled (Covariance PCA) |
|---|---|---|
| **Transformation** | $Z = (X - \mu) / \sigma$ | $X_c = X - \mu$ |
| **Matrix Decomposed** | Correlation matrix $R$ | Covariance matrix $S$ |
| **PC1 Explained Variance** | $\sim 55.9\%$ | $\sim 88.5\%$ |
| **Dominant Features** | Balanced across chemical & environmental sensors | Resistance sensors (`PT08.S1`, `PT08.S5`) dominate |
| **Temperature Loading on PC1** | $-0.28$ (Meaningful) | $-0.0017$ (Virtually zero) |
| **Recommended For** | Multi-sensor arrays with heterogeneous physical units | Features with identical physical units and scales |

---

## 4. API Contract (`/api/v1/pca`)

### `POST /api/v1/pca/analyze`
**Request Payload**:
```json
{
  "features": ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"],
  "standardize": true,
  "n_components": 3,
  "partition": "train",
  "score_pc_x": 1,
  "score_pc_y": 2,
  "sample_size": 280,
  "reconstruction_feature": "PT08.S1(CO)"
}
```

**Response Payload**:
```json
{
  "selected_features": ["PT08.S1(CO)", "..."],
  "standardize": true,
  "partition": "train",
  "total_training_rows": 5394,
  "evaluated_partition_rows": 5394,
  "n_components": 3,
  "total_components": 8,
  "variance_table": [
    {"component": "PC1", "eigenvalue": 4.4719, "singular_value": 155.30, "variance_ratio_pct": 55.89, "cumulative_variance_pct": 55.89, "is_retained": true},
    {"component": "PC2", "eigenvalue": 1.7755, "singular_value": 97.85, "variance_ratio_pct": 22.19, "cumulative_variance_pct": 78.08, "is_retained": true},
    {"component": "PC3", "eigenvalue": 1.2769, "singular_value": 82.98, "variance_ratio_pct": 15.96, "cumulative_variance_pct": 94.04, "is_retained": true}
  ],
  "retained_variance_pct": 94.04,
  "discarded_variance_pct": 5.96,
  "loadings_matrix": [[-0.34, 0.43, "..."]],
  "reconstruction_error_frobenius": 50.7306,
  "theoretical_error_eckart_young": 50.7306,
  "relative_reconstruction_error": 0.2442,
  "feature_reconstruction": [{"feature": "PT08.S1(CO)", "rmse": 63.85, "mae": 49.91, "relative_error_pct": 5.7}],
  "scores_sample": [{"timestamp": "10/03/2004 18.00.00", "score_x": 1.25, "score_y": -0.84, "target": 13.6, "hour_idx": 0}],
  "time_series_reconstruction": [{"timestamp": "10/03/2004 18.00.00", "original": 1360.0, "reconstructed": 1342.1, "residual": 17.9}]
}
```

---

## 5. Verification Results

1. **Automated Backend Tests (`pytest`)**:
   - `test_api_pca_analyze_default`: Verifies $k=3$ captures $\sim 94.04\%$ variance and Eckart-Young empirical/theoretical error parity within $10^{-8}$.
   - `test_api_pca_analyze_unstandardized`: Verifies PC1 captures $> 85\%$ of variance in unscaled Covariance PCA.
   - `test_api_pca_analyze_validation_partition`: Verifies out-of-time validation partition transformation with training-fitted scaler.
   - `test_api_pca_analyze_component_pair_selection`: Verifies arbitrary component pair selection for score projection scatter.
   - **Full test suite**: **84 passed in 3.97s** with zero regressions.
2. **Frontend Production Build (`npm run build`)**:
   - TypeScript compilation (`tsc`): 0 errors, strict mode.
   - Vite bundle: `dist/assets/index-B7z3rTNV.js` built in 7.84s with zero warnings.
3. **Local Route**:
   - `http://localhost:5173/pca-explorer`
