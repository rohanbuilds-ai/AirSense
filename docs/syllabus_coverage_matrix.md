# AirSense: University Syllabus Coverage Matrix
## Mathematical Foundations for AI and Data Science

This matrix provides a line-by-line mapping for university examiners, technical reviewers, and students, tracing every syllabus requirement to:
1. The **exact implementation file** and function
2. The **exact demonstration notebook / Streamlit tab**
3. The **exact output visible to an examiner**
4. The **automated verification test** validating mathematical invariants

---

## Unit 1: Linear Systems, Vector Spaces & Matrix Transformations

| Syllabus Concept | Source Implementation | Demonstration File / Tab | Examiner Visible Output | Verification Test |
|---|---|---|---|---|
| **1.1 Real-World Matrix Representation** | `src/airsense/data_loader.py::<br>split_features_target` | `notebooks/01_dataset_audit.ipynb` (Cell 8)<br>`app.py` Section 2 Tab 1 | Design matrix $X \in \mathbb{R}^{8991 \times 8}$, Target vector $\mathbf{y} \in \mathbb{R}^{8991}$. Zero missing values in $X$. | `tests/test_data_loader.py::<br>test_feature_target_split_safety` |
| **1.2 Gaussian Elimination (Partial Pivoting)** | `src/airsense/math_core.py::<br>gaussian_elimination` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 3)<br>`app.py` Section 3 Tab 1 | Solves $A\mathbf{x} = \mathbf{b}$ for custom/preset matrices; residual $\|A\mathbf{x} - \mathbf{b}\| < 10^{-14}$. | `tests/test_math_core.py::<br>test_gaussian_elimination_deterministic` |
| **1.3 Reduced Row-Echelon Form (RREF)** | `src/airsense/math_core.py::<br>rref` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 5)<br>`app.py` Section 3 Tab 1 | Canonical RREF matrix with identified pivot column indices (e.g. pivots: `[0, 1, 2]`). | `tests/test_math_core.py::<br>test_rref_canonical_free_variables` |
| **1.4 LU Factorization ($PA = LU$)** | `src/airsense/math_core.py::<br>lu_factorize`, `lu_solve` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 7)<br>`app.py` Section 3 Tab 1 | Unit lower $L$, upper $U$, permutation $P$. Factorization error $\|PA - LU\|_\infty < 10^{-14}$. | `tests/test_math_core.py::<br>test_lu_factorize_identity_and_solve` |
| **1.5 Fundamental Subspaces ($C(A), N(A)$)** | `src/airsense/math_core.py::<br>null_space`, `column_space` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 9)<br>`app.py` Section 3 Tab 2 | Column space basis and Null space basis vectors satisfying $\|A N(A)\| < 10^{-14}$. | `tests/test_math_core.py::<br>test_null_space_full_rank`, `test_column_space_properties` |
| **1.6 Rank-Nullity Theorem** | `src/airsense/math_core.py::<br>matrix_rank`, `verify_rank_nullity` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 11)<br>`app.py` Section 3 Tab 2 | Verification badge: $\text{rank}(A) + \text{nullity}(A) = n$. Evaluated on $3 \times 4$ and $8,991 \times 8$ matrices. | `tests/test_math_core.py::<br>test_rank_nullity_theorem_verification` |
| **1.7 Linear Independence & Basis** | `src/airsense/math_core.py::<br>check_linear_independence` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 13)<br>`app.py` Section 3 Tab 2 | Boolean flag confirming whether vector sets are linearly independent with numerical tolerance. | `tests/test_math_core.py::<br>test_gram_schmidt_handles_dependent_columns` |
| **1.8 Linear Transformations** | `src/airsense/calibration.py::<br>FeatureStandardizer` | `notebooks/03_calibration_experiments.ipynb` (Cell 4)<br>`app.py` Section 5 | Diagonal scaling transformation $Z = (X - \mathbf{1}\boldsymbol{\mu}^T) D_{\sigma}^{-1}$. Mean 0, Variance 1. | `tests/test_calibration.py::<br>test_feature_standardizer_zero_leakage` |

---

## Unit 2: Inner Products, Orthogonality, Projections & Spectral Theory

| Syllabus Concept | Source Implementation | Demonstration File / Tab | Examiner Visible Output | Verification Test |
|---|---|---|---|---|
| **2.1 Inner Products, Norms & Cauchy-Schwarz** | `src/airsense/math_core.py::<br>dot_product`, `vector_norm` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 15)<br>`app.py` Section 3 | Euclidean norms $\|\mathbf{u}\|_2$, $\ell_1, \ell_\infty$. Cauchy-Schwarz inequality check $|\mathbf{u}^T \mathbf{v}| \le \|\mathbf{u}\|_2 \|\mathbf{v}\|_2$. | `tests/test_math_core.py::<br>test_airsense_real_data_mathematical_pipeline` |
| **2.2 Angles & Cosine Similarity** | `src/airsense/math_core.py::<br>cosine_similarity` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 16)<br>`app.py` Section 3 | Angular separation between MOX sensors: $r = 0.88 \implies \theta = 28.3^\circ$ (collinearity proof). | `tests/test_math_core.py::<br>test_airsense_real_data_mathematical_pipeline` |
| **2.3 Gram–Schmidt (CGS vs. MGS Stability)** | `src/airsense/math_core.py::<br>gram_schmidt_classical`, `gram_schmidt_modified` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 18)<br>`app.py` Section 3 Tab 3 | Side-by-side bar chart showing MGS maintains $\|Q^T Q - I\| < 10^{-8}$ while CGS loses orthogonality ($10^{-2}$). | `tests/test_math_core.py::<br>test_gram_schmidt_cgs_vs_mgs_stability` |
| **2.4 QR Decomposition** | `src/airsense/math_core.py::<br>qr_decomposition` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 20)<br>`app.py` Section 3 Tab 3 | $A = QR$ with $Q^T Q = I_d$ and upper-triangular $R$. Factorization error $< 10^{-12}$. | `tests/test_math_core.py::<br>test_gram_schmidt_modified_orthogonality_and_reconstruction` |
| **2.5 Orthogonal Projections** | `src/airsense/math_core.py::<br>projection_matrix`, `project_vector` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 22)<br>`app.py` Section 3 Tab 4 | Verification of idempotence $\|P^2 - P\| < 10^{-12}$, symmetry $\|P - P^T\| < 10^{-12}$, and orthogonality $X^T \mathbf{e} = \mathbf{0}$. | `tests/test_math_core.py::<br>test_projection_matrix_idempotence_and_symmetry`, `test_project_vector_residual_orthogonality` |
| **2.6 Least-Squares Solvers (QR & SVD)** | `src/airsense/math_core.py::<br>solve_least_squares_qr`, `solve_least_squares_svd` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 24)<br>`app.py` Section 3 Tab 5 | Stable solution avoiding condition squaring ($\kappa(X) = 1.89 \times 10^4$ vs. $\kappa(X^T X) = 3.57 \times 10^8$). | `tests/test_math_core.py::<br>test_least_squares_qr_and_svd_match_scipy` |
| **2.7 Sample Covariance & PSD** | `src/airsense/math_core.py::<br>compute_covariance`, `is_positive_semidefinite` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 26)<br>`app.py` Section 4 | $8 \times 8$ covariance matrix $S$. Verified symmetric and PSD (all eigenvalues $\lambda_i \ge 0$). | `tests/test_math_core.py::<br>test_covariance_and_psd_properties`, `test_positive_semidefinite_catches_negative_eigenvalue` |
| **2.8 Cyclic Jacobi Symmetric Diagonalization** | `src/airsense/math_core.py::<br>jacobi_eigenvalues_symmetric` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 28)<br>`app.py` Section 4 Tab 2 | Diagonalizes $S = V \Lambda V^T$. Matches SciPy `eigh` within $4.24 \times 10^{-13}$. | `tests/test_math_core.py::<br>test_jacobi_eigenvalues_symmetric_spectral_theorem` |
| **2.9 Singular Value Decomposition (SVD)** | `src/airsense/math_core.py::<br>pca_svd` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 30)<br>`app.py` Section 4 Tab 2 | Demonstrates $\lambda_i = \sigma_i^2 / (n-1)$ to within $10^{-14}$. Thin SVD $\tilde{X} = U \Sigma V^T$. | `tests/test_math_core.py::<br>test_pca_svd_covariance_equivalence_and_eckart_young` |
| **2.10 Low-Rank Approximation (Eckart-Young)** | `src/airsense/math_core.py::<br>low_rank_approximation` | `notebooks/02_linear_algebra_lab.ipynb` (Cell 32)<br>`app.py` Section 4 Tab 4 | Confirms $\|\tilde{X} - \hat{X}_k\|_F^2 = \sum_{j=k+1}^d \sigma_j^2$ across all $k \in [1, 7]$. | `tests/test_math_core.py::<br>test_low_rank_approximation` |

---

## Unit 3: Probability, Distributions, Estimation & Confidence Intervals

| Syllabus Concept | Source Implementation | Demonstration File / Tab | Examiner Visible Output | Verification Test |
|---|---|---|---|---|
| **3.1 Population, Sample & CLT** | `src/airsense/statistical_inference.py::<br>simulate_clt` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 4)<br>`app.py` Section 7 Tab 1 | Sampling distributions for $n \in [2, 100]$ showing convergence of skewed/bimodal data to Gaussian. | `tests/test_statistical_inference.py::<br>test_simulate_clt_convergence` |
| **3.2 Binomial to Normal Rule of Thumb** | `src/airsense/statistical_inference.py::<br>binomial_to_normal_approximation` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 6)<br>`app.py` Section 7 | Rule-of-thumb check $np \ge 10$ and $n(1-p) \ge 10$. Continuity correction demonstration. | `tests/test_statistical_inference.py::<br>test_binomial_to_normal_rule_of_thumb` |
| **3.3 Point Estimation & Normal MLE Bias** | `src/airsense/statistical_inference.py::<br>mle_normal` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 8)<br>`app.py` Section 7 Tab 2 | Small-sample bias proof: $\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$. Bessel's unbiased correction $s^2$. | `tests/test_statistical_inference.py::<br>test_mle_normal_bessel_correction` |
| **3.4 Bayesian MAP Regression (Ridge)** | `src/airsense/statistical_inference.py::<br>map_gaussian_regression` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 10)<br>`app.py` Section 7 | Proves MAP with prior $\mathcal{N}(\mathbf{0}, \tau^2 I)$ is identical to Ridge $\lambda = \sigma^2 / \tau^2$. Convergence as $\tau^2 \to \infty$. | `tests/test_statistical_inference.py::<br>test_map_gaussian_regression_ridge_equivalence` |
| **3.5 Confidence Intervals ($z$ vs. $t$)** | `src/airsense/statistical_inference.py::<br>confidence_interval_mean` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 12)<br>`app.py` Section 7 Tab 3 | Compares Student's $t$ and Normal $z$ intervals for small ($n=10$) vs. large ($n=500$) sample sizes. | `tests/test_statistical_inference.py::<br>test_confidence_interval_small_vs_large_sample` |
| **3.6 Margin of Error Scaling ($1/\sqrt{n}$)** | `src/airsense/statistical_inference.py::<br>margin_of_error_curve` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 14)<br>`app.py` Section 7 Tab 1 | Margin of error curve showing inverse square root diminishing returns as $n$ scales to 500. | `tests/test_statistical_inference.py::<br>test_margin_of_error_inverse_sqrt_scaling` |

---

## Unit 4: Hypothesis Testing, ANOVA & Statistical Power

| Syllabus Concept | Source Implementation | Demonstration File / Tab | Examiner Visible Output | Verification Test |
|---|---|---|---|---|
| **4.1 One-Sample $t$-test & CI Duality** | `src/airsense/statistical_inference.py::<br>one_sample_t_test` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 16)<br>`app.py` Section 7 Tab 3 | Formal proof: $\mu_0 \in \text{CI}_{1-\alpha} \iff \text{Fail to Reject } H_0$ at level $\alpha$ across residuals. | `tests/test_statistical_inference.py::<br>test_one_sample_t_test_and_ci_duality` |
| **4.2 Two-Sample Welch $t$-test** | `src/airsense/statistical_inference.py::<br>two_sample_t_test` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 18)<br>`app.py` Section 7 | Welch-Satterthwaite unequal variance $t$-test comparing Winter vs. Summer error distributions. | `tests/test_statistical_inference.py::<br>test_two_sample_welch_t_test` |
| **4.3 Paired $t$-test** | `src/airsense/statistical_inference.py::<br>paired_t_test` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 20)<br>`app.py` Section 7 | Tests paired difference $d_t = y_{\text{true}, t} - \hat{y}_t$, proving non-zero mean bias over time. | `tests/test_statistical_inference.py::<br>test_paired_t_test` |
| **4.4 One-Way ANOVA** | `src/airsense/statistical_inference.py::<br>one_way_anova` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 22)<br>`app.py` Section 7 Tab 4 | Partitions $SS_{\text{Total}} = SS_{\text{Between}} + SS_{\text{Within}}$. $F = 214.3, p < 10^{-15}, \eta^2 = 0.067$. | `tests/test_statistical_inference.py::<br>test_one_way_anova_ss_partitioning_and_eta_squared` |
| **4.5 Chi-Squared Contingency Test** | `src/airsense/statistical_inference.py::<br>chi_squared_contingency_test` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 24)<br>`app.py` Section 7 | Contingency table of pollution levels vs. seasons. Cochran minimum cell count check ($E_{ij} \ge 5$) & Cramér's $V$. | `tests/test_statistical_inference.py::<br>test_chi_squared_contingency_cochran_and_cramers_v` |
| **4.6 Statistical Power Simulation** | `src/airsense/statistical_inference.py::<br>simulate_power_curve` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 26)<br>`app.py` Section 7 Tab 5 | Power curve ($1 - \beta$) varying sample size $n \in [5, 100]$ and Cohen's effect size $d \in [0.2, 0.8]$. | `tests/test_statistical_inference.py::<br>test_simulate_power_curve_type1_error` |
| **4.7 Modeling Assumption Diagnostics** | `src/airsense/statistical_inference.py::<br>diagnose_assumptions` | `notebooks/04_probability_and_statistics_lab.ipynb` (Cell 28)<br>`app.py` Section 8 Tab 3 | Shapiro-Wilk ($p < 10^{-5}$), D'Agostino-Pearson, Levene, and Durbin-Watson tests on residuals. | `tests/test_statistical_inference.py::<br>test_diagnose_assumptions_catches_non_normality` |
