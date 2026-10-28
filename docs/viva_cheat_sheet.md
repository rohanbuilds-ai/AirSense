# AirSense — Viva Defense Cheat-Sheet & Oral Exam Guide
## Mathematical Foundations for AI and Data Science

This cheat-sheet provides concise, high-impact viva answers formatted according to the required explanatory paradigm:
$$\mathbf{Concept} \longrightarrow \mathbf{Purpose} \longrightarrow \mathbf{Outcome}$$

---

## 1. Matrix Representation of Physical Sensor Systems

- **Concept**:  
  A continuous physical environment measured at discrete hourly intervals across multiple sensor channels is represented as a real design matrix $X \in \mathbb{R}^{n \times d}$, where each row $\mathbf{x}_i^T \in \mathbb{R}^d$ corresponds to a joint observation at time $t_i$, and each column $\mathbf{x}_{*j} \in \mathbb{R}^n$ represents the temporal trajectory of the $j$-th chemical or meteorological variable.
- **Purpose**:  
  To formulate physical sensor calibration, cross-sensitivity unmixing, and environmental correction as linear algebraic operations over Euclidean vector spaces.
- **Outcome**:  
  AirSense models $n = 8,991$ supervised hours across $d = 8$ predictive channels ($5$ metal-oxide chemical sensors and $3$ ambient variables: $T, RH, AH$) predicting certified reference benzene $C_6H_6(GT) \in \mathbb{R}^n$.

---

## 2. Gaussian Elimination, Pivoting, and LU Decomposition

- **Concept**:  
  Elementary row operations (row swapping $E_{\text{swap}}$, row scaling $E_{\text{scale}}$, and row replacement $E_{\text{add}}$) reduce a square non-singular matrix $A \in \mathbb{R}^{m \times m}$ into an upper-triangular matrix $U$. When combined with row permutations $P$ (partial pivoting: selecting the maximum absolute pivot $|a_{kk}|$), the elimination process factorizes $A$ into:
  $$PA = LU$$
  where $L$ is a unit lower-triangular matrix of row multipliers ($l_{ik}$) and $U$ is upper-triangular.
- **Purpose**:  
  To solve linear systems $A\mathbf{x} = \mathbf{b}$ in $\mathcal{O}(m^3)$ operations without explicitly inverting matrices, avoiding catastrophic division by zero or pivot-induced floating-point amplification.
- **Outcome**:  
  Implemented from scratch in `airsense.math_core.lu_factorize` and `lu_solve`. Verified against SciPy with $\|PA - LU\|_\infty < 10^{-12}$. Used for exact forward and backward substitution.

---

## 3. Fundamental Subspaces and the Rank-Nullity Theorem

- **Concept**:  
  Every linear operator $A \in \mathbb{R}^{m \times n}$ partitions $\mathbb{R}^n$ and $\mathbb{R}^m$ into Four Fundamental Subspaces:
  1. **Column Space $C(A) \subseteq \mathbb{R}^m$**: The span of the columns; dimension equals $\text{rank}(A) = r$.
  2. **Null Space $N(A) \subseteq \mathbb{R}^n$**: All vectors satisfying $A\mathbf{x} = \mathbf{0}$; dimension equals $\text{nullity}(A) = n - r$.
  3. **Row Space $C(A^T) \subseteq \mathbb{R}^n$**: The span of the rows; dimension equals $r$.
  4. **Left Null Space $N(A^T) \subseteq \mathbb{R}^m$**: All vectors satisfying $A^T \mathbf{y} = \mathbf{0}$; dimension equals $m - r$.
  
  **Rank-Nullity Theorem**:
  $$\text{rank}(A) + \text{nullity}(A) = n$$
  Furthermore, the Row Space and Null Space are orthogonal complements in $\mathbb{R}^n$: $C(A^T) \perp N(A)$ and $C(A^T) \oplus N(A) = \mathbb{R}^n$.
- **Purpose**:  
  To identify redundant or linearly dependent sensor channels, verify the existence and uniqueness of calibration solutions, and project signals into irreducible signal/noise subspaces.
- **Outcome**:  
  Implemented in `airsense.math_core.matrix_rank` and `null_space`. Verified that the $8,991 \times 8$ AirSense feature matrix has full column rank $r = 8$ and nullity $0$, but exhibits near-deficiency (extreme collinearity among MOX sensors).

---

## 4. Metric Spaces, Vector Norms, Cauchy-Schwarz, and Inner Products

- **Concept**:  
  An inner product on $\mathbb{R}^n$ is a positive-definite symmetric bilinear form $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^n u_i v_i$, inducing the Euclidean metric ($L_2$ norm) $\|\mathbf{u}\|_2 = \sqrt{\langle \mathbf{u}, \mathbf{u} \rangle}$.  
  The Cauchy-Schwarz Inequality guarantees:
  $$|\langle \mathbf{u}, \mathbf{v} \rangle| \le \|\mathbf{u}\|_2 \|\mathbf{v}\|_2$$
  with equality if and only if $\mathbf{u}$ and $\mathbf{v}$ are collinear, defining the geometric angle $\cos \theta = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$.
- **Purpose**:  
  To quantify sensor correlation as geometric angles between feature vectors in $\mathbb{R}^n$ and measure prediction errors using residual vector lengths $\|\mathbf{e}\|_2 = \sqrt{n} \cdot \text{RMSE}$.
- **Outcome**:  
  Proves that high Pearson correlation ($r > 0.88$) between `PT08.S1(CO)` and `PT08.S2(NMHC)` corresponds to a small geometric angle $\theta \approx 28.3^\circ$, explaining collinearity and ill-conditioning in feature space.

---

## 5. Gram–Schmidt Orthogonalization (CGS vs. MGS) and QR Decomposition

- **Concept**:  
  Given linearly independent vectors $\{\mathbf{a}_1, \dots, \mathbf{a}_d\}$, Gram-Schmidt constructs an orthonormal basis $\{\mathbf{q}_1, \dots, \mathbf{q}_d\}$ spanning the same subspace.  
  - **Classical Gram-Schmidt (CGS)**: Projects the raw vector $\mathbf{a}_k$ simultaneously onto all previous basis vectors:
    $$\mathbf{q}_k' = \mathbf{a}_k - \sum_{j=1}^{k-1} \langle \mathbf{q}_j, \mathbf{a}_k \rangle \mathbf{q}_j$$
  - **Modified Gram-Schmidt (MGS)**: Projects successively updated intermediate vectors:
    $$\mathbf{v}_k^{(j)} = \mathbf{v}_k^{(j-1)} - \langle \mathbf{q}_j, \mathbf{v}_k^{(j-1)} \rangle \mathbf{q}_j$$
  Yields the factorization $A = QR$, where $Q \in \mathbb{R}^{n \times d}$ has orthonormal columns ($Q^T Q = I_d$) and $R \in \mathbb{R}^{d \times d}$ is upper-triangular.
- **Purpose**:  
  To avoid catastrophic cancellation in floating-point arithmetic when orthogonalizing near-collinear sensor signals.
- **Outcome**:  
  AirSense empirically proves that on ill-conditioned matrices ($\kappa \approx 10^5$), CGS loses orthogonality with $\|Q^T Q - I\|_F \approx 1.88 \times 10^{-2}$, whereas MGS preserves orthogonality to $\|Q^T Q - I\|_F \approx 2.66 \times 10^{-9}$—a **$7.07 \times 10^6$-fold precision gain**.

---

## 6. Orthogonal Projections and Least-Squares Approximation

- **Concept**:  
  The orthogonal projection of an observation vector $\mathbf{y} \in \mathbb{R}^n$ onto the column space $C(X)$ minimizes the Euclidean distance $\|\mathbf{y} - \hat{\mathbf{y}}\|_2$. The projection operator is:
  $$P = X(X^T X)^{-1} X^T$$
  which satisfies symmetry ($P = P^T$) and idempotence ($P^2 = P$).  
  The residual vector $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = (I - P)\mathbf{y}$ is orthogonal to every column in $X$:
  $$X^T \mathbf{e} = \mathbf{0} \implies X^T (\mathbf{y} - X\mathbf{w}) = \mathbf{0} \implies (X^T X)\mathbf{w} = X^T \mathbf{y} \quad (\text{Normal Equations})$$
- **Purpose**:  
  To compute optimal linear calibration coefficients $\mathbf{w}$ that minimize Sum of Squared Errors.
- **Outcome**:  
  Rather than forming the Normal Equations directly—which squares the condition number $\kappa(X^T X) = \kappa(X)^2 \approx (1.89 \times 10^4)^2 = 3.57 \times 10^8$ and loses $\sim 8$ digits of precision—AirSense uses QR decomposition:
  $$R\mathbf{w} = Q^T \mathbf{y}$$
  which preserves $\kappa(X) = 1.89 \times 10^4$ and guarantees numerical stability.

---

## 7. Sample Covariance, Positive Semi-Definiteness, and the Spectral Theorem

- **Concept**:  
  For centered data $\tilde{X} = X - \mathbf{1}\boldsymbol{\mu}^T$, the sample covariance matrix is:
  $$S = \frac{1}{n-1}\tilde{X}^T \tilde{X} \in \mathbb{R}^{d \times d}$$
  $S$ is symmetric ($S = S^T$) and positive semi-definite (PSD): $\mathbf{z}^T S \mathbf{z} = \frac{1}{n-1}\|\tilde{X}\mathbf{z}\|_2^2 \ge 0$ for all $\mathbf{z} \in \mathbb{R}^d$.  
  By the **Spectral Theorem**, every real symmetric matrix has real eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$ and an orthonormal basis of eigenvectors $V = [\mathbf{v}_1, \dots, \mathbf{v}_d]$:
  $$S = V \Lambda V^T \quad \text{with } V^T V = I$$
- **Purpose**:  
  To uncouple correlated sensor signals by finding the principal axes of the feature dispersion ellipsoid.
- **Outcome**:  
  AirSense implements Cyclic Jacobi rotations from scratch (`airsense.math_core.jacobi_eigenvalues_symmetric`), diagonalizing the $8 \times 8$ covariance matrix with machine-precision parity to SciPy `eigh` (error $< 4.24 \times 10^{-13}$).

---

## 8. Singular Value Decomposition (SVD) and Principal Component Analysis (PCA)

- **Concept**:  
  Every real matrix $\tilde{X} \in \mathbb{R}^{n \times d}$ has a Singular Value Decomposition:
  $$\tilde{X} = U \Sigma V^T$$
  where $U \in \mathbb{R}^{n \times d}$ and $V \in \mathbb{R}^{d \times d}$ have orthonormal columns, and $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_d)$ contains non-negative singular values $\sigma_1 \ge \dots \ge \sigma_d \ge 0$.  
  The SVD of $\tilde{X}$ is directly connected to the spectral decomposition of covariance $S$:
  $$S = \frac{1}{n-1}\tilde{X}^T \tilde{X} = \frac{1}{n-1} V \Sigma U^T U \Sigma V^T = V \left(\frac{\Sigma^2}{n-1}\right) V^T \implies \lambda_i = \frac{\sigma_i^2}{n-1}$$
  The $k$-dimensional PCA projection is $Z_k = \tilde{X} V_k$. By the **Eckart-Young-Mirsky Theorem**, the rank-$k$ truncated matrix $\hat{X}_k = U_k \Sigma_k V_k^T$ minimizes the Frobenius reconstruction error:
  $$\|\tilde{X} - \hat{X}_k\|_F^2 = \sum_{j=k+1}^d \sigma_j^2$$
- **Purpose**:  
  To compress $8$ correlated sensor channels into $k < 8$ uncorrelated components that capture maximal variance while filtering out collinear noise.
- **Outcome**:  
  First 3 components capture $94.04\%$ of total sensor variance ($k=1$: $55.89\%$, $k=2$: $22.19\%$, $k=3$: $15.96\%$). Under the validation selection protocol, $k^*=8$ (Full OLS) was selected ($\text{Val RMSE} = 2.024\ \mu g/m^3$). Post-hoc evaluation on the test set showed $k=3$ achieved lower out-of-time test error ($\text{Test } R^2 = 0.8626$ vs. $0.8079$), demonstrating the regularizing effect of PCA against seasonal drift.

---

## 9. Maximum Likelihood Estimation (MLE) and Bayesian MAP Regularization

- **Concept**:  
  - **Normal MLE**: Assuming Gaussian noise $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$, the log-likelihood is:
    $$\ln L(\mathbf{w}, \sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_{i=1}^n (y_i - \mathbf{x}_i^T \mathbf{w})^2$$
    Maximizing with respect to $\mathbf{w}$ yields the OLS estimate $\hat{\mathbf{w}}_{\text{MLE}} = (X^T X)^{-1}X^T \mathbf{y}$.  
    Maximizing with respect to $\sigma^2$ yields the biased estimator $\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum e_i^2$, which has expectation $\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2$. Bessel's correction restores unbiasedness: $s^2 = \frac{n}{n-1}\hat{\sigma}^2_{\text{MLE}}$.
  - **Bayesian MAP**: Placing a zero-mean Gaussian prior on weights $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$ yields the Maximum A Posteriori (MAP) estimate:
    $$\hat{\mathbf{w}}_{\text{MAP}} = \arg\max \left[ \ln p(\mathbf{y} \mid X, \mathbf{w}) + \ln p(\mathbf{w}) \right] = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$$
    This proves that **Ridge Regression is mathematically identical to Bayesian MAP estimation under a Gaussian prior**, with shrinkage penalty $\lambda = \frac{\sigma^2}{\tau^2}$.
- **Purpose**:  
  To provide probabilistic foundations for least-squares fitting and formalize weight shrinkage as prior belief against extreme, collinear sensor coefficients.
- **Outcome**:  
  Implemented in `airsense.statistical_inference.map_gaussian_regression`. Demonstrates that as prior uncertainty $\tau^2 \to \infty$ ($\lambda \to 0$), MAP weights converge exactly to OLS weights.

---

## 10. Statistical Hypothesis Testing, Confidence Intervals, and Power Analysis

- **Concept**:  
  - **Duality**: A two-sided test of $H_0: \mu = \mu_0$ at level $\alpha$ rejects $H_0$ if and only if $\mu_0$ lies outside the $(1 - \alpha)$ confidence interval:
    $$\text{Reject } H_0 \iff \mu_0 \notin \left[\bar{x} \pm t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}\right]$$
  - **ANOVA**: Partitions total variation into between-group and within-group sums of squares ($SS_{\text{Total}} = SS_{\text{Between}} + SS_{\text{Within}}$), testing equality of seasonal pollutant means via $F = \frac{MS_{\text{Between}}}{MS_{\text{Within}}}$.
  - **Power ($1 - \beta$)**: The probability of correctly rejecting a false null hypothesis for a given effect size $d = \frac{\mu_1 - \mu_0}{\sigma}$ and sample size $n$.
- **Purpose**:  
  To evaluate whether calibration error shifts are statistically significant and ensure studies have sufficient sample size to detect sensor drift.
- **Outcome**:  
  - Verified exact mathematical duality on AirSense residuals across $\alpha \in \{0.01, 0.05, 0.10\}$.
  - ANOVA confirmed significant seasonal variation in benzene concentration ($F = 214.3, p < 10^{-15}, \eta^2 = 0.067$).
  - Power analysis proved that detecting a subtle sensor bias shift of $0.5\ \mu g/m^3$ with $90\%$ power requires $n \ge 85$ independent observations.
