# AirSense: Viva Voce Defense & Technical Examination Pack

This document contains 15 technical viva voce examination questions, common examiner pitfalls, and structured model responses for the **Mathematical Foundations for AI and Data Science** undergraduate research defense.

---

## 1. The Viva Examination Paradigm

Every mathematical concept is articulated through the examiner-aligned triad:
$$\mathbf{Concept} \longrightarrow \mathbf{Purpose} \longrightarrow \mathbf{Outcome}$$

---

## 2. In-Depth Viva Questions & Defense Answers

### Q1: How is a physical multi-sensor environmental station represented as a matrix, and what do its algebraic dimensions signify?
- **Concept**: A continuous physical field sampled at $n$ discrete hourly intervals across $d$ sensor channels forms a real design matrix $X \in \mathbb{R}^{n \times d}$. Each row $\mathbf{x}_i^T \in \mathbb{R}^d$ represents a joint physical state at timestamp $t_i$. Each column $\mathbf{x}_{*j} \in \mathbb{R}^n$ represents the temporal trajectory of the $j$-th chemical or meteorological variable.
- **Purpose**: To formulate physical sensor unmixing, cross-sensitivity correction, and calibration as geometric operations over Euclidean vector spaces.
- **Outcome**: AirSense processes $n = 8,991$ supervised hours across $d = 8$ predictor channels ($5$ metal-oxide sensors and $3$ ambient variables: $T, RH, AH$) to calibrate against certified benzene $C_6H_6(GT) \in \mathbb{R}^n$.

---

### Q2: Why did you exclude all reference analyzer columns ending in `(GT)` from your input features?
- **Model Answer**:
  Columns ending in `(GT)` (`CO(GT)`, `NMHC(GT)`, `NOx(GT)`, `NO2(GT)`) represent certified reference analyzers co-located at the urban station for ground-truth validation.
  1. **Target Leakage**: These instruments measure chemical species that co-vary strongly with benzene. Including them would mask the cross-sensitivities of the low-cost sensors we intend to calibrate.
  2. **Operational Realism**: In an operational deployment, certified analyzers (costing \$30,000+) are absent. A low-cost sensor calibration engine must predict pollutants using low-cost sensor resistances and weather variables only.
  3. **Sensor Defect**: `NMHC(GT)` failed after May 2004 ($90.23\%$ missing values), making it unusable in practice.

---

### Q3: What is Gaussian elimination with partial pivoting, and why is pivoting essential?
- **Concept**: Elementary row operations (row interchanges, row scaling, row additions) transform $A \in \mathbb{R}^{n \times n}$ into an upper-triangular matrix $U$. Partial pivoting searches the active column $k$ for the element with the largest absolute magnitude $|a_{ik}|$ ($i \ge k$) and swaps row $i$ with row $k$ before eliminating sub-diagonal elements.
- **Purpose**:
  1. Prevents catastrophic division by zero if a diagonal pivot is zero ($a_{kk} = 0$).
  2. Mitigates floating-point error magnification when dividing by small pivots ($|a_{kk}| \ll 1$), ensuring numerical stability.
- **Outcome**: Yields the exact matrix factorization $PA = LU$, where $P$ is a permutation matrix, $L$ is unit lower-triangular multipliers, and $U$ is upper-triangular. Verified $\|PA - LU\|_\infty < 10^{-12}$ against SciPy.

---

### Q4: State the Rank-Nullity Theorem and explain its significance in sensor arrays.
- **Concept**: For any linear operator $A \in \mathbb{R}^{m \times n}$:
  $$\text{rank}(A) + \text{nullity}(A) = n$$
  where $\text{rank}(A) = \text{dim}(C(A))$ is the dimension of the column space, and $\text{nullity}(A) = \text{dim}(N(A))$ is the dimension of the null space ($A\mathbf{x} = \mathbf{0}$).
- **Purpose**: To detect redundant or collinear sensor channels. If a sensor array has non-zero nullity ($\text{nullity}(A) > 0$), at least one sensor's response can be written as an exact linear combination of the others, providing zero independent information.
- **Outcome**: The $8,991 \times 8$ AirSense feature matrix has full column rank $r = 8$ and nullity $0$, proving that no sensor channel is completely redundant, although strong collinearity makes the matrix ill-conditioned.

---

### Q5: Why is Modified Gram-Schmidt (MGS) preferred over Classical Gram-Schmidt (CGS)?
- **Concept**: Gram-Schmidt orthogonalizes linearly independent vectors $\{\mathbf{a}_1, \dots, \mathbf{a}_d\}$ to form an orthonormal basis $Q$ ($Q^T Q = I$).
  - **CGS**: Projects $\mathbf{a}_k$ simultaneously against all previous basis vectors:
    $$\mathbf{q}_k' = \mathbf{a}_k - \sum_{j=1}^{k-1} \langle \mathbf{q}_j, \mathbf{a}_k \rangle \mathbf{q}_j$$
  - **MGS**: Successively projects intermediate updated residuals:
    $$\mathbf{v}_k^{(j)} = \mathbf{v}_k^{(j-1)} - \langle \mathbf{q}_j, \mathbf{v}_k^{(j-1)} \rangle \mathbf{q}_j$$
- **Purpose**: In exact arithmetic, CGS and MGS are mathematically identical. But in finite IEEE 754 floating-point arithmetic, subtracting two nearly equal numbers in CGS causes catastrophic cancellation, destroying orthogonality.
- **Outcome**: When tested on near-collinear vectors ($\epsilon = 10^{-5}$), CGS loses orthogonality with $\|Q^T Q - I\|_F \approx 1.88 \times 10^{-2}$, whereas MGS maintains orthogonality to $\|Q^T Q - I\|_F \approx 2.66 \times 10^{-9}$—a **$7.07 \times 10^6$-fold precision gain**.

---

### Q6: Why did you solve least squares via QR decomposition rather than the Normal Equations?
- **Concept**: The 2-norm condition number is $\kappa(X) = \frac{\sigma_{\max}}{\sigma_{\min}}$. The Normal Equations matrix is $X^T X$, whose condition number is:
  $$\kappa(X^T X) = [\kappa(X)]^2$$
- **Purpose**: Squaring the condition number destroys significant digits. In AirSense:
  $$\kappa(X) \approx 1.89 \times 10^4 \implies \kappa(X^T X) \approx 3.57 \times 10^8$$
  In double precision (15–17 digits), forming $X^T X$ loses $\log_{10}(3.57 \times 10^8) \approx 8.5$ significant digits.
- **Outcome**: QR decomposition factors $X = QR$ where $Q^T Q = I$, yielding $R\mathbf{w} = Q^T \mathbf{y}$. Because $Q$ has orthonormal columns, $\kappa(R) = \kappa(X) = 1.89 \times 10^4$, solving the system without condition squaring.

---

### Q7: Explain the geometric properties of the orthogonal projection matrix $P$.
- **Concept**: The orthogonal projector onto the column space $C(X)$ is:
  $$P = X(X^T X)^{-1} X^T$$
- **Two Invariants**:
  1. **Symmetry**: $P^T = [X(X^T X)^{-1} X^T]^T = X(X^T X)^{-1} X^T = P$.
  2. **Idempotence**: $P^2 = P \cdot P = X(X^T X)^{-1} X^T X (X^T X)^{-1} X^T = X(X^T X)^{-1} X^T = P$.
  3. **Residual Orthogonality**: The error vector $\mathbf{e} = (I - P)\mathbf{y}$ is strictly perpendicular to every column of $X$:
     $$X^T \mathbf{e} = X^T (I - P)\mathbf{y} = (X^T - X^T X(X^T X)^{-1} X^T)\mathbf{y} = (X^T - X^T)\mathbf{y} = \mathbf{0}$$
- **Outcome**: AirSense verified $\|P^T - P\| < 10^{-12}$, $\|P^2 - P\| < 10^{-12}$, and $\|X^T \mathbf{e}\| < 10^{-10}$ on real sensor arrays.

---

### Q8: How is SVD mathematically related to the spectral decomposition of the covariance matrix?
- **Concept**: Let $\tilde{X} \in \mathbb{R}^{n \times d}$ be centered, standardized data. Its thin SVD is $\tilde{X} = U \Sigma V^T$.
  The sample covariance matrix is:
  $$S = \frac{1}{n-1}\tilde{X}^T \tilde{X} = \frac{1}{n-1} (V \Sigma U^T)(U \Sigma V^T) = V \left(\frac{\Sigma^2}{n-1}\right) V^T$$
  Because $V^T V = I$ and $\Sigma^2 / (n-1)$ is diagonal, this is the unique spectral decomposition $S = V \Lambda V^T$.
- **Conclusion**:
  1. The right singular vectors $V$ of the data matrix are exactly the eigenvectors of the covariance matrix.
  2. The eigenvalues of the covariance matrix are related to singular values by:
     $$\lambda_i = \frac{\sigma_i^2}{n-1} \iff \sigma_i = \sqrt{(n-1)\lambda_i}$$
- **Outcome**: Verified in `airsense.math_core.pca_svd` to within machine precision ($< 10^{-14}$).

---

### Q9: State the Eckart-Young-Mirsky Theorem and its role in PCA dimensionality reduction.
- **Concept**: For any matrix $\tilde{X}$ with SVD $\sum_{i=1}^d \sigma_i \mathbf{u}_i \mathbf{v}_i^T$, the truncated rank-$k$ matrix:
  $$\hat{X}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T = U_k \Sigma_k V_k^T$$
  minimizes the Frobenius reconstruction error over all matrices of rank at most $k$:
  $$\min_{\text{rank}(B) \le k} \|\tilde{X} - B\|_F^2 = \|\tilde{X} - \hat{X}_k\|_F^2 = \sum_{j=k+1}^d \sigma_j^2$$
- **Purpose**: Proves that PCA truncation is mathematically optimal in the least-squares sense: discarding the last $d - k$ singular values discards the minimum possible energy.
- **Outcome**: AirSense unit tests confirm that empirical Frobenius error $\|\tilde{X} - \hat{X}_k\|_F^2$ matches the singular value tail sum exactly.

---

### Q10: How do you interpret the back-projected physical feature weights $\mathbf{w}_{\text{feature}}$ in PCR?
- **Concept**: Principal Component Regression computes weights $\mathbf{w}_{\text{pca}} \in \mathbb{R}^k$ on standardized principal components:
  $$\hat{\mathbf{y}} = w_0 + Z_k \mathbf{w}_{\text{pca}} = w_0 + (X - \mathbf{1}\boldsymbol{\mu}^T) D_\sigma^{-1} V_k \mathbf{w}_{\text{pca}}$$
- **Formula**:
  $$\mathbf{w}_{\text{feature}} = D_\sigma^{-1} V_k \mathbf{w}_{\text{pca}} \in \mathbb{R}^d$$
- **Purpose**: Raw principal components are abstract orthogonal linear combinations. Back-projecting through $D_\sigma^{-1} V_k$ maps model sensitivities back into physical engineering units ($\mu g/m^3$ change per raw sensor unit), restoring interpretability.

---

### Q11: Which model was selected by your validation protocol, and why does PCR with $k=3$ or $k=4$ perform better on the out-of-time test set?
- **Model Answer**:
  - **Selected Model ($k^* = 8$)**: Under our strict validation protocol, hyperparameter selection was conducted strictly on the validation set (Oct 2004 – Jan 2005). The stated rule selected **Full OLS / PCR ($k^* = 8$)** because it minimized validation error ($\text{Val RMSE} = 2.024\ \mu g/m^3$, $\text{Val } R^2 = 0.9423$).
  - **Out-of-Time Test Assessment**: When evaluated on the untouched out-of-time test set (Jan – Apr 2005), the selected model degraded to $\text{Test } R^2 = 0.8079$ ($\text{RMSE} = 2.855\ \mu g/m^3$).
  - **Post-Hoc Regularization Finding**: Evaluating the full parameter sweep on the test set showed that compact models ($k=3$ and $k=4$) achieved lower test error ($\text{RMSE} = 2.415\ \mu g/m^3$, $R^2 = 0.8626$).
  - **Scientific Explanation**: Discarding the trailing collinear components ($\sigma_7, \sigma_8$) acted as an implicit shrinkage regularizer. The trailing components fit transient seasonal interactions in the training data that inverted during the test period.
  - **Methodological Nuance for Examiners**: In an honest academic defense, you must state that $k=3$ is a **post-hoc empirical observation**, not the validation-selected model. Calling $k=3$ the selected model based on test scores would constitute retrospective test-set leakage. Selecting $k=3$ in production would require a validation protocol that explicitly evaluates across seasonal cycles.

---

### Q12: What is the "Collinear Overfit Trap" observed at $k=7$?
- **Model Answer**:
  Retaining $k=7$ components ($99.69\%$ variance) produces an impressive validation fit ($\text{Val } R^2 = 0.9123$, $\text{RMSE} = 2.496\ \mu g/m^3$). However, when evaluated on the out-of-time test set, performance collapses to $\text{Test } R^2 = \mathbf{0.6225}$ ($\text{RMSE} = 4.003\ \mu g/m^3$).
  The 7th component has low variance ($\lambda_7 = 0.037$) and captures spurious correlations between sensor temperature coefficients and summer atmospheric stability. Over annual cycles, these correlations invert, causing catastrophic out-of-time prediction errors.

---

### Q13: Can you claim you have statistically proven sensor drift?
- **Model Answer**:
  **No. We must not claim that sensor drift has been conclusively proven.**
  While calibration residuals drift negative by **$-4.13\ \mu g/m^3$** over 13 months, outdoor field measurements are confounded by annual weather shifts:
  1. Summer high temperatures ($40^\circ\text{C}$) transition to Winter freezing temperatures ($0^\circ\text{C}$).
  2. Absolute humidity changes surface chemisorption equilibrium.
  3. Real sensors follow power-law kinetics ($R \propto C^{-\alpha}$) rather than linear planes.
  Without periodic recalibration in a controlled laboratory chamber using standard reference gas cylinders, the empirical data strictly demonstrates **changing error dynamics and temporal non-stationarity**, not isolated physical drift.

---

### Q14: How does Bayesian Maximum A Posteriori (MAP) regression connect to Ridge regression?
- **Concept**:
  - Likelihood: $\mathbf{y} \mid X, \mathbf{w} \sim \mathcal{N}(X\mathbf{w}, \sigma^2 I)$.
  - Prior: $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$.
  - The posterior log-probability is:
    $$\ln p(\mathbf{w} \mid X, \mathbf{y}) \propto -\frac{1}{2\sigma^2}\|\mathbf{y} - X\mathbf{w}\|_2^2 - \frac{1}{2\tau^2}\|\mathbf{w}\|_2^2$$
  - Maximizing with respect to $\mathbf{w}$ yields:
    $$\hat{\mathbf{w}}_{\text{MAP}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y} \quad \text{where } \lambda = \frac{\sigma^2}{\tau^2}$$
- **Conclusion**: Ridge regression is mathematically identical to Bayesian MAP estimation under a zero-mean Gaussian prior, where the shrinkage penalty $\lambda$ is the ratio of noise variance to prior uncertainty.

---

### Q15: What does the Durbin-Watson statistic tell us about our calibration residuals?
- **Concept**: The Durbin-Watson statistic tests for first-order autocorrelation in residuals:
  $$DW = \frac{\sum_{t=2}^N (e_t - e_{t-1})^2}{\sum_{t=1}^N e_t^2} \approx 2(1 - \hat{\rho})$$
  $DW = 2.0$ indicates zero serial correlation; $DW < 1.0$ indicates strong positive serial correlation.
- **Empirical Value**: AirSense calibration residuals produce $DW = \mathbf{0.2062} \ll 2.0$ ($\hat{\rho} \approx 0.897$).
- **Implication**: Hourly sensor calibration errors are strongly autocorrelated. Standard ordinary least squares (OLS) confidence intervals assume i.i.d. errors and will severely underestimate parameter uncertainty (too narrow by $3\times$ to $5\times$). Robust Newey-West (HAC) standard errors must be used in operational reporting.

---

## 3. Common Examiner Pitfalls to Avoid

| ❌ Trap to Avoid | ✅ Scientifically Rigorous Response |
|---|---|
| *"We used PCA because it always improves model accuracy."* | *"PCA is a variance regularizer. On training data, PCA strictly reduces $R^2$. Its value lies in out-of-time generalization by discarding collinear noise."* |
| *"Sensor drift is proven because residuals drifted negative."* | *"Residuals drifted by $-4.13\ \mu g/m^3$, demonstrating changing error behavior. However, seasonal temperature and humidity confounding prevent claiming isolated physical drift without chamber ground-truth."* |
| *"The p-value is the probability that the null hypothesis is true."* | *"The p-value is the probability of observing a test statistic at least as extreme as the one observed, assuming the null hypothesis is true."* |
| *"We inverted $X^T X$ to find the optimal weights."* | *"We never invert $X^T X$ directly. Condition number squaring ($\kappa = 3.57 \times 10^8$) destroys precision. We solve $R\mathbf{w} = Q^T \mathbf{y}$ via QR decomposition."* |
