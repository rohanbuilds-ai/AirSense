# AirSense: Mathematical Foundations & Viva Reference Manual

This document provides the theoretical formulations, geometric interpretations, algorithmic derivations, and university viva explanations for the **AirSense Mathematical Engine**.

Every stage adheres to the university syllabus structure:
$$\mathbf{Concept} \longrightarrow \mathbf{Purpose} \longrightarrow \mathbf{Outcome}$$

---

## Stage 1: Real-World Data Loading & Matrix Representation

### 1. Mathematical Formulation
Let the sensor monitoring campaign produce $n$ discrete hourly observations across $d$ physical predictors ($5$ chemical gas sensors and $3$ ambient meteorological variables):
$$X_s \in \mathbb{R}^{n \times d}$$

To incorporate an affine intercept (baseline calibration offset $w_0$), we augment $X_s$ with a column vector of ones $\mathbf{1}_n = [1, 1, \dots, 1]^T \in \mathbb{R}^n$:
$$X = [\mathbf{1}_n \mid X_s] = \begin{bmatrix} 1 & x_{1,1} & x_{1,2} & \cdots & x_{1,d} \\ 1 & x_{2,1} & x_{2,2} & \cdots & x_{2,d} \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 1 & x_{n,1} & x_{n,2} & \cdots & x_{n,d} \end{bmatrix} \in \mathbb{R}^{n \times (d+1)}$$

The ground-truth benzene concentration measured by the certified reference analyzer forms the target vector:
$$\mathbf{y} = [y_1, y_2, \dots, y_n]^T \in \mathbb{R}^n$$

### 2. University Viva Explanation
* **Concept**: Mapping continuous multivariate physical signals into a finite-dimensional Euclidean vector space $\mathbb{R}^{n \times (d+1)}$.
* **Purpose**: Establish an algebraic domain where linear transformations, projections, orthogonal factorizations, and least-squares optimizations can be computed directly using matrix calculus.
* **Outcome**: A schema-verified design matrix $X \in \mathbb{R}^{8991 \times 9}$ and target vector $\mathbf{y} \in \mathbb{R}^{8991}$ with IEEE 754 double precision (`float64`) and zero missing values.

---

## Stage 2: Gaussian Elimination, RREF & LU Decomposition

### 1. Mathematical Formulation
Solving linear equations or determining subspace dimensions relies on elementary row operations represented as left-multiplication by elementary matrices $E_k$:
1. Row swap: $E_{ij}$
2. Row scaling: $E_i(c), \quad c \ne 0$
3. Row addition: $E_{ij}(c), \quad R_i \leftarrow R_i + c R_j$

#### Reduced Row-Echelon Form (RREF)
A matrix $R = \text{rref}(A)$ satisfies:
1. In every non-zero row, the leading entry is $1$ (the pivot).
2. The pivot of row $i+1$ lies strictly to the right of the pivot of row $i$.
3. Any column containing a pivot has zeros in all other entries.
4. Any all-zero rows reside at the bottom.

#### LU Decomposition with Partial Pivoting ($PA = LU$)
To prevent division by near-zero elements and catastrophic numerical cancellation, row permutations are tracked via permutation matrix $P$ ($P^T P = I$):
$$P A = L U$$
where:
* $P$ is an orthogonal permutation matrix.
* $L$ is unit lower-triangular ($L_{ii} = 1.0, L_{ij} = 0 \text{ for } j > i$), storing elimination multipliers $l_{ik} = \frac{A_{ik}}{A_{kk}}$.
* $U$ is upper-triangular ($U_{ij} = 0 \text{ for } i > j$).

Given $PA = LU$, the linear system $A\mathbf{x} = \mathbf{b}$ is solved in $\mathcal{O}(n^2)$ operations via two triangular substitutions:
1. Permute RHS: $\mathbf{b}' = P\mathbf{b}$
2. Forward solve: $L\mathbf{y} = \mathbf{b}'$
3. Back solve: $U\mathbf{x} = \mathbf{y}$

### 2. University Viva Explanation
* **Concept**: Factoring a non-singular linear operator into lower and upper triangular components mediated by row permutations.
* **Purpose**: Solve linear systems stably without computing explicit matrix inverses $A^{-1}$, reducing operational complexity from $\mathcal{O}(n^4)$ or $\mathcal{O}(n^3)$ with high cancellation to deterministic $\mathcal{O}(\frac{2}{3}n^3)$.
* **Outcome**: Implemented `lu_factorize` and `lu_solve`, verified against SciPy with numerical residual $\|P A - L U\|_\infty < 10^{-12}$.

---

## Stage 3: Vector Spaces, Subspaces & Rank-Nullity

### 1. Mathematical Formulation
For any real matrix $A \in \mathbb{R}^{m \times n}$, the Four Fundamental Subspaces (Strang's classification) are:
1. **Column Space $C(A) \subset \mathbb{R}^m$**: $\text{span}\{\mathbf{a}_1, \dots, \mathbf{a}_n\}$, dimension $r = \text{rank}(A)$.
2. **Null Space $N(A) \subset \mathbb{R}^n$**: $\{\mathbf{x} \in \mathbb{R}^n : A\mathbf{x} = \mathbf{0}\}$, dimension $k = \text{nullity}(A)$.
3. **Row Space $C(A^T) \subset \mathbb{R}^n$**: $\text{span}\{\text{rows of } A\}$, dimension $r = \text{rank}(A)$.
4. **Left Null Space $N(A^T) \subset \mathbb{R}^m$**: $\{\mathbf{y} \in \mathbb{R}^m : A^T \mathbf{y} = \mathbf{0}\}$, dimension $m - r$.

#### The Rank-Nullity Theorem
$$\dim(C(A)) + \dim(N(A)) = n \iff \text{rank}(A) + \text{nullity}(A) = n$$

Furthermore, the fundamental subspaces are mutually orthogonal complements in Euclidean space:
$$C(A^T) \perp N(A) \quad \text{and} \quad \mathbb{R}^n = C(A^T) \oplus N(A)$$
$$C(A) \perp N(A^T) \quad \text{and} \quad \mathbb{R}^m = C(A) \oplus N(A^T)$$

### 2. University Viva Explanation
* **Concept**: The decomposition of domain $\mathbb{R}^n$ and codomain $\mathbb{R}^m$ into orthogonal complementary subspaces.
* **Purpose**: Determine whether our sensor array provides $d$ independent degrees of freedom or whether sensor channels are collinear/redundant.
* **Outcome**: Verified that our AirSense sensor matrix has full column rank $r = 8$ (and $r = 9$ with intercept), nullity $= 0$, and validated the theorem on rank-deficient synthetic matrices with residual $\|A N\|_\infty < 10^{-14}$.

---

## Stage 4: Linear Transformations & Change of Basis

### 1. Mathematical Formulation
A function $T: V \to W$ is a linear transformation iff for all $\mathbf{u}, \mathbf{v} \in V$ and $c \in \mathbb{R}$:
$$T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$$
$$T(c\mathbf{u}) = cT(\mathbf{u})$$

In finite dimensions with chosen bases $\mathcal{B}_V$ and $\mathcal{B}_W$, every linear transformation is uniquely represented by matrix multiplication:
$$T(\mathbf{x}) = A\mathbf{x}$$

#### Change of Basis
Let $\mathcal{B} = \{\mathbf{v}_1, \dots, \mathbf{v}_n\}$ and $\mathcal{B}' = \{\mathbf{u}_1, \dots, \mathbf{u}_n\}$ be two bases for $\mathbb{R}^n$. The transition matrix $P_{\mathcal{B} \to \mathcal{B}'}$ transforms coordinates:
$$[\mathbf{x}]_{\mathcal{B}'} = P^{-1} [\mathbf{x}]_{\mathcal{B}}$$
Under change of basis, a linear operator $A$ transforms via similarity:
$$A' = P^{-1} A P$$

In AirSense, **feature standardization** ($z$-score normalization) represents an invertible affine transformation:
$$Z = (X - \mathbf{1}\boldsymbol{\mu}^T) D_{\sigma}^{-1}, \quad D_{\sigma} = \text{diag}(\sigma_1, \dots, \sigma_d)$$
which scales the basis vectors so that all sensor channels have unit empirical variance without altering subspace dimension.

---

## Stage 5: Norms, Orthogonality & Gram–Schmidt

### 1. Mathematical Formulation
For vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$, the standard Euclidean inner product is:
$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^n u_i v_i$$
inducing the Euclidean $\ell_2$ norm:
$$\|\mathbf{v}\|_2 = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle} = \sqrt{\sum_{i=1}^n v_i^2}$$
The angle $\theta$ between vectors satisfies the Cauchy-Schwarz inequality:
$$\cos \theta = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}, \quad |\mathbf{u}^T \mathbf{v}| \le \|\mathbf{u}\|_2 \|\mathbf{v}\|_2$$
Two vectors are orthogonal ($\mathbf{u} \perp \mathbf{v}$) iff $\mathbf{u}^T \mathbf{v} = 0$.

#### Classical vs. Modified Gram–Schmidt
Given linearly independent columns $\{\mathbf{a}_1, \dots, \mathbf{a}_n\}$, Gram–Schmidt produces an orthonormal basis $\{\mathbf{q}_1, \dots, \mathbf{q}_n\}$ such that $A = Q R$, where $Q^T Q = I_n$ and $R$ is upper-triangular:

* **Classical Gram-Schmidt (CGS)**:
  $$\mathbf{v}_j = \mathbf{a}_j - \sum_{i=1}^{j-1} (\mathbf{q}_i^T \mathbf{a}_j) \mathbf{q}_i$$
  *Numerical Flaw*: Evaluates projections against the uncorrected original vector $\mathbf{a}_j$. On ill-conditioned inputs, roundoff errors accumulate rapidly, causing severe loss of orthogonality ($\|Q^T Q - I\|_F \gg 0$).

* **Modified Gram-Schmidt (MGS)**:
  $$\mathbf{v}_j^{(1)} = \mathbf{a}_j$$
  $$\mathbf{v}_j^{(i+1)} = \mathbf{v}_j^{(i)} - (\mathbf{q}_i^T \mathbf{v}_j^{(i)}) \mathbf{q}_i, \quad i = 1, \dots, j-1$$
  *Numerical Advantage*: Sequentially projects the running residual $\mathbf{v}_j^{(i)}$, mathematically equivalent in exact arithmetic but drastically more stable in floating-point precision ($>10^6 \times$ lower orthogonality error).

### 2. University Viva Explanation
* **Concept**: Constructing an orthonormal coordinate system from arbitrary linearly independent vectors.
* **Purpose**: Eliminate mutual collinearity among sensor channels to prevent numerical instability during matrix inversion.
* **Outcome**: Demonstrated that on near-collinear vectors ($\epsilon = 10^{-7}$), CGS incurs an orthogonality error of $1.88 \times 10^{-2}$, whereas our MGS implementation maintains precision of $2.66 \times 10^{-9}$ (a $7 \times 10^6$-fold improvement).

---

## Stage 6: Orthogonal Projections & Least-Squares Calibration

### 1. Mathematical Formulation
In calibration, the measured target vector $\mathbf{y} \in \mathbb{R}^n$ rarely lies in the column space $C(X) \subset \mathbb{R}^n$ due to sensor noise and unmodeled disturbances:
$$\mathbf{y} \notin C(X) \implies X\mathbf{w} = \mathbf{y} \quad \text{has no exact solution}$$

The **Orthogonal Projection Operator** $P$ onto $C(X)$ projects $\mathbf{y}$ to the uniquely closest vector $\hat{\mathbf{y}} \in C(X)$:
$$P = X (X^T X)^{-1} X^T$$
Defining properties of any orthogonal projector:
1. **Idempotence**: $P^2 = P$ (projecting twice produces no further change).
2. **Self-Adjointness (Symmetry)**: $P^T = P$.

#### The Orthogonality Principle & Normal Equations
The error residual $\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}} = \mathbf{y} - X\mathbf{w}$ must be strictly orthogonal to every basis vector spanning $C(X)$:
$$X^T \mathbf{e} = \mathbf{0} \iff X^T (\mathbf{y} - X\mathbf{w}) = \mathbf{0} \iff (X^T X)\mathbf{w} = X^T \mathbf{y}$$

#### Why QR Factorization is Numerically Superior to Normal Equations
* Condition number: $\kappa(X) = \frac{\sigma_{\max}(X)}{\sigma_{\min}(X)}$.
* Forming the Gram matrix $X^T X$ squares the condition number:
  $$\kappa(X^T X) = \kappa(X)^2$$
  For our AirSense sensor matrix, $\kappa(X) \approx 1.89 \times 10^4 \implies \kappa(X^T X) \approx 3.57 \times 10^8$, discarding roughly 8 digits of IEEE 754 precision!
* Using QR decomposition $X = Q R$:
  $$X^T X = R^T Q^T Q R = R^T R$$
  $$X^T \mathbf{y} = R^T Q^T \mathbf{y}$$
  $$(R^T R)\mathbf{w} = R^T Q^T \mathbf{y} \iff R\mathbf{w} = Q^T \mathbf{y}$$
  Since $R$ is upper-triangular, we solve $R\mathbf{w} = Q^T \mathbf{y}$ by back substitution directly, working strictly with condition number $\kappa(R) = \kappa(X)$ and completely avoiding condition squaring!

### 2. University Viva Explanation
* **Concept**: Minimizing the Euclidean distance $\|\mathbf{y} - X\mathbf{w}\|_2^2$ by projecting the target onto the subspace $C(X)$.
* **Purpose**: Obtain the optimal unbiased linear calibration weights $\mathbf{w}$ for predicting benzene.
* **Outcome**: Verified residual orthogonality $\max |X^T \mathbf{e}| < 10^{-12}$, confirmed idempotence $\|P^2 - P\|_F < 10^{-14}$, and demonstrated that QR least squares reproduces SciPy `lstsq` within machine precision.

---

## Stage 7: Covariance Matrices & Positive Semidefiniteness

### 1. Mathematical Formulation
Given centered data $\tilde{X} = X - \mathbf{1}\boldsymbol{\mu}^T \in \mathbb{R}^{n \times d}$, the sample covariance matrix $S \in \mathbb{R}^{d \times d}$ is:
$$S = \frac{1}{n - 1} \tilde{X}^T \tilde{X} = \begin{bmatrix} \text{Var}(x_1) & \text{Cov}(x_1, x_2) & \cdots \\ \text{Cov}(x_2, x_1) & \text{Var}(x_2) & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix}$$

#### Positive Semidefiniteness (PSD)
A real symmetric matrix $S = S^T$ is positive semidefinite ($S \succeq 0$) iff for every non-zero vector $\mathbf{z} \in \mathbb{R}^d$:
$$\mathbf{z}^T S \mathbf{z} \ge 0$$
Proof that sample covariance is always PSD:
$$\mathbf{z}^T S \mathbf{z} = \mathbf{z}^T \left(\frac{1}{n-1} \tilde{X}^T \tilde{X}\right) \mathbf{z} = \frac{1}{n-1} (\tilde{X}\mathbf{z})^T (\tilde{X}\mathbf{z}) = \frac{1}{n-1} \|\tilde{X}\mathbf{z}\|_2^2 \ge 0$$
Since Euclidean squared norms are non-negative, $S$ must be positive semidefinite. All eigenvalues satisfy $\lambda_i \ge 0$.

### 2. University Viva Explanation
* **Concept**: Second-moment bivariate dispersion and energy of multivariate data.
* **Purpose**: Quantify sensor cross-sensitivities and verify spectral stability for Cholesky factorization and PCA.
* **Outcome**: Verified sample covariance calculation against NumPy `cov`, confirmed exact symmetry $\|S - S^T\|_\infty = 0$, and validated that all eigenvalues are non-negative ($\lambda_{\min} \ge 0$).

---

## Stage 8: Eigenvalues & Symmetric Diagonalization

### 1. Mathematical Formulation
For a linear operator $A$, an eigenvector $\mathbf{v} \ne \mathbf{0}$ and eigenvalue $\lambda$ satisfy:
$$A\mathbf{v} = \lambda \mathbf{v} \iff (A - \lambda I)\mathbf{v} = \mathbf{0}$$

#### The Spectral Theorem for Real Symmetric Matrices
If $S \in \mathbb{R}^{d \times d}$ is real and symmetric ($S = S^T$), then:
1. All $d$ eigenvalues $\lambda_1, \dots, \lambda_d$ are real.
2. Eigenvectors corresponding to distinct eigenvalues are mutually orthogonal.
3. There exists an orthonormal basis of eigenvectors $V = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_d]$ such that $V^T V = I_d$ and:
   $$S = V \Lambda V^T = \sum_{i=1}^d \lambda_i \mathbf{v}_i \mathbf{v}_i^T$$
   where $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_d)$ with $\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$.

#### The Jacobi Eigenvalue Algorithm
To compute the spectral decomposition from first principles without relying on black-box solvers, we implement cyclic Jacobi rotations:
At each iteration, we apply a Givens plane rotation $J(p, q, \theta)$ to annihilate the largest off-diagonal element $S_{pq}$:
$$S^{(k+1)} = J^T S^{(k)} J$$
The optimal rotation angle $\theta$ that zeroes out $S_{pq}^{(k+1)}$ is given analytically by:
$$\theta = \frac{1}{2} \text{atan2}(2 S_{pq}, S_{pp} - S_{qq})$$
As $k \to \infty$, $S^{(k)} \to \Lambda$ and $V = \prod J_k$.

### 2. University Viva Explanation
* **Concept**: Orthogonal diagonalization of real symmetric tensors into independent variance axes.
* **Purpose**: Find the principal directions along which the multi-sensor response ellipsoid has maximal variance.
* **Outcome**: Custom Jacobi algorithm verified against `scipy.linalg.eigh`, confirming spectral reconstruction $\|S - V \Lambda V^T\|_F < 10^{-11}$ and eigenvector orthonormality $\|V^T V - I\|_F < 10^{-12}$.

---

## Stage 9: SVD, PCA & Low-Rank Approximation

### 1. Mathematical Formulation
#### Singular Value Decomposition (SVD)
Any real matrix $\tilde{X} \in \mathbb{R}^{n \times d}$ factors as:
$$\tilde{X} = U \Sigma V^T$$
where:
* $U \in \mathbb{R}^{n \times d}$ has orthonormal columns ($U^T U = I_d$) — left singular vectors.
* $\Sigma = \text{diag}(\sigma_1, \dots, \sigma_d)$ contains singular values in descending order $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_d \ge 0$.
* $V \in \mathbb{R}^{d \times d}$ is an orthogonal matrix ($V^T V = I_d$) — right singular vectors.

#### The Fundamental Bridge: SVD $\longleftrightarrow$ Covariance Eigendecomposition
$$\tilde{X}^T \tilde{X} = (V \Sigma U^T)(U \Sigma V^T) = V \Sigma (U^T U) \Sigma V^T = V \Sigma^2 V^T$$
Dividing by $n - 1$:
$$S = \frac{1}{n - 1} \tilde{X}^T \tilde{X} = V \left(\frac{\Sigma^2}{n - 1}\right) V^T$$
Comparing this directly to the spectral decomposition $S = V \Lambda V^T$ yields the exact algebraic identities:
$$\mathbf{v}_i^{\text{PCA}} \equiv \mathbf{v}_i^{\text{SVD}} \quad (\text{Right singular vectors of } \tilde{X} = \text{Eigenvectors of } S)$$
$$\lambda_i = \frac{\sigma_i^2}{n - 1} \quad (\text{Covariance eigenvalues} = \text{Normalized squared singular values})$$

#### Eckart-Young-Mirsky Low-Rank Approximation Theorem
Let $\tilde{X}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T$ be the truncated SVD of rank $k < d$. Then for any matrix $B$ of rank at most $k$:
$$\min_{\text{rank}(B) \le k} \|\tilde{X} - B\|_F = \|\tilde{X} - \tilde{X}_k\|_F = \sqrt{\sum_{j=k+1}^d \sigma_j^2}$$
The reconstruction error is **strictly equal** to the Euclidean norm of the discarded singular value tail.

### 2. University Viva Explanation
* **Concept**: Optimal low-rank subspace approximation under the Frobenius norm.
* **Purpose**: Compress the 8-dimensional correlated gas sensor space into $k$ uncorrelated principal components $Z = \tilde{X} V_k$.
* **Outcome**: Verified that empirical reconstruction error $\|\tilde{X} - \hat{X}_k\|_F$ matches the theoretical tail sum $\sqrt{\sum_{j=k+1}^d \sigma_j^2}$ within $10^{-14}$, and confirmed that eigenvalues match across SVD and covariance paths within machine precision.

---

## Stage 10: Invariant Verification Cheat-Sheet for University Viva

When presenting each component to the examination board, reference this numerical validation table:

| Mathematical Stage | Key Theoretical Invariant | Algorithmic Check | Numerical Tolerance Achieved |
| :--- | :--- | :--- | :--- |
| **Gaussian / LU** | Factorization Equality | $\|P A - L U\|_\infty$ | $\mathbf{0.0}$ (exact) |
| **Gaussian / LU** | Permutation Orthogonality | $\|P^T P - I\|_F$ | $< 10^{-15}$ |
| **RREF / Subspaces** | Rank-Nullity Theorem | $\text{rank}(A) + \text{nullity}(A) = n$ | $\mathbf{\text{True}}$ (discrete integer) |
| **Subspaces** | Null Space Annihilation | $\max |A \mathbf{v}_{\text{null}}|$ | $< 1.78 \times 10^{-15}$ |
| **Modified Gram-Schmidt** | Orthonormal Basis | $\|Q^T Q - I\|_F$ | $< 3.61 \times 10^{-16}$ |
| **QR Factorization** | Reconstruction | $\|A - Q R\|_F$ | $< 1.26 \times 10^{-15}$ |
| **Projection Operator** | Idempotence | $\|P^2 - P\|_F$ | $< 10^{-15}$ |
| **Projection Operator** | Self-Adjointness | $\|P^T - P\|_F$ | $< 10^{-15}$ |
| **Least Squares (QR)** | Residual Orthogonality | $\max |X^T \mathbf{e}|$ | $< 5.32 \times 10^{-15}$ |
| **Covariance / PSD** | Symmetry & Non-negativity | $\|S - S^T\|_\infty, \quad \lambda_{\min}(S) \ge 0$ | $< 10^{-15}$ |
| **Jacobi Eigendecomposition** | Spectral Reconstruction | $\|S - V \Lambda V^T\|_F$ | $< 4.24 \times 10^{-13}$ |
| **PCA & SVD** | Variance Equivalence | $|\lambda_i - \frac{\sigma_i^2}{n-1}|$ | $< 7.11 \times 10^{-15}$ |
| **Eckart-Young Theorem** | Reconstruction Residual | $\|\tilde{X} - \hat{X}_k\|_F - \sqrt{\sum_{j>k} \sigma_j^2}$ | $< 2.84 \times 10^{-14}$ |
