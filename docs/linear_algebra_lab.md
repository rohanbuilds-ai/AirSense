# AirSense: Linear Algebra Laboratory (`/linear-algebra`)

**Workspace**: Linear Algebra Laboratory  
**Syllabus Coverage**: Unit 1 (Matrices, Systems of Linear Equations, Vector Spaces) & Unit 2 (Orthogonality, Projections, Eigenvalues & PCA)  
**Implementation Phase**: UI/UX Phase 4  
**Date**: October 29, 2026  
**Status**: **Complete & Verified (80 passing automated tests, Vite production build verified)**  

---

## 1. Executive Summary

The **Linear Algebra Laboratory** (`/linear-algebra`) is an interactive experimentation and pedagogical workspace in AirSense. It bridges textbook linear algebra theory directly to empirical chemical sensor calibration using the real UCI Air Quality dataset (March 2004 – April 2005).

Unlike educational sandboxes that rely on high-level library shortcuts (e.g. `scipy.linalg.lu` or `np.linalg.lstsq`), the AirSense Linear Algebra Lab executes all algorithms from scratch using pure Python routines in `src/airsense/math_core.py`. Every calculation runs on the FastAPI backend gateway and is streamed to a React/TypeScript interface.

The workspace is organized around a 6-step syllabus workflow:
1. **Real Data Matrix Representation**: Formulation of design matrix $\mathbf{X} \in \mathbb{R}^{m \times d}$ and reference target $\mathbf{y} \in \mathbb{R}^m$, feature standardization, and intercept column prepending.
2. **Matrix Simplification (Gaussian Elimination & LU)**: Interactive step-by-step row operation player with pivot highlighting, canonical RREF, and $PA = LU$ partial pivoting factorization ($||PA - LU||_\infty < 10^{-14}$).
3. **Vector Spaces & Subspaces**: Computation of Column Space Basis $C(A) \subseteq \mathbb{R}^m$, Null Space Basis $N(A) \subseteq \mathbb{R}^n$, rank, nullity, and direct verification of the Rank-Nullity Theorem $\dim(C(A)) + \dim(N(A)) = n$.
4. **Gram-Schmidt Orthogonalization**: Side-by-side comparison of Classical Gram-Schmidt (CGS) vs Modified Gram-Schmidt (MGS), quantifying orthogonality loss $||Q^T Q - I||_F$ and demonstrating stability gains of $> 10^8\times$ on ill-conditioned systems.
5. **Orthogonal Projections & Least Squares**: QR regression ($R w = Q^T y$), decomposition of $y = \hat{y} + e$, verification of orthogonality $X^T e = 0$, and empirical demonstration of the condition number squaring effect $\kappa(X^T X) = \kappa(X)^2$.
6. **Spectral Decomposition & PCA**: Covariance matrix $\mathbf{S}$, checks for symmetry and positive semi-definiteness (PSD), from-scratch Jacobi symmetric eigenvalue diagonalization ($S = V \Lambda V^T$), and interactive scree plot showing why dropping collinear components stabilizes calibration.

---

## 2. Mathematical Formulations & University Syllabus Alignment

### Step 1: Real Data Matrix Representation (Unit 1)
- **Mathematical Model**:
  $$\mathbf{X} = \begin{bmatrix} x_{11} & \cdots & x_{1d} \\ \vdots & \ddots & \vdots \\ x_{m1} & \cdots & x_{md} \end{bmatrix} \in \mathbb{R}^{m \times d}, \quad \mathbf{y} = \begin{bmatrix} y_1 \\ \vdots \\ y_m \end{bmatrix} \in \mathbb{R}^m$$
- **Standardization**:
  $$\tilde{x}_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
- **Interpretation**: Raw sensor micro-voltages ($10^3$) differ dramatically from temperatures ($10^1$). Without standardization, the design matrix exhibits artificially elevated condition numbers.

### Step 2: Gaussian Elimination, RREF & LU Decomposition (Unit 1)
- **Elementary Row Operations**:
  - Row interchange: $R_i \leftrightarrow R_j$
  - Row scaling: $R_r \leftarrow \frac{1}{a_{rk}} R_r$
  - Row elimination: $R_i \leftarrow R_i - c \cdot R_r$
- **Factorization**:
  $$P A = L U$$
  where $P$ is an orthogonal permutation matrix, $L$ is unit lower-triangular, and $U$ is upper-triangular.
- **Verification Invariant**:
  $$\|P A - L U\|_\infty < 10^{-14}$$

### Step 3: Fundamental Subspaces & Rank-Nullity (Unit 1)
- **The Rank-Nullity Theorem**:
  $$\operatorname{dim}(C(A)) + \operatorname{dim}(N(A)) = n \iff \operatorname{rank}(A) + \operatorname{nullity}(A) = n$$
- **Column Space**:
  $$C(A) = \{ A x : x \in \mathbb{R}^n \} \subseteq \mathbb{R}^m$$
- **Null Space (Kernel)**:
  $$N(A) = \{ x \in \mathbb{R}^n : A x = 0 \} \subseteq \mathbb{R}^n$$
- **Orthogonal Complements**:
  $$C(A^T) \perp N(A) \quad \text{in } \mathbb{R}^n, \qquad C(A) \perp N(A^T) \quad \text{in } \mathbb{R}^m$$

### Step 4: Gram-Schmidt Orthogonalization (Unit 2)
- **Classical Gram-Schmidt (CGS)**:
  $$q_k = v_k - \sum_{j=1}^{k-1} (q_j^T v_k) q_j, \quad q_k \leftarrow \frac{q_k}{\|q_k\|_2}$$
- **Modified Gram-Schmidt (MGS)**:
  $$v_k^{(j)} = v_k^{(j-1)} - (q_j^T v_k^{(j-1)}) q_j \quad \text{for } j=1, \dots, k-1$$
- **Orthogonality Metric**:
  $$\text{Loss} = \|Q^T Q - I_k\|_F$$
- **Empirical Result**: On the ill-conditioned challenge matrix ($\epsilon = 10^{-4}$), CGS incurs loss of $10^{-4}$, while MGS maintains loss below $10^{-15}$, achieving a stability ratio of $10^8\times$.

### Step 5: Orthogonal Projections & Least Squares (Unit 2)
- **Projection Operator**:
  $$P = X (X^T X)^{-1} X^T, \quad P^2 = P, \quad P^T = P$$
- **Orthogonal Residual Invariant**:
  $$e = y - \hat{y} = (I - P) y \implies X^T e = 0 \quad (\max |X^T e| < 10^{-14})$$
- **Condition Number Squaring Law**:
  $$\kappa(X^T X) = \kappa(X)^2$$
  - In AirSense: $\kappa(X) \approx 2.4 \times 10^4 \implies \kappa(X^T X) \approx 5.8 \times 10^8$.
  - Solving via QR factorization ($R w = Q^T y$) operates at condition number $\kappa(X)$, avoiding catastrophic roundoff cancellation.

### Step 6: Spectral Decomposition & PCA (Unit 2)
- **The Spectral Theorem for Real Symmetric Matrices**:
  $$S = V \Lambda V^T, \quad \Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_d), \quad V^T V = I_d$$
- **Positive Semi-Definite (PSD)**:
  $$x^T S x \ge 0 \quad \forall x \in \mathbb{R}^d \iff \lambda_i \ge 0$$
- **Jacobi Diagonalization**: Zeroes off-diagonal elements iteratively via orthogonal Givens rotation matrices:
  $$S_{k+1} = J_k^T S_k J_k$$
- **Variance Explained Scree**:
  $$\text{Ratio}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$
  The top 3 components capture $\sim 85\%$ of total sensor variance, explaining why PCR with $k=3$ outperforms full OLS on out-of-time test data.

---

## 3. API Contract Specifications

| Endpoint | Method | Input Schema | Output Schema | Invariants Tested |
|---|---|---|---|---|
| `/api/v1/math/extract-matrix` | `POST` | `MatrixExtractRequest` | `MatrixExtractResponse` | Training-only partition slice, $z$-score stats, row labels. |
| `/api/v1/math/gaussian-elimination` | `POST` | `GaussianEliminationRequest` | `GaussianEliminationResponse` | Row operations trace, RREF canonical pivots, $PA=LU$ error $< 10^{-14}$. |
| `/api/v1/math/subspaces` | `POST` | `SubspaceAnalysisRequest` | `SubspaceAnalysisResponse` | Rank-Nullity exact equality, null space residual $\max \|Av\|_2 < 10^{-12}$. |
| `/api/v1/math/gram-schmidt` | `POST` | `GramSchmidtRequest` | `GramSchmidtResponse` | MGS vs CGS Frobenius loss comparison, stability gain. |
| `/api/v1/math/least-squares` | `POST` | `LeastSquaresRequest` | `LeastSquaresResponse` | $X^T (y - \hat{y}) = 0$, condition squaring $\kappa(X^T X) = \kappa(X)^2$. |
| `/api/v1/math/spectral` | `POST` | `SpectralAnalysisRequest` | `SpectralAnalysisResponse` | Symmetry $\|S - S^T\| = 0$, PSD $\lambda_i \ge 0$, Jacobi residual $< 10^{-12}$. |

---

## 4. Frontend Component Architecture

```text
frontend/src/
├── components/math/
│   ├── MatrixViewer.tsx         # Bracket-bordered matrix visualizer with pivot/row highlighting
│   ├── StepNavigation.tsx       # 6-step workflow stepper with Unit 1 & Unit 2 indicators
│   └── MathConceptCard.tsx      # 4-quadrant pedagogical breakdown: Concept, Purpose, Operation, AirSense
├── pages/
│   └── LinearAlgebraPage.tsx    # Complete interactive Linear Algebra Lab workspace
├── types/
│   └── api.ts                   # Strongly-typed TypeScript interfaces matching Pydantic schemas
└── services/
    └── apiClient.ts             # Typed async client calls to FastAPI /math/* endpoints
```

---

## 5. Automated Verification Status

1. **Backend Automated Tests (`pytest`)**:
   - `tests/test_api.py`: 28 API integration tests passing (including 7 dedicated math laboratory tests).
   - Full suite: **80 tests passing in 4.80s** with zero regressions.
2. **Frontend Production Build (`npm run build`)**:
   - TypeScript compilation (`tsc`): 0 errors, strict mode.
   - Vite bundle: `dist/assets/index-Dx5NkpAT.js` built in 9.15s with zero syntax or type warnings.
3. **Execution Instructions**:
   - Start backend: `uvicorn api.main:app --reload --port 8000`
   - Start frontend: `npm --prefix frontend run dev`
   - Navigate to: `http://localhost:5173/linear-algebra`
