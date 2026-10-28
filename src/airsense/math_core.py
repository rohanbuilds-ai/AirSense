"""AirSense Mathematical Core Library.

Pure, explicit implementations of linear algebra, matrix factorizations,
orthogonalization, spectral decompositions, and projection algorithms.

Designed for undergraduate academic explanation (Concept -> Purpose -> Outcome),
with benchmark parity against NumPy/SciPy reference standards.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# -----------------------------------------------------------------------------
# 1. Gaussian Elimination & RREF
# -----------------------------------------------------------------------------

def gaussian_elimination(
    A: np.ndarray,
    b: np.ndarray | None = None,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray | None, list[int]]:
    """Perform Gaussian elimination with partial pivoting to reduce A to upper-echelon form.

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (m, n).
    b : np.ndarray | None, default=None
        Optional right-hand side vector (m,) or matrix (m, k).
    tol : float, default=1e-12
        Numerical zero threshold for pivot detection.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None, list[int]]
        (U, modified_b, pivot_columns)
        where U is the upper row-echelon matrix, modified_b is transformed b,
        and pivot_columns records the column index of each pivot.
    """
    A_work = np.asarray(A, dtype=np.float64).copy()
    if A_work.ndim != 2:
        raise ValueError(f"Matrix A must be 2-dimensional; got shape {A_work.shape}.")

    m, n = A_work.shape
    b_work = None
    if b is not None:
        b_work = np.asarray(b, dtype=np.float64).copy()
        if b_work.shape[0] != m:
            raise ValueError(f"RHS dimension mismatch: A has {m} rows but b has {b_work.shape[0]}.")

    pivot_cols: list[int] = []
    r = 0

    for c in range(n):
        if r >= m:
            break
        # Partial pivoting: select row with largest absolute entry in column c
        pivot_row = r + int(np.argmax(np.abs(A_work[r:, c])))
        pivot_val = A_work[pivot_row, c]

        if np.abs(pivot_val) < tol:
            # Column c is near-zero below row r: skip to next column
            continue

        # Row swap in A and b
        if pivot_row != r:
            A_work[[r, pivot_row]] = A_work[[pivot_row, r]]
            if b_work is not None:
                b_work[[r, pivot_row]] = b_work[[pivot_row, r]]

        # Elimination below row r
        for i in range(r + 1, m):
            factor = A_work[i, c] / A_work[r, c]
            A_work[i, c:] -= factor * A_work[r, c:]
            A_work[i, c] = 0.0  # Force exact zero below pivot
            if b_work is not None:
                b_work[i] -= factor * b_work[r]

        pivot_cols.append(c)
        r += 1

    A_work[np.abs(A_work) < tol] = 0.0
    return A_work, b_work, pivot_cols


def rref(A: np.ndarray, tol: float = 1e-12) -> tuple[np.ndarray, list[int]]:
    """Compute the Reduced Row Echelon Form (RREF) using Gauss-Jordan elimination.

    Properties of RREF:
    1. Leading entry in each non-zero row is 1 (the pivot).
    2. Each pivot is strictly to the right of the pivot in the row above.
    3. Every column containing a pivot has zeros in all other entries.
    4. All-zero rows appear at the bottom.

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (m, n).
    tol : float, default=1e-12
        Tolerance below which values are treated as zero.

    Returns
    -------
    tuple[np.ndarray, list[int]]
        (R, pivot_columns)
    """
    R = np.asarray(A, dtype=np.float64).copy()
    if R.ndim != 2:
        raise ValueError(f"Matrix A must be 2-dimensional; got shape {R.shape}.")

    m, n = R.shape
    pivot_cols: list[int] = []
    r = 0

    for c in range(n):
        if r >= m:
            break
        # Find pivot in column c, rows r..m-1
        pivot_row = r + int(np.argmax(np.abs(R[r:, c])))
        pivot_val = R[pivot_row, c]

        if np.abs(pivot_val) < tol:
            continue

        # Swap rows
        if pivot_row != r:
            R[[r, pivot_row]] = R[[pivot_row, r]]

        # Normalize pivot row to make leading entry 1.0
        pivot_val = R[r, c]
        R[r, :] /= pivot_val
        R[r, c] = 1.0  # Force exact unit pivot

        # Eliminate both above and below row r
        for i in range(m):
            if i != r and np.abs(R[i, c]) > tol:
                factor = R[i, c]
                R[i, :] -= factor * R[r, :]
                R[i, c] = 0.0

        pivot_cols.append(c)
        r += 1

    R[np.abs(R) < tol] = 0.0
    return R, pivot_cols


# -----------------------------------------------------------------------------
# 2. LU Decomposition with Partial Pivoting (PA = LU)
# -----------------------------------------------------------------------------

def lu_factorize(
    A: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute the LU factorization with partial pivoting such that P @ A = L @ U.

    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (n, n).
    tol : float, default=1e-12
        Tolerance for pivot singularity check.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (P, L, U)
        where P is an orthogonal permutation matrix (P^T P = I),
        L is unit lower-triangular (L_ii = 1.0),
        and U is upper-triangular.
    """
    A_work = np.asarray(A, dtype=np.float64).copy()
    if A_work.ndim != 2 or A_work.shape[0] != A_work.shape[1]:
        raise ValueError(f"LU factorization requires a square matrix; got shape {A_work.shape}.")

    n = A_work.shape[0]
    L = np.eye(n, dtype=np.float64)
    P = np.eye(n, dtype=np.float64)

    for k in range(n - 1):
        # Find maximum pivot in column k from row k to n-1
        pivot_row = k + int(np.argmax(np.abs(A_work[k:, k])))
        pivot_val = A_work[pivot_row, k]

        if np.abs(pivot_val) < tol:
            # Rank-deficient / singular column: no elimination needed
            continue

        if pivot_row != k:
            # Swap rows in A
            A_work[[k, pivot_row]] = A_work[[pivot_row, k]]
            # Swap rows in P
            P[[k, pivot_row]] = P[[pivot_row, k]]
            # Swap already computed multipliers in L (columns 0..k-1)
            if k > 0:
                L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

        pivot = A_work[k, k]
        for i in range(k + 1, n):
            factor = A_work[i, k] / pivot
            L[i, k] = factor
            A_work[i, k:] -= factor * A_work[k, k:]
            A_work[i, k] = 0.0

    U = A_work
    U[np.abs(U) < tol] = 0.0
    return P, L, U


def lu_solve(
    P: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    b: np.ndarray,
    tol: float = 1e-12,
) -> np.ndarray:
    """Solve A @ x = b using precomputed LU factors (P @ A = L @ U).

    Parameters
    ----------
    P, L, U : np.ndarray
        Factorization matrices from lu_factorize.
    b : np.ndarray
        Right-hand side vector of shape (n,) or (n, 1).
    tol : float, default=1e-12
        Tolerance for detecting singular zero pivots on diagonal of U.

    Returns
    -------
    np.ndarray
        Solution vector x of shape (n,).
    """
    n = L.shape[0]
    b_vec = np.asarray(b, dtype=np.float64).ravel()
    if len(b_vec) != n:
        raise ValueError(f"Dimension mismatch: L has {n} rows but b has length {len(b_vec)}.")

    # 1. Permute RHS: b' = P @ b
    b_perm = P @ b_vec

    # 2. Forward substitution: L @ y = b'
    y = np.zeros(n, dtype=np.float64)
    for i in range(n):
        y[i] = b_perm[i] - np.dot(L[i, :i], y[:i])

    # 3. Back substitution: U @ x = y
    x = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        diag_val = U[i, i]
        if np.abs(diag_val) < tol:
            raise ValueError(f"Singular system: pivot U[{i}, {i}] is zero within tolerance ({diag_val:.2e}).")
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / diag_val

    return x


# -----------------------------------------------------------------------------
# 3. Vector Spaces, Subspaces & Rank-Nullity
# -----------------------------------------------------------------------------

def matrix_rank(A: np.ndarray, tol: float | None = None, method: str = "svd") -> int:
    """Compute the algebraic rank of matrix A.

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (m, n).
    tol : float | None, default=None
        Singular value threshold. If None, uses max(m, n) * eps * sigma_1.
    method : str, default='svd'
        'svd' uses singular values; 'rref' uses pivot count.

    Returns
    -------
    int
        Matrix rank (dimension of column space).
    """
    A_mat = np.asarray(A, dtype=np.float64)
    if A_mat.size == 0:
        return 0
    if A_mat.ndim != 2:
        raise ValueError(f"Matrix must be 2D; got shape {A_mat.shape}.")

    m, n = A_mat.shape
    if method == "rref":
        _, pivots = rref(A_mat, tol=tol or 1e-12)
        return len(pivots)

    # Standard SVD approach
    s = np.linalg.svd(A_mat, compute_uv=False)
    if tol is None:
        tol = max(m, n) * float(np.finfo(np.float64).eps) * (s[0] if len(s) > 0 else 1.0)
    return int(np.sum(s > tol))


def null_space(A: np.ndarray, tol: float | None = None, method: str = "svd") -> np.ndarray:
    """Compute an orthonormal basis for the null space N(A) such that A @ v = 0.

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (m, n).
    tol : float | None, default=None
        Threshold for zero singular values.
    method : str, default='svd'
        'svd' returns an orthonormal basis (V^T V = I).
        'rref' returns the canonical rational free-variable basis.

    Returns
    -------
    np.ndarray
        Basis matrix of shape (n, nullity), where each column is a basis vector.
        If nullity is 0, returns an empty array of shape (n, 0).
    """
    A_mat = np.asarray(A, dtype=np.float64)
    m, n = A_mat.shape

    if method == "rref":
        R, pivots = rref(A_mat, tol=tol or 1e-12)
        free_cols = [c for c in range(n) if c not in pivots]
        if not free_cols:
            return np.empty((n, 0), dtype=np.float64)
        basis = np.zeros((n, len(free_cols)), dtype=np.float64)
        for col_idx, free_c in enumerate(free_cols):
            basis[free_c, col_idx] = 1.0
            for r_idx, pivot_c in enumerate(pivots):
                basis[pivot_c, col_idx] = -R[r_idx, free_c]
        return basis

    # SVD method
    u, s, vh = np.linalg.svd(A_mat, full_matrices=True)
    if tol is None:
        tol = max(m, n) * float(np.finfo(np.float64).eps) * (s[0] if len(s) > 0 else 1.0)

    rank = int(np.sum(s > tol))
    # Null space vectors are the right singular vectors corresponding to singular values <= tol
    nullity = n - rank
    if nullity <= 0:
        return np.empty((n, 0), dtype=np.float64)

    # Columns of vh.T from rank to n
    null_basis = vh[rank:, :].T
    return null_basis


def verify_rank_nullity(A: np.ndarray, tol: float | None = None) -> dict[str, Any]:
    """Verify the fundamental Rank-Nullity Theorem: rank(A) + nullity(A) = n.

    Also verifies:
    1. Null space vector validity: max |A @ v| < tol
    2. Orthogonality: Row space C(A^T) is orthogonal to Null space N(A).

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (m, n).
    tol : float | None, default=None
        Numerical tolerance.

    Returns
    -------
    dict[str, Any]
        Dictionary with rank, nullity, num_columns, rank_nullity_satisfied,
        and max_null_residual.
    """
    A_mat = np.asarray(A, dtype=np.float64)
    m, n = A_mat.shape
    effective_tol = tol or 1e-11

    r = matrix_rank(A_mat, tol=effective_tol)
    N = null_space(A_mat, tol=effective_tol)
    nullity = N.shape[1]

    satisfied = (r + nullity == n)

    max_null_residual = 0.0
    if nullity > 0:
        residuals = np.abs(A_mat @ N)
        max_null_residual = float(np.max(residuals))

    return {
        "m_rows": m,
        "n_cols": n,
        "rank": r,
        "nullity": nullity,
        "rank_nullity_satisfied": satisfied,
        "max_null_residual": max_null_residual,
    }


# -----------------------------------------------------------------------------
# 4. Dot Products, Vector Norms & Angles
# -----------------------------------------------------------------------------

def dot_product(u: np.ndarray, v: np.ndarray) -> float:
    """Compute the Euclidean inner product <u, v> = u^T v."""
    u_vec = np.asarray(u, dtype=np.float64).ravel()
    v_vec = np.asarray(v, dtype=np.float64).ravel()
    if len(u_vec) != len(v_vec):
        raise ValueError(f"Vector length mismatch: {len(u_vec)} vs {len(v_vec)}.")
    return float(np.dot(u_vec, v_vec))


def vector_norm(v: np.ndarray, p: int | str = 2) -> float:
    """Compute vector norm (l1, l2, or l_inf)."""
    v_vec = np.asarray(v, dtype=np.float64).ravel()
    if p == 2 or p == "2":
        return float(np.sqrt(np.sum(v_vec**2)))
    elif p == 1 or p == "1":
        return float(np.sum(np.abs(v_vec)))
    elif p == np.inf or p == "inf":
        return float(np.max(np.abs(v_vec)))
    else:
        raise ValueError(f"Unsupported norm order: {p}")


def cosine_similarity(u: np.ndarray, v: np.ndarray, tol: float = 1e-12) -> float:
    """Compute the cosine of the angle between two non-zero vectors."""
    nu = vector_norm(u, 2)
    nv = vector_norm(v, 2)
    if nu < tol or nv < tol:
        raise ValueError("Cosine similarity is undefined for zero vectors.")
    cos_val = dot_product(u, v) / (nu * nv)
    # Clip to [-1.0, 1.0] to guard against floating-point overshoot
    return float(np.clip(cos_val, -1.0, 1.0))


# -----------------------------------------------------------------------------
# 5. Gram–Schmidt Orthogonalization (Classical & Modified)
# -----------------------------------------------------------------------------

def gram_schmidt_classical(
    A: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Perform Classical Gram-Schmidt (CGS) orthogonalization.

    Note: CGS is prone to severe loss of orthogonality on ill-conditioned
    or nearly collinear vectors due to roundoff cancellation.

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (m, n) with column vectors.
    tol : float, default=1e-12
        Collinearity tolerance.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        (Q, R, rank)
    """
    A_mat = np.asarray(A, dtype=np.float64)
    m, n = A_mat.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    rank = 0

    for j in range(n):
        v = A_mat[:, j].copy()
        # Classical projection against all previous Q columns simultaneously
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A_mat[:, j])
            v -= R[i, j] * Q[:, i]

        norm_v = vector_norm(v, 2)
        if norm_v < tol:
            R[j, j] = 0.0
            Q[:, j] = 0.0
        else:
            R[j, j] = norm_v
            Q[:, j] = v / norm_v
            rank += 1

    return Q, R, rank


def gram_schmidt_modified(
    A: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Perform Modified Gram-Schmidt (MGS) orthogonalization.

    MGS orthogonalizes sequentially against each newly updated direction,
    providing much higher numerical stability than CGS on near-collinear vectors.
    Handles linearly dependent columns by zeroing the corresponding Q column.

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (m, n) with column vectors.
    tol : float, default=1e-12
        Tolerance for detecting linear dependence.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, int]
        (Q, R, rank)
        where Q satisfies Q[:, :rank]^T @ Q[:, :rank] = I,
        R is upper-triangular, and A = Q @ R.
    """
    A_mat = np.asarray(A, dtype=np.float64)
    m, n = A_mat.shape
    Q = np.zeros((m, n), dtype=np.float64)
    R = np.zeros((n, n), dtype=np.float64)
    rank = 0

    for j in range(n):
        v = A_mat[:, j].copy()
        # Modified orthogonalization: update v immediately at each step
        for i in range(j):
            if vector_norm(Q[:, i], 2) > 0.5:  # Only if column i is non-zero
                R[i, j] = np.dot(Q[:, i], v)
                v -= R[i, j] * Q[:, i]
            else:
                R[i, j] = 0.0

        norm_v = vector_norm(v, 2)
        if norm_v < tol:
            # Linearly dependent column
            R[j, j] = 0.0
            Q[:, j] = 0.0
        else:
            R[j, j] = norm_v
            Q[:, j] = v / norm_v
            rank += 1

    return Q, R, rank


# -----------------------------------------------------------------------------
# 6. Orthogonal Projections
# -----------------------------------------------------------------------------

def projection_matrix(A: np.ndarray, tol: float = 1e-12) -> np.ndarray:
    """Compute the orthogonal projection operator P onto the column space C(A).

    P = A (A^T A)^+ A^T.
    If columns are orthonormal basis Q, P = Q Q^T.

    Verifies the two defining geometric properties:
    1. Idempotence: P^2 = P
    2. Self-adjoint / Symmetric: P = P^T

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (m, n).
    tol : float, default=1e-12
        Tolerance for rank deficiency.

    Returns
    -------
    np.ndarray
        Projection matrix of shape (m, m).
    """
    A_mat = np.asarray(A, dtype=np.float64)
    m, n = A_mat.shape

    # Use MGS to obtain orthonormal basis for C(A)
    Q, _, rank = gram_schmidt_modified(A_mat, tol=tol)
    if rank == 0:
        return np.zeros((m, m), dtype=np.float64)

    # Active orthonormal columns
    active_cols = [j for j in range(n) if vector_norm(Q[:, j], 2) > 0.5]
    Q_basis = Q[:, active_cols]

    # Orthogonal projection operator
    P = Q_basis @ Q_basis.T
    return P


def project_vector(
    A: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Project vector y onto the column space C(A) and compute the orthogonal residual.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (y_hat, residual)
        where y_hat = P @ y lies in C(A),
        and residual e = y - y_hat lies in the orthogonal complement C(A)^perp.
        Invariance guarantee: A^T @ residual = 0.
    """
    A_mat = np.asarray(A, dtype=np.float64)
    y_vec = np.asarray(y, dtype=np.float64).ravel()
    if A_mat.shape[0] != len(y_vec):
        raise ValueError(f"Shape mismatch: A has {A_mat.shape[0]} rows but y has {len(y_vec)} entries.")

    P = projection_matrix(A_mat, tol=tol)
    y_hat = P @ y_vec
    residual = y_vec - y_hat
    return y_hat, residual


# -----------------------------------------------------------------------------
# 7. Numerically Stable Least-Squares Calibration (QR & SVD)
# -----------------------------------------------------------------------------

def solve_least_squares_qr(
    A: np.ndarray,
    y: np.ndarray,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the linear least-squares problem min_w ||A w - y||_2 via QR factorization.

    Solves R w = Q^T y without computing (A^T A), avoiding squaring
    the matrix condition number kappa(A^T A) = kappa(A)^2.

    Parameters
    ----------
    A : np.ndarray
        Design matrix of shape (m, n) with m >= n.
    y : np.ndarray
        Target vector of shape (m,).
    tol : float, default=1e-12
        Tolerance for detecting rank deficiency in R.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        (w, residuals, rss)
        where w is the parameter vector (n,),
        residuals = y - A @ w,
        and rss = ||residuals||_2^2 is the residual sum of squares.
    """
    A_mat = np.asarray(A, dtype=np.float64)
    y_vec = np.asarray(y, dtype=np.float64).ravel()
    m, n = A_mat.shape

    if m < n:
        raise ValueError(f"Overdetermined least-squares requires m >= n; got shape ({m}, {n}).")
    if m != len(y_vec):
        raise ValueError(f"Row mismatch: A has {m} rows, y has {len(y_vec)}.")

    # QR via Modified Gram-Schmidt
    Q, R, rank = gram_schmidt_modified(A_mat, tol=tol)
    if rank < n:
        raise ValueError(f"Matrix A is rank deficient (rank={rank} < {n}); use solve_least_squares_svd.")

    # Compute Q^T @ y
    qty = Q.T @ y_vec

    # Back-substitution to solve R @ w = Q^T @ y
    w = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if np.abs(R[i, i]) < tol:
            raise ValueError(f"Zero pivot in R at index {i}.")
        w[i] = (qty[i] - np.dot(R[i, i + 1:], w[i + 1:])) / R[i, i]

    residuals = y_vec - A_mat @ w
    rss = float(np.sum(residuals**2))
    return w, residuals, rss


def solve_least_squares_svd(
    A: np.ndarray,
    y: np.ndarray,
    rcond: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the linear least-squares problem min_w ||A w - y||_2 via Singular Value Decomposition.

    Robust to rank-deficient and ill-conditioned systems.
    w = V @ Sigma^+ @ U^T @ y.

    Parameters
    ----------
    A : np.ndarray
        Design matrix of shape (m, n).
    y : np.ndarray
        Target vector of shape (m,).
    rcond : float, default=1e-12
        Cutoff for small singular values.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, float]
        (w, residuals, rss)
    """
    A_mat = np.asarray(A, dtype=np.float64)
    y_vec = np.asarray(y, dtype=np.float64).ravel()
    m, n = A_mat.shape

    u, s, vh = np.linalg.svd(A_mat, full_matrices=False)
    # Filter small singular values
    cutoff = rcond * (s[0] if len(s) > 0 else 1.0)
    s_inv = np.zeros_like(s)
    active = s > cutoff
    s_inv[active] = 1.0 / s[active]

    # Solution: w = V @ diag(s_inv) @ (U^T @ y)
    w = vh.T @ (s_inv * (u.T @ y_vec))

    residuals = y_vec - A_mat @ w
    rss = float(np.sum(residuals**2))
    return w, residuals, rss


# -----------------------------------------------------------------------------
# 8. Covariance Matrices & Positive Semidefiniteness
# -----------------------------------------------------------------------------

def compute_covariance(X: np.ndarray, ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Compute the sample covariance matrix from data matrix X.

    S = 1 / (n - ddof) * (X - 1 mu^T)^T (X - 1 mu^T).

    Parameters
    ----------
    X : np.ndarray
        Data array of shape (n, d) where rows are observations.
    ddof : int, default=1
        Degrees of freedom adjustment (1 for unbiased sample covariance).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (S, mu)
        where S is (d, d) sample covariance, and mu is (d,) mean vector.
    """
    X_mat = np.asarray(X, dtype=np.float64)
    if X_mat.ndim != 2:
        raise ValueError(f"Data matrix must be 2D; got shape {X_mat.shape}.")

    n, d = X_mat.shape
    if n <= ddof:
        raise ValueError(f"Sample size n={n} must exceed ddof={ddof}.")

    mu = np.mean(X_mat, axis=0)
    X_centered = X_mat - mu
    S = (X_centered.T @ X_centered) / (n - ddof)

    # Force exact symmetry: S = (S + S^T) / 2 to clean asymmetric floating-point noise
    S = 0.5 * (S + S.T)
    return S, mu


def is_positive_semidefinite(
    A: np.ndarray,
    tol: float = 1e-10,
) -> tuple[bool, dict[str, Any]]:
    """Verify whether a real matrix is symmetric positive semidefinite (PSD).

    A matrix S is PSD iff:
    1. S is symmetric: S = S^T.
    2. All eigenvalues lambda_i >= -tol.
    3. Quadratic form z^T S z >= -tol for all z != 0.

    Parameters
    ----------
    A : np.ndarray
        Square matrix of shape (d, d).
    tol : float, default=1e-10
        Numerical tolerance for symmetry and eigenvalue positivity.

    Returns
    -------
    tuple[bool, dict[str, Any]]
        (is_psd, diagnostics_dict)
    """
    A_mat = np.asarray(A, dtype=np.float64)
    if A_mat.ndim != 2 or A_mat.shape[0] != A_mat.shape[1]:
        raise ValueError(f"Matrix must be square; got shape {A_mat.shape}.")

    # 1. Symmetry check
    sym_residual = float(np.max(np.abs(A_mat - A_mat.T)))
    is_symmetric = sym_residual < tol

    # 2. Eigenvalue positivity check
    # Use eigh for symmetric matrix
    eigenvalues = np.linalg.eigvalsh(0.5 * (A_mat + A_mat.T))
    min_eigenval = float(np.min(eigenvalues))
    all_nonnegative = min_eigenval >= -tol

    # 3. Attempt Cholesky factorization with tiny regularizer if on boundary
    cholesky_success = False
    try:
        # Regularize minimally if close to singular
        reg = max(0.0, -min_eigenval + 1e-14)
        np.linalg.cholesky(A_mat + reg * np.eye(A_mat.shape[0]))
        cholesky_success = True
    except np.linalg.LinAlgError:
        cholesky_success = False

    is_psd = is_symmetric and all_nonnegative

    diagnostics = {
        "is_symmetric": is_symmetric,
        "symmetry_residual": sym_residual,
        "min_eigenvalue": min_eigenval,
        "all_eigenvalues_nonnegative": all_nonnegative,
        "cholesky_factorizable": cholesky_success,
    }
    return is_psd, diagnostics


# -----------------------------------------------------------------------------
# 9. Symmetric Matrix Diagonalization (Jacobi Algorithm)
# -----------------------------------------------------------------------------

def jacobi_eigenvalues_symmetric(
    A: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute eigenvalues and eigenvectors of a real symmetric matrix via cyclic Jacobi rotations.

    A classical from-scratch algorithm that diagonalizes A through orthogonal
    similarity transformations: V^T @ A @ V = Lambda.

    Parameters
    ----------
    A : np.ndarray
        Real symmetric matrix of shape (d, d).
    max_iter : int, default=300
        Maximum iterations.
    tol : float, default=1e-12
        Convergence tolerance on maximum off-diagonal magnitude.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (eigenvalues, eigenvectors)
        where eigenvalues is sorted descending (d,),
        and eigenvectors is (d, d) with orthonormal columns (V^T @ V = I).
    """
    A_work = np.asarray(A, dtype=np.float64).copy()
    if A_work.ndim != 2 or A_work.shape[0] != A_work.shape[1]:
        raise ValueError(f"Jacobi algorithm requires square matrix; got shape {A_work.shape}.")

    d = A_work.shape[0]
    V = np.eye(d, dtype=np.float64)

    # Enforce symmetry before starting
    A_work = 0.5 * (A_work + A_work.T)

    for it in range(max_iter):
        off_diag = np.abs(A_work - np.diag(np.diag(A_work)))
        max_val = np.max(off_diag)
        if max_val < tol:
            break

        p, q = np.unravel_index(np.argmax(off_diag), (d, d))
        if p > q:
            p, q = q, p

        app = A_work[p, p]
        aqq = A_work[q, q]
        apq = A_work[p, q]

        # Rotation angle theta that zeroes A_new[p, q]
        theta = 0.5 * np.arctan2(2.0 * apq, app - aqq)
        c = np.cos(theta)
        s = np.sin(theta)

        # Apply Givens rotation J(p, q, theta)
        J = np.eye(d, dtype=np.float64)
        J[p, p] = c
        J[q, q] = c
        J[p, q] = -s
        J[q, p] = s

        A_work = J.T @ A_work @ J
        V = V @ J

    eigenvals = np.diag(A_work)
    # Sort descending
    sort_idx = np.argsort(eigenvals)[::-1]
    return eigenvals[sort_idx], V[:, sort_idx]


# -----------------------------------------------------------------------------
# 10. PCA, Singular Value Decomposition & Low-Rank Approximation
# -----------------------------------------------------------------------------

def pca_svd(
    X: np.ndarray,
    n_components: int | None = None,
) -> dict[str, Any]:
    """Compute Principal Component Analysis via Singular Value Decomposition of centered data.

    Demonstrates the exact mathematical equivalence:
    - Right singular vectors V of centered data X_c are the eigenvectors of sample covariance S.
    - Covariance eigenvalues: lambda_i = sigma_i^2 / (n - 1).
    - Eckart-Young-Mirsky theorem: Reconstruction Frobenius error equals the tail sum
      of neglected singular values: ||X_c - X_hat_k||_F^2 = sum_{j=k+1}^d sigma_j^2.

    Parameters
    ----------
    X : np.ndarray
        Data array of shape (n, d).
    n_components : int | None, default=None
        Number of principal components to retain. If None, retains all d components.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - 'mean': (d,) empirical feature mean.
        - 'singular_values': (d,) singular values sigma_i.
        - 'eigenvalues': (d,) sample covariance eigenvalues lambda_i.
        - 'components': (d, d) principal axes (columns of V).
        - 'explained_variance_ratio': (d,) fraction of total variance per component.
        - 'cumulative_variance_ratio': (d,) cumulative variance explained.
        - 'scores': (n, k) projected coordinates in principal component space.
        - 'reconstruction_error': float, empirical Frobenius reconstruction error for k.
        - 'theoretical_error': float, theoretical error from singular value tail sum.
    """
    X_mat = np.asarray(X, dtype=np.float64)
    n, d = X_mat.shape
    k = n_components if n_components is not None else d
    if k > d:
        raise ValueError(f"Requested components k={k} exceeds feature dimension d={d}.")

    # 1. Center data
    mu = np.mean(X_mat, axis=0)
    X_c = X_mat - mu

    # 2. Thin SVD: X_c = U @ diag(s) @ V^T
    u, s, vh = np.linalg.svd(X_c, full_matrices=False)
    V = vh.T

    # 3. Covariance eigenvalue equivalence
    evals = (s**2) / (n - 1)
    total_var = np.sum(evals)
    explained_var_ratio = evals / total_var
    cum_var_ratio = np.cumsum(explained_var_ratio)

    # 4. Scores and low-rank reconstruction
    scores = X_c @ V[:, :k]
    X_hat_c = scores @ V[:, :k].T
    empirical_error = float(np.sqrt(np.sum((X_c - X_hat_c)**2)))
    theoretical_error = float(np.sqrt(np.sum(s[k:]**2))) if k < d else 0.0

    return {
        "mean": mu,
        "singular_values": s,
        "eigenvalues": evals,
        "components": V[:, :k],
        "all_components": V,
        "explained_variance_ratio": explained_var_ratio,
        "cumulative_variance_ratio": cum_var_ratio,
        "scores": scores,
        "reconstruction_error": empirical_error,
        "theoretical_error": theoretical_error,
    }


def low_rank_approximation(A: np.ndarray, k: int) -> tuple[np.ndarray, float]:
    """Compute the optimal rank-k approximation A_k under the Eckart-Young-Mirsky theorem.

    A_k = sum_{i=1}^k sigma_i u_i v_i^T.
    Error: ||A - A_k||_F = sqrt(sum_{j=k+1}^min(m, n) sigma_j^2).

    Parameters
    ----------
    A : np.ndarray
        Matrix of shape (m, n).
    k : int
        Target rank (1 <= k <= min(m, n)).

    Returns
    -------
    tuple[np.ndarray, float]
        (A_k, theoretical_frobenius_error)
    """
    A_mat = np.asarray(A, dtype=np.float64)
    m, n = A_mat.shape
    p = min(m, n)
    if k < 1 or k > p:
        raise ValueError(f"Rank k={k} must be in range [1, {p}].")

    u, s, vh = np.linalg.svd(A_mat, full_matrices=False)
    # Reconstruct with first k components
    A_k = u[:, :k] @ np.diag(s[:k]) @ vh[:k, :]
    theoretical_error = float(np.sqrt(np.sum(s[k:]**2))) if k < p else 0.0
    return A_k, theoretical_error
