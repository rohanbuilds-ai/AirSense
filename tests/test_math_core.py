"""Unit and integration tests for the AirSense Mathematical Core library.

Verifies theoretical invariants, numerical precision, edge cases,
and equivalence against NumPy and SciPy reference implementations.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import linalg as sla

from airsense.math_core import (
    compute_covariance,
    cosine_similarity,
    dot_product,
    gaussian_elimination,
    gram_schmidt_classical,
    gram_schmidt_modified,
    is_positive_semidefinite,
    jacobi_eigenvalues_symmetric,
    low_rank_approximation,
    lu_factorize,
    lu_solve,
    matrix_rank,
    null_space,
    pca_svd,
    project_vector,
    projection_matrix,
    rref,
    solve_least_squares_qr,
    solve_least_squares_svd,
    vector_norm,
    verify_rank_nullity,
)

PROCESSED_DATA_PATH = Path("data/processed/air_quality_supervised.csv")


# -----------------------------------------------------------------------------
# 1. Tests for Gaussian Elimination & RREF
# -----------------------------------------------------------------------------

def test_gaussian_elimination_deterministic() -> None:
    """Test Gaussian elimination on a deterministic 3x3 non-singular system."""
    A = np.array([
        [2.0, 1.0, -1.0],
        [-3.0, -1.0, 2.0],
        [-2.0, 1.0, 2.0],
    ])
    b = np.array([8.0, -11.0, -3.0])

    U, mod_b, pivots = gaussian_elimination(A, b)

    # Check upper-echelon structure
    assert pivots == [0, 1, 2]
    assert np.allclose(np.tril(U, -1), 0.0, atol=1e-12)

    # Back-solve to confirm solution
    x = np.zeros(3)
    for i in range(2, -1, -1):
        x[i] = (mod_b[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    expected_x = np.linalg.solve(A, b)
    assert np.allclose(x, expected_x, atol=1e-12)


def test_rref_canonical_free_variables() -> None:
    """Test RREF on a 3x4 matrix with rank 2 and 2 free variables."""
    A = np.array([
        [1.0, 2.0, 2.0, 2.0],
        [2.0, 4.0, 6.0, 8.0],
        [3.0, 6.0, 8.0, 10.0],
    ])

    R, pivots = rref(A)

    assert pivots == [0, 2]  # Columns 0 and 2 are pivot columns
    # In RREF, pivot rows must have leading 1s and zeros in the rest of pivot columns
    assert R[0, 0] == pytest.approx(1.0)
    assert R[0, 2] == pytest.approx(0.0)
    assert R[1, 2] == pytest.approx(1.0)
    assert R[1, 0] == pytest.approx(0.0)
    assert np.allclose(R[2, :], 0.0, atol=1e-12)  # Third row should be all zeros


# -----------------------------------------------------------------------------
# 2. Tests for LU Decomposition with Partial Pivoting (PA = LU)
# -----------------------------------------------------------------------------

def test_lu_factorize_identity_and_solve() -> None:
    """Verify factorization PA = LU and solution accuracy against SciPy."""
    A = np.array([
        [1.0, 2.0, 4.0],
        [3.0, 8.0, 14.0],
        [2.0, 6.0, 13.0],
    ])
    b = np.array([3.0, 1.0, 4.0])

    P, L, U = lu_factorize(A)

    # Invariant 1: Permutation matrix orthogonality P^T P = I
    assert np.allclose(P.T @ P, np.eye(3), atol=1e-12)

    # Invariant 2: L is unit lower-triangular (diag = 1, upper = 0)
    assert np.allclose(np.diag(L), 1.0, atol=1e-12)
    assert np.allclose(np.triu(L, 1), 0.0, atol=1e-12)

    # Invariant 3: U is upper-triangular
    assert np.allclose(np.tril(U, -1), 0.0, atol=1e-12)

    # Invariant 4: Factorization equality P @ A = L @ U
    assert np.allclose(P @ A, L @ U, atol=1e-12)

    # Solve test
    x = lu_solve(P, L, U, b)
    ref_x = np.linalg.solve(A, b)
    assert np.allclose(x, ref_x, atol=1e-12)
    assert np.linalg.norm(A @ x - b) < 1e-12


def test_lu_singular_system_raises() -> None:
    """Ensure lu_solve raises ValueError when matrix is singular."""
    A_singular = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [1.0, 1.0, 1.0],
    ])
    b = np.array([1.0, 2.0, 3.0])
    P, L, U = lu_factorize(A_singular)
    with pytest.raises(ValueError, match="Singular system"):
        lu_solve(P, L, U, b)


# -----------------------------------------------------------------------------
# 3. Tests for Vector Spaces, Subspaces & Rank-Nullity
# -----------------------------------------------------------------------------

def test_rank_nullity_theorem_verification() -> None:
    """Test Rank-Nullity theorem on rank-deficient matrix: rank + nullity = n."""
    A = np.array([
        [1.0, 3.0, 1.0, 4.0],
        [2.0, 4.0, 2.0, 6.0],
        [3.0, 7.0, 3.0, 10.0],
    ])  # 3 rows, 4 columns. Row 3 = Row 1 + Row 2. Column 3 = Column 1.

    diag = verify_rank_nullity(A)
    assert diag["m_rows"] == 3
    assert diag["n_cols"] == 4
    assert diag["rank"] == 2
    assert diag["nullity"] == 2
    assert diag["rank_nullity_satisfied"] is True
    assert diag["max_null_residual"] < 1e-12

    # Verify basis vectors in null space satisfy A @ v = 0
    N = null_space(A)
    assert N.shape == (4, 2)
    assert np.allclose(A @ N, 0.0, atol=1e-12)
    # Check that null space basis is orthonormal
    assert np.allclose(N.T @ N, np.eye(2), atol=1e-12)


def test_null_space_full_rank() -> None:
    """Full column rank matrix should have trivial null space (nullity = 0)."""
    A_full = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ])
    assert matrix_rank(A_full) == 2
    N = null_space(A_full)
    assert N.shape == (2, 0)


# -----------------------------------------------------------------------------
# 4. Tests for Gram–Schmidt Orthogonalization (CGS vs MGS)
# -----------------------------------------------------------------------------

def test_gram_schmidt_modified_orthogonality_and_reconstruction() -> None:
    """Test MGS factorizes A = Q @ R with Q^T Q = I and exact reconstruction."""
    A = np.array([
        [1.0, 2.0, 4.0],
        [0.0, 0.0, 5.0],
        [0.0, 3.0, 6.0],
        [2.0, 1.0, 0.0],
    ])

    Q, R, rank = gram_schmidt_modified(A)

    assert rank == 3
    # Orthogonality: Q^T Q = I
    assert np.allclose(Q.T @ Q, np.eye(3), atol=1e-12)
    # R is upper triangular
    assert np.allclose(np.tril(R, -1), 0.0, atol=1e-12)
    # Reconstruction: A = Q @ R
    assert np.allclose(A, Q @ R, atol=1e-12)


def test_gram_schmidt_handles_dependent_columns() -> None:
    """MGS must detect dependent columns, zeroing corresponding Q columns without crashing."""
    v1 = np.array([1.0, 0.0, 1.0])
    v2 = np.array([0.0, 1.0, 1.0])
    v3 = 2.0 * v1 + 3.0 * v2  # Linearly dependent column
    A = np.column_stack([v1, v2, v3])

    Q, R, rank = gram_schmidt_modified(A)

    assert rank == 2
    assert np.linalg.norm(Q[:, 2]) == pytest.approx(0.0)
    assert R[2, 2] == pytest.approx(0.0)
    # Reconstruction remains exact
    assert np.allclose(A, Q @ R, atol=1e-12)


def test_gram_schmidt_cgs_vs_mgs_stability() -> None:
    """Demonstrate numerical superiority of MGS over CGS on near-collinear vectors."""
    eps = 1e-7
    # Near-collinear columns
    A_ill = np.array([
        [1.0, 1.0, 1.0],
        [eps, 0.0, 0.0],
        [0.0, eps, 0.0],
        [0.0, 0.0, eps],
    ])

    Q_cgs, _, _ = gram_schmidt_classical(A_ill)
    Q_mgs, _, _ = gram_schmidt_modified(A_ill)

    # Check orthogonality error ||Q^T Q - I||_F
    err_cgs = np.linalg.norm(Q_cgs.T @ Q_cgs - np.eye(3))
    err_mgs = np.linalg.norm(Q_mgs.T @ Q_mgs - np.eye(3))

    # MGS preserves orthogonality significantly better than CGS
    assert err_mgs < 1e-8
    assert err_cgs > 1e-3
    assert err_cgs > 1e4 * err_mgs


# -----------------------------------------------------------------------------
# 5. Tests for Projections, Orthogonality & Residuals
# -----------------------------------------------------------------------------

def test_projection_matrix_idempotence_and_symmetry() -> None:
    """Verify projection matrix properties: P^2 = P and P = P^T."""
    A = np.array([
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
        [1.0, 2.0],
    ])

    P = projection_matrix(A)

    # 1. Idempotence: P @ P = P
    assert np.allclose(P @ P, P, atol=1e-12)
    # 2. Symmetry: P^T = P
    assert np.allclose(P.T, P, atol=1e-12)


def test_project_vector_residual_orthogonality() -> None:
    """Verify that projection residual e = y - y_hat is strictly orthogonal to C(A)."""
    A = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ])
    y = np.array([1.0, 2.0, 9.0])

    y_hat, residual = project_vector(A, y)

    # Invariant 1: y = y_hat + residual
    assert np.allclose(y, y_hat + residual, atol=1e-12)

    # Invariant 2: Orthogonality principle A^T @ residual = 0
    assert np.allclose(A.T @ residual, 0.0, atol=1e-12)

    # Invariant 3: Pythagorean theorem ||y||^2 = ||y_hat||^2 + ||residual||^2
    assert vector_norm(y, 2)**2 == pytest.approx(
        vector_norm(y_hat, 2)**2 + vector_norm(residual, 2)**2, abs=1e-11
    )


# -----------------------------------------------------------------------------
# 6. Tests for Least-Squares Calibration (QR & SVD)
# -----------------------------------------------------------------------------

def test_least_squares_qr_and_svd_match_scipy() -> None:
    """Verify QR and SVD least-squares solutions match SciPy lstsq reference."""
    np.random.seed(42)
    m, n = 25, 4
    X = np.random.randn(m, n)
    true_w = np.array([1.5, -2.0, 3.2, 0.5])
    y = X @ true_w + 0.05 * np.random.randn(m)

    w_qr, res_qr, rss_qr = solve_least_squares_qr(X, y)
    w_svd, res_svd, rss_svd = solve_least_squares_svd(X, y)
    ref_w, ref_rss, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Match within machine precision
    assert np.allclose(w_qr, ref_w, atol=1e-12)
    assert np.allclose(w_svd, ref_w, atol=1e-12)
    assert np.allclose(w_qr, w_svd, atol=1e-12)

    # Residual orthogonality: X^T @ (y - Xw) = 0
    assert np.allclose(X.T @ res_qr, 0.0, atol=1e-12)


# -----------------------------------------------------------------------------
# 7. Tests for Covariance & Positive Semidefiniteness
# -----------------------------------------------------------------------------

def test_covariance_and_psd_properties() -> None:
    """Test covariance calculation, exact symmetry, and positive semidefiniteness."""
    np.random.seed(99)
    X = np.random.randn(100, 5)

    S, mu = compute_covariance(X, ddof=1)
    ref_S = np.cov(X, rowvar=False, ddof=1)

    assert np.allclose(S, ref_S, atol=1e-12)
    assert np.allclose(mu, np.mean(X, axis=0), atol=1e-12)

    # Test PSD verification
    is_psd, diag = is_positive_semidefinite(S)
    assert is_psd is True
    assert diag["is_symmetric"] is True
    assert diag["all_eigenvalues_nonnegative"] is True
    assert diag["cholesky_factorizable"] is True


def test_positive_semidefinite_catches_negative_eigenvalue() -> None:
    """Ensure non-PSD matrix is correctly detected."""
    # Matrix with negative eigenvalue
    A_non_psd = np.array([
        [1.0, 2.0],
        [2.0, 1.0],
    ])  # eigenvalues: 3 and -1
    is_psd, diag = is_positive_semidefinite(A_non_psd)
    assert is_psd is False
    assert diag["all_eigenvalues_nonnegative"] is False
    assert diag["min_eigenvalue"] < 0.0


# -----------------------------------------------------------------------------
# 8. Tests for Symmetric Diagonalization (Jacobi Method)
# -----------------------------------------------------------------------------

def test_jacobi_eigenvalues_symmetric_spectral_theorem() -> None:
    """Verify spectral theorem S = V @ diag(lambda) @ V^T via Jacobi rotations."""
    S = np.array([
        [4.0, 1.0, 2.0],
        [1.0, 5.0, 3.0],
        [2.0, 3.0, 6.0],
    ])

    evals, V = jacobi_eigenvalues_symmetric(S)
    ref_evals, ref_V = np.linalg.eigh(S)
    ref_evals = ref_evals[::-1]

    # Eigenvalues match
    assert np.allclose(evals, ref_evals, atol=1e-11)

    # Invariant 1: Orthonormal eigenvectors V^T @ V = I
    assert np.allclose(V.T @ V, np.eye(3), atol=1e-12)

    # Invariant 2: Spectral reconstruction S = V @ Lambda @ V^T
    assert np.allclose(S, V @ np.diag(evals) @ V.T, atol=1e-11)


# -----------------------------------------------------------------------------
# 9. Tests for PCA via SVD & Low-Rank Approximation
# -----------------------------------------------------------------------------

def test_pca_svd_covariance_equivalence_and_eckart_young() -> None:
    """Verify exact equivalence lambda_i = sigma_i^2 / (n - 1) and Eckart-Young error."""
    np.random.seed(42)
    n, d = 80, 6
    X = np.random.randn(n, d)

    res = pca_svd(X, n_components=3)

    # Invariant 1: Covariance eigenvalues lambda_i == sigma_i^2 / (n - 1)
    expected_evals = (res["singular_values"]**2) / (n - 1)
    assert np.allclose(res["eigenvalues"], expected_evals, atol=1e-12)

    # Invariant 2: Variance ratios sum to 1.0
    assert np.sum(res["explained_variance_ratio"]) == pytest.approx(1.0)

    # Invariant 3: Eckart-Young reconstruction error matches theoretical tail sum
    assert res["reconstruction_error"] == pytest.approx(res["theoretical_error"], rel=1e-10)


def test_low_rank_approximation() -> None:
    """Verify Eckart-Young optimal low-rank approximation."""
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
    ])
    A_k, err = low_rank_approximation(A, k=2)

    # Rank of A_k must be 2
    assert matrix_rank(A_k) == 2
    # Empirical Frobenius error must match theoretical singular value tail
    empirical_err = float(np.linalg.norm(A - A_k, "fro"))
    assert empirical_err == pytest.approx(err, abs=1e-12)


# -----------------------------------------------------------------------------
# 10. Integration Test: Real AirSense Sensor Matrix
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not PROCESSED_DATA_PATH.is_file(), reason="Processed dataset not available")
def test_airsense_real_data_mathematical_pipeline() -> None:
    """End-to-end integration test executing the mathematical core on real AirSense features."""
    import pandas as pd
    df = pd.read_csv(PROCESSED_DATA_PATH)

    sensor_features = [
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "PT08.S3(NOx)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH",
    ]
    target_col = "C6H6(GT)"

    # Take a 200-sample contiguous slice
    sub_df = df.iloc[:200]
    X_raw = sub_df[sensor_features].to_numpy(dtype=np.float64)
    y = sub_df[target_col].to_numpy(dtype=np.float64)

    # 1. Rank & Condition Number
    r = matrix_rank(X_raw)
    assert r == 8  # Full column rank

    # 2. Add intercept column
    X_design = np.hstack([np.ones((len(X_raw), 1)), X_raw])
    assert matrix_rank(X_design) == 9

    # 3. Solve OLS via QR
    w_qr, residuals, rss = solve_least_squares_qr(X_design, y)
    assert len(w_qr) == 9
    # Residual orthogonality on real data
    assert np.allclose(X_design.T @ residuals, 0.0, atol=1e-8)

    # 4. Covariance & PSD on real sensor channels
    S, mu = compute_covariance(X_raw)
    is_psd, diag = is_positive_semidefinite(S)
    assert is_psd is True

    # 5. PCA on real sensor channels
    pca_res = pca_svd(X_raw, n_components=4)
    # First 4 principal components explain > 90% of total sensor variance
    assert pca_res["cumulative_variance_ratio"][3] > 0.85
    # Eckart-Young error matches theoretical singular value tail
    assert pca_res["reconstruction_error"] == pytest.approx(pca_res["theoretical_error"], rel=1e-8)


def test_column_space_properties():
    """Verify column space basis extraction via RREF pivot columns and SVD."""
    from airsense.math_core import column_space, matrix_rank

    A = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 4.0, 6.0, 8.0],
        [1.0, 0.0, 1.0, 2.0],
    ])
    r = matrix_rank(A)
    assert r == 2

    # Pivot method
    col_basis_pivot = column_space(A, method="pivot")
    assert col_basis_pivot.shape == (3, 2)
    # Verify rank of basis is 2
    assert matrix_rank(col_basis_pivot) == 2

    # SVD method
    col_basis_svd = column_space(A, method="svd")
    assert col_basis_svd.shape == (3, 2)
    assert np.allclose(col_basis_svd.T @ col_basis_svd, np.eye(2), atol=1e-12)

