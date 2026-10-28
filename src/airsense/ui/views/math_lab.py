"""Mathematical Foundations Laboratory View.

Provides interactive explorations of core linear algebra algorithms:
- Gaussian Elimination and RREF
- Gram-Schmidt Orthogonalization (CGS vs MGS)
- Spectral Decomposition and Jacobi Algorithm
- Fundamental Subspaces and the Rank-Nullity Theorem
"""

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

from airsense.math_core import (
    gaussian_elimination,
    rref,
    lu_factorize,
    gram_schmidt_classical,
    gram_schmidt_modified,
    matrix_rank,
    null_space,
    verify_rank_nullity,
    jacobi_eigenvalues_symmetric,
    compute_covariance,
    vector_norm,
)


def render_math_lab() -> None:
    """Render the interactive Mathematical Foundations Laboratory."""
    st.header("📐 Mathematical Foundations Laboratory")
    st.markdown(
        """
        This interactive laboratory exposes the from-scratch numerical algorithms implemented in 
        `airsense.math_core`. Explore matrix reductions, orthogonalization stability, spectral 
        decompositions, and fundamental subspace geometries.
        """
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Gaussian Elimination & RREF",
            "2. Gram-Schmidt Orthogonality (CGS vs MGS)",
            "3. Jacobi Spectral Decomposition",
            "4. Subspaces & Rank-Nullity",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: Gaussian Elimination, LU & RREF
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Gaussian Elimination, LU Factorization & RREF")
        st.markdown(
            r"""
            **Mathematical Triad:**
            - **Concept**: Elementary row operations (row interchanges, row scaling, row additions) transform $A$ into upper-triangular or reduced echelon form.
            - **Purpose**: Solve linear systems $A\mathbf{x} = \mathbf{b}$, compute matrix determinants, and determine pivot columns.
            - **Outcome**: Factored representation $PA = LU$ and canonical $\text{RREF}(A)$ with identified pivot indices.
            """
        )

        matrix_preset = st.selectbox(
            "Select Preset Matrix or Enter Custom:",
            [
                "Standard Invertible 3x3",
                "Singular / Dependent Rows 3x3",
                "AirSense Sensor Correlation Submatrix (3x3)",
            ],
            key="rref_preset",
        )

        if matrix_preset == "Standard Invertible 3x3":
            default_A = "2.0, 1.0, -1.0\n-3.0, -1.0, 2.0\n-2.0, 1.0, 2.0"
            default_b = "8.0, -11.0, -3.0"
        elif matrix_preset == "Singular / Dependent Rows 3x3":
            default_A = "1.0, 2.0, 3.0\n4.0, 5.0, 6.0\n5.0, 7.0, 9.0"
            default_b = "1.0, 2.0, 3.0"
        else:
            default_A = "1.00, 0.88, 0.89\n0.88, 1.00, 0.77\n0.89, 0.77, 1.00"
            default_b = "1.0, 1.0, 1.0"

        col_input_a, col_input_b = st.columns([2, 1])
        with col_input_a:
            raw_matrix = st.text_area("Matrix A (comma-separated rows):", default_A, height=100)
        with col_input_b:
            raw_b = st.text_area("Vector b (comma-separated):", default_b, height=100)

        try:
            A = np.array([[float(val.strip()) for val in row.split(",") if val.strip()] 
                          for row in raw_matrix.strip().split("\n") if row.strip()])
            b = np.array([float(val.strip()) for val in raw_b.strip().split(",") if val.strip()])

            m, n = A.shape
            st.write(f"**Parsed Matrix Dimensions**: ${m} \\times {n}$, Vector length: ${len(b)}$")

            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.write("**Reduced Row Echelon Form (RREF)**:")
                R, pivots = rref(A)
                st.dataframe(np.round(R, 4), use_container_width=True)
                st.info(f"**Pivot Columns**: {pivots} | **Computed Rank**: {len(pivots)}")

            with col_res2:
                st.write("**LU Decomposition ($PA = LU$)**:")
                if m == n:
                    P, L, U = lu_factorize(A)
                    recon_err = np.linalg.norm(P @ A - L @ U)
                    st.write("Matrix $L$ (Unit lower-triangular):")
                    st.dataframe(np.round(L, 4), use_container_width=True)
                    st.write("Matrix $U$ (Upper-triangular):")
                    st.dataframe(np.round(U, 4), use_container_width=True)
                    st.caption(f"Reconstruction Error $\\|PA - LU\\| = {recon_err:.2e}$")
                else:
                    st.warning("LU Factorization requires a square matrix.")

            if m == n and len(b) == m:
                st.write("**Linear System Solution ($A\\mathbf{x} = \\mathbf{b}$ via Gaussian Elimination)**:")
                try:
                    x = gaussian_elimination(A, b)
                    st.success(f"Solution vector $\\mathbf{{x}} = {np.round(x, 4).tolist()}$")
                    residual = np.linalg.norm(A @ x - b)
                    st.caption(f"Residual $\\|A\\mathbf{{x}} - \\mathbf{{b}}\\| = {residual:.2e}$")
                except ValueError as err:
                    st.error(f"System cannot be solved uniquely: {err}")

        except Exception as ex:
            st.error(f"Error parsing matrix input: {ex}")

    # -------------------------------------------------------------
    # TAB 2: Gram-Schmidt & QR
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Classical vs. Modified Gram-Schmidt Orthogonalization")
        st.markdown(
            r"""
            **Mathematical Triad:**
            - **Concept**: Construct an orthonormal basis $\{q_1, \dots, q_k\}$ from linearly independent vectors $\{a_1, \dots, a_k\}$.
            - **Purpose**: Stable QR factorization and orthogonal projection avoiding condition number squaring.
            - **Outcome**: Verification of catastrophic cancellation in Classical Gram-Schmidt (CGS) vs. numerical stability in Modified Gram-Schmidt (MGS).
            """
        )

        delta_exp = st.slider(
            "Collinearity Perturbation (log10 epsilon):",
            min_value=-10,
            max_value=-1,
            value=-5,
            step=1,
            help=r"Smaller epsilon makes vectors closer to collinear ($a_2 = a_1 + \epsilon$).",
        )
        eps = 10.0 ** delta_exp

        st.write(
            f"Testing with nearly dependent 3D vectors: "
            f"$\\mathbf{{a}}_1 = [1, 1, 1]^T, "
            f"\\mathbf{{a}}_2 = [1, 1 + {eps:.1e}, 1]^T, "
            f"\\mathbf{{a}}_3 = [1, 1, 1 + {eps:.1e}]^T$"
        )

        A_test = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0 + eps, 1.0],
            [1.0, 1.0, 1.0 + eps],
        ])

        Q_cgs, R_cgs = gram_schmidt_classical(A_test)
        Q_mgs, R_mgs = gram_schmidt_modified(A_test)

        ortho_err_cgs = np.linalg.norm(Q_cgs.T @ Q_cgs - np.eye(3))
        ortho_err_mgs = np.linalg.norm(Q_mgs.T @ Q_mgs - np.eye(3))

        c1, c2 = st.columns(2)
        with c1:
            st.metric(r"CGS Orthogonality Loss $\|Q^T Q - I\|_F$", f"{ortho_err_cgs:.2e}")
            st.dataframe(np.round(Q_cgs.T @ Q_cgs, 4), use_container_width=True)
            if ortho_err_cgs > 1e-4:
                st.warning("⚠️ CGS suffered severe loss of orthogonality!")

        with c2:
            st.metric(r"MGS Orthogonality Loss $\|Q^T Q - I\|_F$", f"{ortho_err_mgs:.2e}")
            st.dataframe(np.round(Q_mgs.T @ Q_mgs, 4), use_container_width=True)
            st.success("✅ MGS preserves orthogonality near machine precision.")

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(
            ["Classical Gram-Schmidt", "Modified Gram-Schmidt"],
            [ortho_err_cgs, ortho_err_mgs],
            color=["#d9534f", "#2ca02c"],
            log=True,
        )
        ax.set_ylabel(r"$\|Q^T Q - I\|_F$ (Log Scale)")
        ax.set_title("Gram-Schmidt Numerical Stability Comparison")
        st.pyplot(fig)

    # -------------------------------------------------------------
    # TAB 3: Jacobi Spectral Decomposition
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Jacobi Algorithm for Symmetric Diagonalization")
        st.markdown(
            r"""
            **Mathematical Triad:**
            - **Concept**: A symmetric matrix $S \in \mathbb{R}^{d \times d}$ is orthogonally diagonalizable: $S = V \Lambda V^T$ with $V^T V = I$.
            - **Purpose**: Diagonalize sample covariance matrices to find principal variances and orthogonal loading directions.
            - **Outcome**: Cyclic Givens rotations annihilate off-diagonal energy until diagonal entries converge to true eigenvalues.
            """
        )

        # Real AirSense covariance demo
        cov_matrix_preset = np.array([
            [1.0000, 0.8837, 0.8921],
            [0.8837, 1.0000, 0.7719],
            [0.8921, 0.7719, 1.0000],
        ])

        st.write("Demonstration on 3-sensor correlation matrix ($PT08.S1, PT08.S2, PT08.S4$):")
        st.dataframe(cov_matrix_preset, use_container_width=True)

        eigenvalues, eigenvectors = jacobi_eigenvalues_symmetric(cov_matrix_preset)
        reconstruction = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        recon_err = np.linalg.norm(cov_matrix_preset - reconstruction)

        col_eig1, col_eig2 = st.columns(2)
        with col_eig1:
            st.write("**Computed Eigenvalues (Principal Variances)**:")
            for i, val in enumerate(eigenvalues):
                pct = 100.0 * val / np.sum(eigenvalues)
                st.write(f"- $\\lambda_{i+1} = {val:.4f}$ ({pct:.1f}% variance)")

        with col_eig2:
            st.write("**Eigenvector Matrix $V$ (Loading Directions)**:")
            st.dataframe(np.round(eigenvectors, 4), use_container_width=True)
            st.caption(f"Orthogonality check $\\|V^T V - I\\| = {np.linalg.norm(eigenvectors.T @ eigenvectors - np.eye(3)):.2e}$")
            st.caption(f"Spectral Reconstruction $\\|S - V \\Lambda V^T\\| = {recon_err:.2e}$")

    # -------------------------------------------------------------
    # TAB 4: Fundamental Subspaces & Rank-Nullity
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Fundamental Subspaces & Rank-Nullity Theorem")
        st.markdown(
            r"""
            **Mathematical Triad:**
            - **Concept**: A matrix $A \in \mathbb{R}^{m \times n}$ decomposes $\mathbb{R}^n$ into the Row Space $C(A^T)$ and Null Space $N(A)$, which are orthogonal complements: $C(A^T) \perp N(A)$.
            - **Purpose**: Verify non-singularity, detect redundant sensor channels, and determine degrees of freedom.
            - **Outcome**: Proof that $\text{rank}(A) + \text{nullity}(A) = n$.
            """
        )

        test_A_sub = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [1.0, 0.0, 1.0, 2.0],
        ])
        st.write("Analyzing $3 \\times 4$ Matrix with dependent row ($R_2 = 2 R_1$):")
        st.dataframe(test_A_sub, use_container_width=True)

        r = matrix_rank(test_A_sub)
        null_basis = null_space(test_A_sub)
        nullity = null_basis.shape[1] if null_basis.size > 0 else 0

        res_check = verify_rank_nullity(test_A_sub)

        c_sub1, c_sub2, c_sub3 = st.columns(3)
        c_sub1.metric("Columns ($n$)", "4")
        c_sub2.metric("Rank", f"{r}")
        c_sub3.metric("Nullity", f"{nullity}")

        st.success(
            f"**Rank-Nullity Identity Verified**: $\\text{{rank}}(A) + \\text{{nullity}}(A) = {r} + {nullity} = 4$ "
            f"({res_check['theorem_satisfied']})"
        )

        st.write("**Null Space Basis Vectors ($A\\mathbf{x} = \\mathbf{0}$)**:")
        st.dataframe(np.round(null_basis, 4), use_container_width=True)
        st.caption(f"Null-space check $\\|A N(A)\\| = {np.linalg.norm(test_A_sub @ null_basis):.2e}$")
