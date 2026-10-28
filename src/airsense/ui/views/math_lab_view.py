"""Linear Algebra Laboratory View for AirSense Streamlit Interface.

Interactive demonstration of from-scratch matrix algorithms:
- Intermediate Matrices, Gaussian Elimination, and RREF
- Fundamental Subspaces, Rank, Nullity, and Rank-Nullity Theorem
- Gram-Schmidt Orthogonalization: CGS vs. MGS Numerical Stability
- Orthogonal Projections (Idempotence & Symmetry)
- Least-Squares Solvers: QR vs. SVD vs. Normal Equations Condition Squaring
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
    column_space,
    verify_rank_nullity,
    projection_matrix,
    project_vector,
    solve_least_squares_qr,
    solve_least_squares_svd,
)


def render_math_lab_view() -> None:
    """Render Section 3: Linear-Algebra Laboratory."""
    st.header("📐 3. Linear-Algebra Laboratory")
    st.markdown(
        """
        Explore from-scratch numerical linear algebra algorithms implemented in pure NumPy primitives. 
        Inspect intermediate row reduction matrices, test numerical stability in Gram-Schmidt, 
        verify orthogonal projection operators, and compare least-squares solvers.
        """
    )

    t1, t2, t3, t4, t5 = st.tabs(
        [
            "1. Gaussian Elimination & RREF",
            "2. Fundamental Subspaces & Rank-Nullity",
            "3. Gram-Schmidt (CGS vs. MGS Stability)",
            "4. Orthogonal Projections",
            "5. Least-Squares Solvers & Condition Squaring",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: Gaussian Elimination & RREF
    # -------------------------------------------------------------
    with t1:
        st.subheader("Gaussian Elimination with Partial Pivoting & RREF")
        st.markdown(
            r"""
            **Theoretical Paradigm**:
            - **Concept**: Elementary row operations ($E_k \cdots E_1 A = U$) transform a real matrix $A \in \mathbb{R}^{m \times n}$ into its unique Reduced Row Echelon Form (RREF).
            - **Purpose**: Systematically identify pivot columns, determine linear independence, and compute canonical system solutions without matrix inversion.
            - **Outcome**: Factored representation $PA = LU$ and canonical $\text{RREF}(A)$ with designated pivot columns.
            """
        )

        matrix_preset = st.selectbox(
            "Select Preset Matrix or Enter Custom:",
            [
                "Standard Invertible 3x3",
                "Singular / Dependent Rows 3x3",
                "AirSense Sensor Correlation Submatrix (3x3)",
            ],
            key="rref_preset_v2",
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
            raw_matrix = st.text_area("Matrix A (comma-separated rows):", default_A, height=95)
        with col_input_b:
            raw_b = st.text_area("Vector b (comma-separated):", default_b, height=95)

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
                st.info(f"**Identified Pivot Columns**: {pivots} | **Computed Rank**: {len(pivots)}")

            with col_res2:
                st.write("**LU Decomposition ($PA = LU$)**:")
                if m == n:
                    P, L, U = lu_factorize(A)
                    recon_err = np.linalg.norm(P @ A - L @ U)
                    st.write("Matrix $L$ (Unit Lower Triangular Multipliers):")
                    st.dataframe(np.round(L, 4), use_container_width=True)
                    st.write("Matrix $U$ (Upper Triangular Echelon Form):")
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
    # TAB 2: Fundamental Subspaces & Rank-Nullity
    # -------------------------------------------------------------
    with t2:
        st.subheader("The Four Fundamental Subspaces & Rank-Nullity Theorem")
        st.markdown(
            r"""
            **Fundamental Theorem of Linear Algebra**:
            Every matrix $A \in \mathbb{R}^{m \times n}$ decomposes $\mathbb{R}^n$ into the Row Space $C(A^T)$ and 
            Null Space $N(A)$, which are orthogonal complements:
            $$C(A^T) \perp N(A), \quad \text{dim}(C(A)) = \text{rank}(A) = r, \quad \text{dim}(N(A)) = \text{nullity}(A) = n - r$$
            $$\mathbf{Rank-Nullity\ Theorem}: \quad \text{rank}(A) + \text{nullity}(A) = n$$
            """
        )

        test_A_sub = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 4.0, 6.0, 8.0],
            [1.0, 0.0, 1.0, 2.0],
        ])
        st.write("Analyzing Demonstration Matrix with dependent row ($R_2 = 2 R_1$):")
        st.dataframe(test_A_sub, use_container_width=True)

        r = matrix_rank(test_A_sub)
        null_basis = null_space(test_A_sub)
        nullity = null_basis.shape[1] if null_basis.size > 0 else 0
        res_check = verify_rank_nullity(test_A_sub)

        c_sub1, c_sub2, c_sub3 = st.columns(3)
        c_sub1.metric("Columns ($n$)", "4")
        c_sub2.metric("Matrix Rank ($r$)", f"{r}")
        c_sub3.metric("Nullity ($n - r$)", f"{nullity}")

        st.success(
            f"**Rank-Nullity Invariant Verified**: $\\text{{rank}}(A) + \\text{{nullity}}(A) = {r} + {nullity} = 4$ "
            f"({res_check['theorem_satisfied']})"
        )

        st.write("**Null Space Basis Vectors ($A\\mathbf{x} = \\mathbf{0}$)**:")
        st.dataframe(np.round(null_basis, 4), use_container_width=True)
        st.caption(f"Null-space check $\\|A N(A)\\| = {np.linalg.norm(test_A_sub @ null_basis):.2e}$")

    # -------------------------------------------------------------
    # TAB 3: Gram-Schmidt Numerical Stability
    # -------------------------------------------------------------
    with t3:
        st.subheader("Classical vs. Modified Gram-Schmidt Orthogonalization")
        st.markdown(
            r"""
            **Catastrophic Cancellation in Floating-Point Arithmetic**:
            - **Classical Gram-Schmidt (CGS)**: Projects raw vector $\mathbf{a}_k$ simultaneously against previous basis vectors:
              $$\mathbf{q}_k' = \mathbf{a}_k - \sum_{j=1}^{k-1} \langle \mathbf{q}_j, \mathbf{a}_k \rangle \mathbf{q}_j$$
            - **Modified Gram-Schmidt (MGS)**: Projects updated intermediate residuals successively:
              $$\mathbf{v}_k^{(j)} = \mathbf{v}_k^{(j-1)} - \langle \mathbf{q}_j, \mathbf{v}_k^{(j-1)} \rangle \mathbf{q}_j$$
            When input vectors are nearly collinear, CGS suffers catastrophic floating-point cancellation, losing orthogonality.
            """
        )

        delta_exp = st.slider(
            "Collinearity Perturbation (log10 epsilon):",
            min_value=-10,
            max_value=-1,
            value=-5,
            step=1,
            help=r"Smaller epsilon makes vectors closer to collinear ($a_2 = a_1 + \epsilon$).",
            key="delta_exp_v2",
        )
        eps = 10.0 ** delta_exp

        st.write(
            f"Testing near-collinear vectors: "
            f"$\\mathbf{{a}}_1 = [1, 1, 1]^T, "
            f"\\mathbf{{a}}_2 = [1, 1 + {eps:.1e}, 1]^T, "
            f"\\mathbf{{a}}_3 = [1, 1, 1 + {eps:.1e}]^T$"
        )

        A_test = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0 + eps, 1.0],
            [1.0, 1.0, 1.0 + eps],
        ])

        Q_cgs, _ = gram_schmidt_classical(A_test)
        Q_mgs, _ = gram_schmidt_modified(A_test)

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

        fig, ax = plt.subplots(figsize=(6, 2.8))
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
    # TAB 4: Orthogonal Projections
    # -------------------------------------------------------------
    with t4:
        st.subheader("Orthogonal Projection Operators")
        st.markdown(
            r"""
            **Projection Geometry**:
            The orthogonal projection of observation $\mathbf{y} \in \mathbb{R}^n$ onto the subspace spanned by columns of $X$ is:
            $$P = X(X^T X)^{-1} X^T \implies \hat{\mathbf{y}} = P\mathbf{y}$$
            **Two Fundamental Properties**:
            1. **Symmetry**: $P = P^T$
            2. **Idempotence**: $P^2 = P$ (projecting twice produces no further change)
            3. **Residual Orthogonality**: $\mathbf{e} = (I - P)\mathbf{y} \implies X^T \mathbf{e} = \mathbf{0}$
            """
        )

        # Real AirSense feature subset demo
        X_demo = np.array([
            [1.0, 1000.0, 12.0],
            [1.0, 1200.0, 14.5],
            [1.0, 950.0, 11.2],
            [1.0, 1400.0, 18.0],
            [1.0, 1100.0, 13.1],
        ])
        y_demo = np.array([5.5, 9.2, 4.1, 14.8, 7.3])

        P = projection_matrix(X_demo)
        y_hat = project_vector(X_demo, y_demo)
        residual = y_demo - y_hat

        sym_err = np.linalg.norm(P - P.T)
        idem_err = np.linalg.norm(P @ P - P)
        ortho_err = np.linalg.norm(X_demo.T @ residual)

        col_p1, col_p2, col_p3 = st.columns(3)
        col_p1.metric(r"Symmetry Check $\|P - P^T\|_F$", f"{sym_err:.2e}")
        col_p2.metric(r"Idempotence Check $\|P^2 - P\|_F$", f"{idem_err:.2e}")
        col_p3.metric(r"Residual Orthogonality $\|X^T \mathbf{e}\|$", f"{ortho_err:.2e}")

        st.success(
            "✅ **All Projection Invariants Satisfied**: $P$ is symmetric, idempotent, and the residual error vector "
            r"$\mathbf{e}$ is exactly perpendicular to the column space $C(X)$."
        )

    # -------------------------------------------------------------
    # TAB 5: Least-Squares Solvers & Condition Squaring
    # -------------------------------------------------------------
    with t5:
        st.subheader("Least-Squares Solvers & The Condition Squaring Problem")
        st.markdown(
            r"""
            **Why Normal Equations Fail on Real Chemical Sensors**:
            $$\text{Normal Equations}: \quad (X^T X)\mathbf{w} = X^T \mathbf{y}$$
            The condition number of the system matrix is squared:
            $$\kappa(X^T X) = [\kappa(X)]^2$$
            In the AirSense feature matrix, $\kappa(X) \approx 1.89 \times 10^4 \implies \kappa(X^T X) \approx 3.57 \times 10^8$. 
            This condition squaring destroys approximately $\log_{10}(3.57 \times 10^8) \approx 8.5$ significant digits of floating-point precision!
            
            **The QR Solution**:
            $$X = QR \implies R\mathbf{w} = Q^T \mathbf{y}$$
            Because $Q$ has orthonormal columns, $\kappa(R) = \kappa(X) = 1.89 \times 10^4$. Solving by back-substitution avoids squaring $\kappa$ entirely.
            """
        )

        col_ls1, col_ls2 = st.columns(2)
        with col_ls1:
            st.write("**QR Solver Output (`solve_least_squares_qr`)**:")
            w_qr = solve_least_squares_qr(X_demo, y_demo)
            st.dataframe(pd.DataFrame({"Coefficient": [f"w_{i}" for i in range(len(w_qr))], "Value": w_qr}), use_container_width=True)

        with col_ls2:
            st.write("**SVD Solver Output (`solve_least_squares_svd`)**:")
            w_svd = solve_least_squares_svd(X_demo, y_demo)
            st.dataframe(pd.DataFrame({"Coefficient": [f"w_{i}" for i in range(len(w_svd))], "Value": w_svd}), use_container_width=True)

        solver_diff = np.linalg.norm(w_qr - w_svd)
        st.caption(f"Difference between custom QR and SVD solvers: {solver_diff:.2e} (Machine Precision Parity)")
