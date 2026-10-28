"""Methodology and Viva Defense View for AirSense Streamlit Interface.

Provides:
- Core Mathematical Equations & Derivations
- Statistical Assumption Diagnostics (Normality, Independence, Linearity)
- Numerical Verification Invariant Tables & Machine Precision Tolerances
- Concept -> Purpose -> Outcome 10-Point Viva Framework
- Anticipated Viva Examiner Questions & Structured Defense Answers
"""

import pandas as pd
import streamlit as st


def render_methodology_view() -> None:
    """Render Section 8: Methodology and Viva."""
    st.header("🎓 8. Methodology & Viva Defense Dossier")
    st.markdown(
        r"""
        This dossier equips the student for university viva defense and technical review, presenting 
        formal derivations, diagnostic checks of modeling assumptions, numerical verification 
        invariants, and structured **Concept $\to$ Purpose $\to$ Outcome** responses.
        """
    )

    t1, t2, t3, t4 = st.tabs(
        [
            "1. Concept -> Purpose -> Outcome (Viva Triad)",
            "2. Mathematical Equations & Derivations",
            "3. Modeling Assumptions & Diagnostics",
            "4. Viva Examiner Q&A Simulator",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: Viva Triad (10 Points)
    # -------------------------------------------------------------
    with t1:
        st.subheader("10-Point Viva Defense Framework")
        st.markdown(
            r"""
            Every linear algebra and statistical method in AirSense is articulated through the viva paradigm:
            $$\mathbf{Concept} \longrightarrow \mathbf{Purpose} \longrightarrow \mathbf{Outcome}$$
            """
        )

        concepts = [
            (
                "1. Matrix Representation of Physical Sensors",
                r"Continuous multi-channel field measurements form design matrix $X \in \mathbb{R}^{n \times d}$, where each row is an hourly state and each column is a physical sensor trajectory.",
                r"Formulate sensor calibration, cross-sensitivity unmixing, and environmental correction as linear vector operations.",
                r"Processed 8,991 supervised hours across 8 predictor channels predicting certified reference benzene $C_6H_6(GT)$."
            ),
            (
                "2. Gaussian Elimination & LU Decomposition",
                r"Row operations with partial pivoting decompose $A$ into $PA = LU$, where $P$ is a permutation matrix, $L$ is unit lower-triangular, and $U$ is upper-triangular.",
                r"Solve linear systems $A\mathbf{x} = \mathbf{b}$ in $\mathcal{O}(n^3)$ operations without matrix inversion, avoiding zero pivots.",
                r"Implemented from scratch; verified $\|PA - LU\|_\infty < 10^{-12}$ against SciPy."
            ),
            (
                "3. Fundamental Subspaces & Rank-Nullity",
                r"Every matrix $A \in \mathbb{R}^{m \times n}$ partitions $\mathbb{R}^n$ into Row Space $C(A^T)$ and Null Space $N(A)$, which are orthogonal complements. Rank-Nullity states $\text{rank}(A) + \text{nullity}(A) = n$.",
                r"Detect redundant sensor channels and verify system non-singularity.",
                r"Proved feature matrix has full column rank $r = 8$ and nullity $0$, but near-deficiency due to collinearity."
            ),
            (
                "4. Metric Spaces & Cauchy-Schwarz",
                r"Inner product $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v}$ induces Euclidean norm $\|\mathbf{u}\|_2$. Cauchy-Schwarz bounds $|\mathbf{u}^T \mathbf{v}| \le \|\mathbf{u}\| \|\mathbf{v}\|$, defining angle $\cos \theta$.",
                r"Quantify sensor collinearity as geometric angles in $\mathbb{R}^n$ and prediction errors as residual vector lengths.",
                r"Proved high sensor correlation ($r > 0.88$) corresponds to small geometric angles ($\theta \approx 28.3^\circ$)."
            ),
            (
                "5. Gram-Schmidt & QR Stability",
                r"Gram-Schmidt constructs orthonormal basis $Q$ ($Q^T Q = I$). Classical (CGS) projects against raw vectors; Modified (MGS) projects against updated intermediate residuals.",
                r"Prevent catastrophic floating-point cancellation when orthogonalizing collinear sensor signals.",
                r"MGS achieves $7.07 \times 10^6$-fold precision gain over CGS on near-collinear sensor matrices."
            ),
            (
                "6. Orthogonal Projections & Least-Squares",
                r"Projection operator $P = X(X^T X)^{-1}X^T$ satisfies $P^2=P, P=P^T$. Residual error $\mathbf{e} = (I - P)\mathbf{y}$ is strictly orthogonal to column space: $X^T \mathbf{e} = \mathbf{0}$.",
                r"Find optimal calibration weights minimizing Sum of Squared Errors.",
                r"Solved via QR ($R\mathbf{w} = Q^T \mathbf{y}$), avoiding Normal Equations condition number squaring ($\kappa = 1.89 \times 10^4$ vs. $3.57 \times 10^8$)."
            ),
            (
                "7. Covariance & Spectral Theorem",
                r"Sample covariance $S = \frac{1}{n-1}\tilde{X}^T \tilde{X}$ is symmetric and positive semi-definite. Spectral Theorem guarantees $S = V \Lambda V^T$ with $V^T V = I$.",
                r"Uncouple collinear sensor signals along principal dispersion axes.",
                r"Implemented Cyclic Jacobi rotations, diagonalizing covariance with machine-precision parity to SciPy eigh ($< 4.24 \times 10^{-13}$)."
            ),
            (
                "8. SVD & Eckart-Young PCA",
                r"Thin SVD $\tilde{X} = U \Sigma V^T$ directly yields covariance eigenvalues $\lambda_i = \sigma_i^2 / (n-1)$. Eckart-Young bounds rank-$k$ error: $\|\tilde{X} - \hat{X}_k\|_F^2 = \sum_{j=k+1}^d \sigma_j^2$.",
                r"Compress 8 sensor channels into $k < 8$ uncorrelated components, filtering out noise.",
                r"PCR ($k=3$) achieves superior out-of-time test accuracy ($R^2 = 0.8626, \text{RMSE} = 2.415$) over Full OLS ($R^2 = 0.8079, \text{RMSE} = 2.855$)."
            ),
            (
                "9. Normal MLE & Bayesian MAP Regularization",
                r"Gaussian log-likelihood yields OLS weights and biased variance $\hat{\sigma}^2_{\text{MLE}}$. Gaussian prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$ yields MAP estimate identical to Ridge regression $\lambda = \sigma^2 / \tau^2$.",
                r"Provide probabilistic foundations for least-squares and formalize weight shrinkage.",
                r"Proved that as prior variance $\tau^2 \to \infty$ ($\lambda \to 0$), MAP converges exactly to OLS."
            ),
            (
                "10. Hypothesis Testing, CI Duality & ANOVA",
                r"Two-sided test of $H_0: \mu = \mu_0$ rejects $H_0 \iff \mu_0 \notin [ \bar{x} \pm t_{\alpha/2} \frac{s}{\sqrt{n}} ]$. ANOVA partitions variation $SS_{\text{Total}} = SS_{\text{Between}} + SS_{\text{Within}}$.",
                r"Statistically evaluate error shifts and test for seasonal pollutant variations.",
                r"Empirically verified exact CI duality; ANOVA confirmed significant seasonal variation ($F = 214.3, p < 10^{-15}$)."
            ),
        ]

        for title, concept, purpose, outcome in concepts:
            with st.expander(f"📌 {title}", expanded=False):
                st.markdown(f"**Concept**: {concept}")
                st.markdown(f"**Purpose**: {purpose}")
                st.markdown(f"**Outcome**: {outcome}")

    # -------------------------------------------------------------
    # TAB 2: Mathematical Equations & Derivations
    # -------------------------------------------------------------
    with t2:
        st.subheader("Key Derivations and Numerical Tolerances")
        st.markdown(
            r"""
            ### 1. Condition Number Squaring Proof
            Let $X = U \Sigma V^T$ be the SVD of the design matrix, with singular values $\sigma_1 \ge \dots \ge \sigma_d > 0$.
            The 2-norm condition number of $X$ is:
            $$\kappa(X) = \|X\|_2 \|X^+\|_2 = \frac{\sigma_1}{\sigma_d}$$
            The normal equations matrix is $A = X^T X = V \Sigma U^T U \Sigma V^T = V \Sigma^2 V^T$.
            Its eigenvalues are $\lambda_i = \sigma_i^2$. Therefore:
            $$\kappa(X^T X) = \frac{\lambda_1}{\lambda_d} = \frac{\sigma_1^2}{\sigma_d^2} = \left(\frac{\sigma_1}{\sigma_d}\right)^2 = [\kappa(X)]^2$$
            *Implication*: In AirSense, $\kappa(X) = 1.89 \times 10^4 \implies \kappa(X^T X) = 3.57 \times 10^8$. 
            Using QR decomposition preserves $\kappa(X)$ and avoids losing $\sim 8.5$ decimal digits of precision.
            
            ---
            
            ### 2. Physical Feature Weights Back-Projection
            Let the PCA transformation be $Z_k = \tilde{X}_{\text{std}} V_k \in \mathbb{R}^{n \times k}$, where $\tilde{X}_{\text{std}} = (X - \mathbf{1}\boldsymbol{\mu}^T) D_\sigma^{-1}$.
            The PCR regression model is:
            $$\hat{\mathbf{y}} = w_0 + Z_k \mathbf{w}_{\text{pca}} = w_0 + (X - \mathbf{1}\boldsymbol{\mu}^T) D_\sigma^{-1} V_k \mathbf{w}_{\text{pca}}$$
            Expanding with respect to raw feature matrix $X$:
            $$\hat{\mathbf{y}} = \left(w_0 - \boldsymbol{\mu}^T D_\sigma^{-1} V_k \mathbf{w}_{\text{pca}}\right) + X \left(D_\sigma^{-1} V_k \mathbf{w}_{\text{pca}}\right)$$
            Therefore, the back-projected physical feature weight vector is:
            $$\mathbf{w}_{\text{feature}} = D_\sigma^{-1} V_k \mathbf{w}_{\text{pca}} \in \mathbb{R}^d$$
            """
        )

    # -------------------------------------------------------------
    # TAB 3: Modeling Assumptions & Diagnostics
    # -------------------------------------------------------------
    with t3:
        st.subheader("Statistical Diagnostics & Boundary Checks")
        st.markdown(
            """
            Scientific models must be evaluated against the classical Gauss-Markov linear regression assumptions:
            """
        )

        diag_data = [
            ("Linearity", "Residuals vs Fitted Values", "Mild non-linearity at high concentrations", "Pass with caveats (power-law kinetics)"),
            ("Independence", "Durbin-Watson Test", "DW = 0.2062 << 2.0 (Autocorrelation)", "FAIL: Hourly readings are serially correlated"),
            ("Homoscedasticity", "Breusch-Pagan / Levene Test", "Residual variance increases with concentration", "FAIL: Heteroscedastic error variance"),
            ("Normality of Errors", "Shapiro-Wilk / D'Agostino Test", "p < 10^-5 (Heavy positive skewness)", "FAIL: Residuals have leptokurtic tails"),
            ("No Multicollinearity", "Variance Inflation Factor (VIF)", "VIF > 25 across PT08 sensors", "FAIL: Severe collinearity (mitigated by PCR)"),
        ]

        st.dataframe(pd.DataFrame(diag_data, columns=["Assumption", "Diagnostic Tool", "Observed AirSense Metric", "Scientific Verdict"]), use_container_width=True)

        st.warning(
            "⚠️ **Gauss-Markov Reality Check**: Real-world chemical sensor residuals violate i.i.d. Gaussian assumptions! "
            "This empirical reality justifies why robust standard errors (HAC / Newey-West) and non-parametric bootstrap "
            "must be utilized when interpreting parameter uncertainties."
        )

    # -------------------------------------------------------------
    # TAB 4: Viva Examiner Q&A Simulator
    # -------------------------------------------------------------
    with t4:
        st.subheader("Anticipated Viva Examiner Questions & Model Answers")

        q_list = [
            (
                "Why did you exclude reference analyzer columns ending in (GT) from your predictors?",
                "Columns ending in (GT) represent certified ground-truth analyzers. Including CO(GT) or NOx(GT) to predict C6H6(GT) would cause target leakage: in an operational low-cost deployment, these expensive analyzers are absent. The engine must calibrate using low-cost sensor resistances only."
            ),
            (
                "Why does Principal Component Regression with k=3 outperform Full OLS on the test set, even though Full OLS has higher R² on the training set?",
                "The 8 features exhibit severe multicollinearity (κ(X) ≈ 1.89e4). Full OLS utilizes all 8 dimensions, allowing low-variance components to fit transient seasonal noise in the training set (e.g. k=7 captures seasonal temperature interactions). By truncating to k=3 or k=4 components (94-96% variance), PCR acts as a regularizer, eliminating collinear noise and achieving 15.4% lower RMSE on out-of-time test data."
            ),
            (
                "Why is Modified Gram-Schmidt preferred over Classical Gram-Schmidt?",
                "In exact arithmetic, CGS and MGS are identical. But in IEEE 754 floating-point arithmetic, CGS projects against raw vectors, leading to catastrophic cancellation when vectors are nearly collinear. MGS successively updates intermediate vectors, preserving orthogonality to near machine precision (||Q^T Q - I|| ≈ 10^-9 vs 10^-2 for CGS)."
            ),
            (
                "Can you claim you have statistically proven sensor drift?",
                "No. While calibration residuals drift negative by -4.13 µg/m³ over 13 months, outdoor field measurements are confounded by seasonal weather shifts (temperature inversions and humidity). Without periodic chamber recalibration with standard reference gas cylinders, we can only claim changing error dynamics and temporal non-stationarity, not isolated physical sensor drift."
            ),
            (
                "Why did you use QR decomposition instead of Normal Equations for least-squares?",
                "The Normal Equations (X^T X)w = X^T y square the condition number: κ(X^T X) = [κ(X)]^2. With κ(X) ≈ 1.89e4, κ(X^T X) exceeds 3.57e8, destroying ~8.5 decimal digits of precision. QR factorization solves Rw = Q^T y without squaring the condition number, preserving numerical stability."
            ),
        ]

        for q, a in q_list:
            with st.expander(f"❓ {q}"):
                st.markdown(f"**Model Viva Answer**: {a}")
