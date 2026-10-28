# AirSense: A Mathematical Engine for Sensor Calibration, PCA, and Drift Monitoring

[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-53%20passed-brightgreen.svg)](tests/)
[![Dataset](https://img.shields.io/badge/UCI-Air%20Quality%20(ID%20360)-orange.svg)](https://archive.ics.uci.edu/dataset/360/air+quality)
[![Architecture](https://img.shields.io/badge/algorithms-from--scratch%20%2B%20verified-purple.svg)](src/airsense/math_core.py)
[![UI](https://img.shields.io/badge/dashboard-Streamlit-red.svg)](src/airsense/ui/app.py)

AirSense is a scientific computing and linear algebra research engine engineered for the university syllabus *Mathematical Foundations for AI and Data Science*. It solves real-world environmental sensor calibration challenges using the official **UCI Machine Learning Repository Air Quality Dataset** (March 2004 – April 2005).

Every mathematical concept is implemented from scratch using NumPy/SciPy primitives, tested for exact mathematical invariants (orthogonality, idempotence, spectral reconstruction), benchmarked against SciPy/scikit-learn references, and explained through the viva triad:
$$\mathbf{Concept} \longrightarrow \mathbf{Purpose} \longrightarrow \mathbf{Outcome}$$

---

## 🏛️ Key Research Question & Empirical Findings

> **Research Question**: *How does PCA-based dimensionality reduction affect the predictive accuracy and temporal stability of least-squares calibration for benzene concentration using real-world gas-sensor measurements?*

### Empirical Findings:
1. **Short-Term Fit vs. Long-Term Generalization**:
   - In-sample and short-term validation favor the **Full OLS Model ($k=8$)** ($\text{Train } R^2 = 0.9826$, $\text{Val } R^2 = 0.9423$), because subtle chemical selectivity information is distributed across all 5 metal-oxide sensors.
   - However, evaluated on the out-of-time test partition **one year later**, Full OLS degrades to $\text{Test } R^2 = 0.8079$ ($\text{RMSE} = 2.855\ \mu g/m^3$).
2. **Principal Component Regularization Outperforms Full OLS**:
   - Retaining **$k = 3$ or $k = 4$ principal components** ($94.04\%$ variance) acts as a variance regularizer, achieving superior out-of-time accuracy:
     $$\text{PCR } (k=3): \quad \text{Test } R^2 = \mathbf{0.8626}, \quad \text{Test RMSE} = \mathbf{2.415\ \mu g/m^3} \quad (15.4\%\text{ error reduction})$$
3. **The "Collinear Overfit Trap" ($k=7$)**:
   - Retaining $k=7$ components ($99.69\%$ variance) yields a strong validation fit ($R^2 = 0.9123$), but collapses on the test set ($\text{Test } R^2 = \mathbf{0.6225}$, $\text{RMSE} = 4.003\ \mu g/m^3$), because the 7th component fits transient seasonal training correlations.
4. **Physical Sensor Drift & Autocorrelation**:
   - Monthly Mean Bias drifts steadily from $+0.44\ \mu g/m^3$ down to **$-3.69\ \mu g/m^3$** (a net $-4.13\ \mu g/m^3$ shift) as sensor surface resistance drops.
   - Durbin-Watson statistic is $0.2062 \ll 2.0$, confirming severe positive serial autocorrelation in hourly errors.

### Comparative Benchmark Summary ($k \in \{1, \dots, 8\}$):
| Model | Components ($k$) | Cum. Var (%) | Val RMSE ($\mu g/m^3$) | Val $R^2$ | Test RMSE ($\mu g/m^3$) | Test $R^2$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| PCR | 1 | 55.89% | 3.720 | 0.8052 | 3.211 | 0.7571 |
| PCR | 2 | 78.08% | 3.588 | 0.8188 | 3.103 | 0.7732 |
| **PCR (Optimal)** | **3** | **94.04%** | **2.826** | **0.8876** | **2.415** | **0.8626** |
| **PCR (Robust)** | **4** | **96.50%** | **3.038** | **0.8701** | **2.380** | **0.8665** |
| PCR | 5 | 98.01% | 3.412 | 0.8362 | 2.817 | 0.8130 |
| PCR | 6 | 99.23% | 3.371 | 0.8401 | 2.978 | 0.7910 |
| PCR (Overfit Trap) | 7 | 99.69% | 2.496 | 0.9123 | 4.003 | 0.6225 |
| Full OLS | 8 | 100.00% | 2.024 | 0.9423 | 2.855 | 0.8079 |

---

## 🚀 Quickstart & Interactive Dashboard

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/username/AirSense.git
cd AirSense
pip install -e ".[dev]"
```

### 2. Run Test Suite
Validate all mathematical invariants and software pipelines:
```bash
pytest
```
*Output: 53 passed in ~5 seconds.*

### 3. Launch Interactive Streamlit Dashboard
```bash
streamlit run src/airsense/ui/app.py
```
The dashboard opens an interactive laboratory with 5 modules:
- 🏛️ **Overview & Audit**: Dataset ingestion, missingness profile, and baseline statistics.
- 📐 **Mathematical Foundations Lab**: Interactive Gaussian elimination, Gram-Schmidt numerical stability (CGS vs MGS), Jacobi diagonalization, and Rank-Nullity visualizer.
- 🔬 **Calibration & PCR Engine**: Dynamic slider for $k$ components, live scree plot, True vs. Predicted scatter plot, and physical feature sensitivity.
- 📉 **Sensor Drift & Residuals**: 13-month residual timeline, CUSUM drift alarm tuning ($\delta, H$), monthly bias progression, and weather confounding.
- 📊 **Probability & Inference Lab**: Interactive Central Limit Theorem (CLT) simulator, Normal MLE vs Bessel's bias, hypothesis testing & CI duality, and seasonal ANOVA.

---

## 🧮 Mathematical Engine Overview (`airsense.math_core`)

All core mathematical routines are implemented from first principles:
- **Gaussian Elimination & LU Factorization**: Partial pivoting row interchanges satisfying $\|PA - LU\|_\infty < 10^{-12}$.
- **RREF & Pivot Tracking**: Exact column rank and null space basis computation satisfying $\text{rank}(A) + \text{nullity}(A) = n$.
- **Gram-Schmidt Orthogonalization**: Both Classical (CGS) and Modified (MGS) algorithms, demonstrating MGS's $7 \times 10^6\times$ numerical stability advantage on ill-conditioned matrices.
- **Orthogonal Projections**: Projectors satisfying idempotence $P^2 = P$ and symmetry $P = P^T$, with residual orthogonality $X^T \mathbf{e} = \mathbf{0}$.
- **Least-Squares Solvers**: Numerically stable QR and SVD solvers avoiding condition number squaring ($\kappa(X) = 1.89 \times 10^4$ vs. $\kappa(X^T X) = 3.57 \times 10^8$).
- **Cyclic Jacobi Diagonalization**: Analytic Givens rotations diagonalizing real symmetric covariance matrices with machine-precision parity to SciPy `eigh` ($< 4.24 \times 10^{-13}$).
- **Thin SVD & PCA**: Low-rank approximation satisfying the Eckart-Young-Mirsky theorem $\|\tilde{X} - \hat{X}_k\|_F^2 = \sum_{j=k+1}^d \sigma_j^2$.
- **Probabilistic Inference**: Normal MLE, Bayesian MAP linear regression (Ridge equivalence $\lambda = \sigma^2 / \tau^2$), Welch's $t$-test, One-Way ANOVA, and power curve simulations.

---

## 📂 Repository Architecture

```text
AirSense/
├── pyproject.toml              # Packaging and dependency configuration
├── requirements.txt            # Core production dependencies
├── requirements-dev.txt        # Development dependencies (pytest)
├── README.md                   # This project landing page
├── PROJECT_BRIEF.md            # University master project brief
├── IMPLEMENTATION_PLAN.md      # Engineering architecture & milestone specs
│
├── docs/                       # Research documentation & viva resources
│   ├── data_dictionary.md          # UCI schema, sentinel audit, exclusion rules
│   ├── mathematical_foundations.md # Complete linear algebra derivations
│   ├── experimental_protocol.md    # Leakage-free split & evaluation protocol
│   ├── statistical_methods.md      # Probability, estimation & inference theory
│   ├── assumptions_and_limitations.md # Modeling boundary diagnostics
│   ├── viva_cheat_sheet.md         # Concept -> Purpose -> Outcome 10-point viva pack
│   └── research_findings.md        # Formal academic paper synthesis
│
├── notebooks/                  # Interactive pedagogical laboratories
│   ├── 01_dataset_audit.ipynb
│   ├── 02_linear_algebra_lab.ipynb
│   ├── 03_calibration_experiments.ipynb
│   └── 04_probability_and_statistics_lab.ipynb
│
├── src/airsense/               # Modular Python package
│   ├── data_loader.py          # CSV dialect parser & leakage-free splitter
│   ├── math_core.py            # From-scratch linear algebra algorithms
│   ├── calibration.py          # Standardizer & linear calibration model
│   ├── pca_model.py            # PCR calibration with back-projected weights
│   ├── evaluation.py           # Regression metrics, Durbin-Watson, drift
│   ├── statistical_inference.py# CLT, MLE, MAP, hypothesis tests, ANOVA, power
│   └── ui/                     # Interactive Streamlit application
│       ├── app.py              # Main dashboard entrypoint
│       └── views/              # Modular view components (Math, PCR, Drift, Stats)
│
├── tests/                      # Automated invariant verification suite
│   ├── test_data_loader.py     # 10 tests: schema, sentinels, splits
│   ├── test_math_core.py       # 18 tests: LU, RREF, MGS, QR, Jacobi, SVD
│   ├── test_calibration.py     # 10 tests: OLS, PCR, condition numbers
│   ├── test_statistical_inference.py # 13 tests: CLT, MLE, duality, ANOVA
│   └── test_ui.py              # 2 tests: cached loaders, CUSUM logic
│
└── reports/                    # Generated experimental artifacts
    ├── experiment_results.csv  # Benchmark metrics across k=1..8
    └── figures/                # Publication-grade figures (01 to 09)
```

---

## 📖 Citation & License

This project was built for educational and research purposes using the **UCI Machine Learning Repository Air Quality Dataset**:
> S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia, *"On field calibration of an electronic nose for benzene estimation in an urban environment,"* Sensors and Actuators B: Chemical, vol. 129, no. 2, pp. 750-757, 2008.

Licensed under the MIT License.
