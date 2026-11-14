import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  BookOpen,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  Binary,
  Database,
  HelpCircle,
  Mic,
  Clock,
  Sparkles,
  ExternalLink,
  ChevronDown,
  ChevronUp,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { Card } from '../components/ui/Card';
import { MathInline } from '../components/ui/Math';
import { tabContentVariants } from '../lib/motion';

interface StepInfo {
  step: number;
  title: string;
  shortTitle: string;
  subtitle: string;
  route: string;
  routeLabel: string;
  problemSummary: React.ReactNode;
  triad: {
    concept: React.ReactNode;
    purpose: React.ReactNode;
    airsenseUsage: React.ReactNode;
    actualOutput: React.ReactNode;
    interpretation: React.ReactNode;
  };
  keyTakeaway: React.ReactNode;
}

const WALKTHROUGH_STEPS: StepInfo[] = [
  {
    step: 1,
    title: 'Step 1: Real-World Research Problem',
    shortTitle: '1. Problem Statement',
    subtitle: 'Estimating benzene concentration from low-cost metal-oxide chemical sensors',
    route: '/dataset',
    routeLabel: 'View Problem & Dataset',
    problemSummary: (
      <span>
        Benzene (<MathInline math="\mathrm{C}_6\mathrm{H}_6" />) is a toxic airborne aromatic hydrocarbon monitored under European clean air directives.
        Certified optical analyzers are bulky, delicate and cost upwards of €25,000. In contrast, solid-state metal-oxide semiconductor (MOX) sensors
        cost under €20, but exhibit high cross-sensitivity, thermal drift, and non-linear resistance responses. The research question: Can linear algebra
        and PCA-based calibration reliably predict true benzene concentrations from multi-sensor arrays over a continuous 13-month field deployment?
      </span>
    ),
    triad: {
      concept: 'Multivariate Calibration & Chemical Sensor Transduction',
      purpose: 'Map electrical resistance responses from 5 MOX sensors and 3 weather parameters into physical benzene concentration.',
      airsenseUsage: 'The project ingests 13 months of hourly field records from an Italian roadside monitoring station (UCI Air Quality #360).',
      actualOutput: '8,991 usable supervised hours recorded between March 10, 2004 and April 4, 2005.',
      interpretation: 'Demonstrates a real scientific calibration problem rather than synthetic or toy benchmark data.',
    },
    keyTakeaway: 'AirSense addresses the trade-off between expensive reference analyzers and affordable, noisy sensor arrays.',
  },
  {
    step: 2,
    title: 'Step 2: Dataset, Measurements & Ingestion Cleaning',
    shortTitle: '2. Dataset & Cleaning',
    subtitle: 'Data cleaning, missingness treatment, and anti-leakage feature selection',
    route: '/dataset',
    routeLabel: 'Inspect Ingestion Audit',
    problemSummary:
      'The raw dataset contains 9,357 hourly records stored in Italian CSV dialect (semicolon separators, comma decimals). Missing readings are encoded as the integer sentinel -200. The project parses European numbers, converts -200 to IEEE NaN, and inspects sensor dropout patterns.',
    triad: {
      concept: 'Missingness Pruning & Anti-Leakage Feature Filtering',
      purpose: 'Establish a clean supervised design matrix without introducing future information or target leakage.',
      airsenseUsage: 'Pruning 366 coincident blackout hours where the reference benzene instrument was offline leaves exactly 8,991 complete rows with zero missing values across all 8 predictor features.',
      actualOutput: '9,357 raw slots → 366 blackout rows pruned → 8,991 usable supervised rows.',
      interpretation: 'Reference analyzer columns (CO(GT), NMHC(GT), NOx(GT), NO2(GT)) are strictly excluded to prevent target contamination.',
    },
    keyTakeaway: 'Zero missing values in predictor features after removing instrument blackout hours; reference analyzer columns excluded to prevent leakage.',
  },
  {
    step: 3,
    title: 'Step 3: Matrix Representation of Real Data',
    shortTitle: '3. Matrix Representation',
    subtitle: 'Formulating empirical measurements into design matrix X and target vector y',
    route: '/dataset',
    routeLabel: 'Open Matrix Explorer',
    problemSummary: (
      <span>
        To apply linear algebra, physical measurements must become a matrix. Each hourly measurement forms an 8-dimensional row vector <MathInline math="\mathbf{x}_i^\mathsf{T}" /> consisting of 5 MOX sensor responses (PT08.S1 to PT08.S5) and 3 weather variables (T, RH, AH). The certified benzene measurement forms the target scalar <MathInline math="y_i" />.
      </span>
    ),
    triad: {
      concept: (
        <span>Design Matrix <MathInline math="X \in \mathbb{R}^{8991 \times 8}" /> and Target Vector <MathInline math="\mathbf{y} \in \mathbb{R}^{8991 \times 1}" /></span>
      ),
      purpose: 'Represent multi-channel temporal telemetry as standard matrix-vector algebraic entities.',
      airsenseUsage: (
        <span>Standardization <MathInline math="z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}" /> is fitted strictly on the 5,394 training rows to ensure dimensional invariance for PCA without data snooping.</span>
      ),
      actualOutput: (
        <span><MathInline math="n = 8991" /> valid rows, <MathInline math="d = 8" /> feature columns, 1 target vector with physical units (mV, °C, %, g/m³).</span>
      ),
      interpretation: 'Row i corresponds to a specific timestamp; columns correspond to physical sensor channels.',
    },
    keyTakeaway: (
      <span><MathInline math="X \in \mathbb{R}^{8991 \times 8}" /> and <MathInline math="\mathbf{y} \in \mathbb{R}^{8991 \times 1}" />; standardization parameters fitted exclusively on training data.</span>
    ),
  },
  {
    step: 4,
    title: 'Step 4: Mathematical Foundations & Primitives',
    shortTitle: '4. Mathematical Primitives',
    subtitle: 'Custom row reduction, orthogonalisation, least squares, and spectral SVD',
    route: '/linear-algebra',
    routeLabel: 'Open Linear Algebra Lab',
    problemSummary:
      'AirSense implements all core algorithms in Python without delegating the linear algebra to black-box solvers. Solvers include Gaussian elimination with partial pivoting, RREF, Modified Gram-Schmidt QR factorisation, Jacobi symmetric eigenvalue diagonalization, and Thin SVD.',
    triad: {
      concept: 'QR Factorisation & Condition Number Squaring Avoidance',
      purpose: (
        <span>Solve least-squares calibration <MathInline math="X^\mathsf{T} X \beta = X^\mathsf{T} \mathbf{y}" /> without squaring the condition number <MathInline math="\kappa(X)" />.</span>
      ),
      airsenseUsage: (
        <span>Modified Gram-Schmidt computes <MathInline math="X = QR" />, solving <MathInline math="R \beta = Q^\mathsf{T} \mathbf{y}" /> via stable back-substitution.</span>
      ),
      actualOutput: (
        <span><MathInline math="\kappa(X) \approx 1.89 \times 10^4" />. Normal equations square this to <MathInline math="\kappa(X^\mathsf{T} X) \approx 3.58 \times 10^8" />; QR preserves <MathInline math="\kappa(X)" />. MGS achieves <MathInline math="7.07 \times 10^6 \times" /> lower cancellation error than Classical GS.</span>
      ),
      interpretation: 'Numerical stability is preserved, avoiding catastrophic floating-point cancellation.',
    },
    keyTakeaway: (
      <span>MGS QR avoids condition number squaring; custom Jacobi and SVD routines verified against SciPy within machine precision (&lt; <MathInline math="10^{-14}" />).</span>
    ),
  },
  {
    step: 5,
    title: 'Step 5: Calibration Experiment & Zero-Leakage Splitting',
    shortTitle: '5. Calibration Experiment',
    subtitle: 'Contiguous chronological train, validation, and held-out test evaluation',
    route: '/calibration',
    routeLabel: 'Open Calibration Results',
    problemSummary:
      'Random k-fold cross-validation is invalid for temporal sensor data because serial autocorrelation (DW = 0.2062) allows adjacent hours to leak identical noise. AirSense enforces strictly contiguous chronological partitioning: Training (60%, 5,394 hrs), Validation (20%, 1,798 hrs), and Held-Out Test (20%, 1,799 hrs).',
    triad: {
      concept: 'Hyperparameter Tuning vs Held-Out Out-of-Time Generalization',
      purpose: (
        <span>Evaluate Full OLS against Principal Component Regression (PCR, <MathInline math="k = 1 \dots 8" />) under strict validation rules.</span>
      ),
      airsenseUsage: 'The model selection rule was applied strictly to the validation partition prior to touching test data.',
      actualOutput: (
        <span>Validation selection selected Full OLS (<MathInline math="k^* = 8" />), achieving Val RMSE = 2.024 <MathInline math="\mu\mathrm{g/m}^3" /> and Val <MathInline math="R^2 = 0.9423" />.</span>
      ),
      interpretation: 'Full OLS captures subtle chemical cross-sensitivities across all 5 metal-oxide channels during the short term.',
    },
    keyTakeaway: (
      <span>Validation tuning selected Full OLS (<MathInline math="k^* = 8" />); parameters frozen before evaluating held-out test data.</span>
    ),
  },
  {
    step: 6,
    title: 'Step 6: Verified Results & The Generalization Paradox',
    shortTitle: '6. Verified Results',
    subtitle: 'Held-out test performance and why compact PCR regularizes against drift',
    route: '/pca-explorer',
    routeLabel: 'Open PCA Explorer',
    problemSummary: (
      <span>
        When the validation-selected Full OLS model was evaluated on the untouched held-out test partition (winter 2005), its performance degraded to Test RMSE = 2.855 <MathInline math="\mu\mathrm{g/m}^3" /> (<MathInline math="R^2 = 0.8079" />). In post-hoc exploratory comparison, compact PCR (<MathInline math="k = 3" />) achieved lower test error: RMSE = 2.415 <MathInline math="\mu\mathrm{g/m}^3" /> (<MathInline math="R^2 = 0.8626" />, a 15.4% improvement).
      </span>
    ),
    triad: {
      concept: 'Spectral Truncation as Shrinkage Regularization',
      purpose: 'Understand why discarding trailing principal components improves out-of-time stability.',
      airsenseUsage: 'PCA Explorer proves that the first 3 components capture 94.04% of total variance. Dropping the smallest 5 components discards noise dimensions that overfit transient seasonal interactions.',
      actualOutput: (
        <span>Full OLS: Test RMSE 2.855 <MathInline math="\mu\mathrm{g/m}^3" /> vs Compact PCR (<MathInline math="k = 3" />): Test RMSE 2.415 <MathInline math="\mu\mathrm{g/m}^3" />.</span>
      ),
      interpretation: 'Retaining more variance does not guarantee better generalization. However, k = 3 is an exploratory finding, not the validation-selected model.',
    },
    keyTakeaway: (
      <span>PCR (<MathInline math="k = 3" />) regularizes against drift by filtering thermal noise; <MathInline math="k = 3" /> must be honestly reported as post-hoc.</span>
    ),
  },
  {
    step: 7,
    title: 'Step 7: Longitudinal Drift & Scientific Limitations',
    shortTitle: '7. Drift & Limitations',
    subtitle: 'The attribution dilemma: sensor degradation vs seasonal climate confounding',
    route: '/drift-analysis',
    routeLabel: 'Open Drift & Statistics',
    problemSummary: (
      <span>
        Over the 13 months, model prediction bias drifted from +0.44 <MathInline math="\mu\mathrm{g/m}^3" /> to -3.69 <MathInline math="\mu\mathrm{g/m}^3" />, and PT08.S4 dropped by -499.4 units (Cohen's <MathInline math="d = -1.87" />). While naive statistical tests flag severe drift, outdoor temperature simultaneously dropped from 23.1°C to 10.0°C (Cohen's <MathInline math="d = -1.87" />).
      </span>
    ),
    triad: {
      concept: 'The Attribution Dilemma & Non-Stationarity Diagnostics',
      purpose: 'Distinguish true physical sensor catalyst aging from outdoor weather confounding.',
      airsenseUsage: (
        <span>The CUSUM trajectory tracks cumulative error bias; Durbin-Watson diagnostic confirms <MathInline math="DW = 0.2062" /> (<MathInline math="\rho \approx 0.90" />).</span>
      ),
      actualOutput: (
        <span>Durbin-Watson <MathInline math="DW = 0.2062" />; standard errors are underestimated by <MathInline math="4.29 \times" /> under naive i.i.d. assumptions.</span>
      ),
      interpretation: 'Without periodic certified span gas in a controlled chamber, physical aging cannot be isolated from climate confounding.',
    },
    keyTakeaway: 'Severe sensor distribution shift tracks seasonal cooling; physical aging cannot be claimed without controlled span-gas chambers.',
  },
  {
    step: 8,
    title: 'Step 8: Academic Conclusion & Future Work',
    shortTitle: '8. Conclusion',
    subtitle: 'Summary of findings, methodology integrity, and operational recommendations',
    route: '/walkthrough',
    routeLabel: 'Viva Preparation',
    problemSummary:
      'AirSense proves that multivariate linear calibration translates solid-state gas sensor readings into meaningful benzene estimates, but field stability requires spectral shrinkage (PCR) and seasonal recalibration. The project demonstrates rigorous linear algebra, zero-leakage protocols, and candid reporting of scientific limitations.',
    triad: {
      concept: 'Synthesis of Mathematical Theory & Empirical Practice',
      purpose: 'Deliver a defensible, mathematically transparent university project.',
      airsenseUsage: 'The complete software suite pairs interactive visualisations with custom Python linear algebra implementations.',
      actualOutput: '88 passing automated tests, zero TypeScript errors, and zero fabricated performance metrics.',
      interpretation: 'Future work: hardware temperature-stabilized sensor chambers and multi-season calibration cycles.',
    },
    keyTakeaway: 'AirSense bridges linear algebra theory with real-world sensor challenges, backed by reproducible code and honest academic limits.',
  },
];

interface ConceptMapItem {
  name: string;
  category: 'pipeline' | 'analysis' | 'lab';
  categoryLabel: string;
  meaning: React.ReactNode;
  relevance: React.ReactNode;
  whereUsed: string;
  whatToShow: React.ReactNode;
  route: string;
}

const CONCEPT_MAP: ConceptMapItem[] = [
  {
    name: 'Matrix Representation (X and y)',
    category: 'pipeline',
    categoryLabel: 'Used directly in model pipeline',
    meaning: (
      <span>Representing observations as rows and features as columns: <MathInline math="X \in \mathbb{R}^{n \times 8}" />, <MathInline math="\mathbf{y} \in \mathbb{R}^{n \times 1}" />.</span>
    ),
    relevance: 'Transforms physical multi-sensor readings into a structured mathematical object for linear algebra operations.',
    whereUsed: 'Ingestion pipeline, Dataset & Matrix workspace, calibration model input.',
    whatToShow: 'Show the Dataset & Matrix Explorer with 8,991 rows, 8 feature columns, raw vs standardized values, and linked target vector y.',
    route: '/dataset',
  },
  {
    name: 'Chronological Splitting & Preprocessing',
    category: 'pipeline',
    categoryLabel: 'Used directly in model pipeline',
    meaning: 'Contiguous time-series partitioning: Train (60%), Validation (20%), Held-Out Test (20%).',
    relevance: 'Prevents data leakage and temporal snooping. Standardizer means/stds are fitted strictly on training data.',
    whereUsed: 'airsense.calibration.chronological_split, FeatureStandardizer.',
    whatToShow: 'Show the 3 non-overlapping date ranges and the zero-temporal-overlap assertion in the Partitions tab.',
    route: '/dataset',
  },
  {
    name: 'Modified Gram-Schmidt & QR Factorisation',
    category: 'pipeline',
    categoryLabel: 'Used directly in model pipeline',
    meaning: (
      <span>Factorising design matrix <MathInline math="X" /> into orthogonal <MathInline math="Q" /> (<MathInline math="Q^\mathsf{T} Q = I" />) and upper-triangular <MathInline math="R" /> (<MathInline math="X = QR" />).</span>
    ),
    relevance: (
      <span>Avoids condition number squaring <MathInline math="\kappa(X)^2" /> in normal equations; achieves <MathInline math="7.07 \times 10^6 \times" /> lower cancellation than Classical GS.</span>
    ),
    whereUsed: 'airsense.math_core.solve_least_squares_qr, airsense.calibration.LinearCalibrationModel.',
    whatToShow: (
      <span>Show the Gram-Schmidt comparison in Linear Algebra Lab: <MathInline math="\|Q^\mathsf{T} Q - I\|_F \approx 2.45 \times 10^{-16}" /> for MGS vs <MathInline math="1.73 \times 10^{-9}" /> for CGS.</span>
    ),
    route: '/linear-algebra',
  },
  {
    name: 'Least Squares Calibration',
    category: 'pipeline',
    categoryLabel: 'Used directly in model pipeline',
    meaning: (
      <span>Minimising the sum of squared prediction residuals <MathInline math="\|\mathbf{y} - X \beta\|_2^2" />.</span>
    ),
    relevance: (
      <span>Finds optimal linear calibration weights <MathInline math="\beta" /> to convert sensor voltages into benzene concentration.</span>
    ),
    whereUsed: 'airsense.calibration.LinearCalibrationModel, airsense.calibration.PCACalibrationModel.',
    whatToShow: 'Show the calibration regression weights and test RMSE in Calibration Results workspace.',
    route: '/calibration',
  },
  {
    name: 'Thin SVD & Principal Component Analysis',
    category: 'pipeline',
    categoryLabel: 'Used directly in model pipeline',
    meaning: (
      <span>Decomposing standardized data matrix <MathInline math="X_c = U \Sigma V^\mathsf{T}" /> to find orthogonal axes of maximum variance.</span>
    ),
    relevance: 'Reduces 8 collinear sensor features to k principal components; acts as shrinkage regularization against drift.',
    whereUsed: 'airsense.math_core.pca_svd, airsense.pca_model.PCACalibrationModel.',
    whatToShow: (
      <span>Show the Scree curve in PCA Explorer: first 3 components capture 94.04% variance, and Eckart-Young bounds parity (&lt; <MathInline math="10^{-14}" />).</span>
    ),
    route: '/pca-explorer',
  },
  {
    name: 'Subspace Projections & Geometry',
    category: 'analysis',
    categoryLabel: 'Used for analysis & diagnostics',
    meaning: (
      <span>Orthogonal projection of target vector <MathInline math="\mathbf{y}" /> onto the column space <MathInline math="C(X)" />: <MathInline math="\hat{\mathbf{y}} = P \mathbf{y} = X (X^\mathsf{T} X)^{-1} X^\mathsf{T} \mathbf{y}" />.</span>
    ),
    relevance: (
      <span>Proves geometrically that residual vector <MathInline math="\mathbf{e} = \mathbf{y} - \hat{\mathbf{y}}" /> is orthogonal to every feature column in <MathInline math="X" />: <MathInline math="X^\mathsf{T} \mathbf{e} = \mathbf{0}" />.</span>
    ),
    whereUsed: 'airsense.math_core.projection_matrix, Step 5 of Linear Algebra Lab.',
    whatToShow: (
      <span>Show the projection visualizer in Linear Algebra Lab verifying <MathInline math="\|X^\mathsf{T} \mathbf{e}\|_2 < 10^{-12}" />.</span>
    ),
    route: '/linear-algebra',
  },
  {
    name: 'Residual Analysis & CUSUM Tracking',
    category: 'analysis',
    categoryLabel: 'Used for analysis & diagnostics',
    meaning: (
      <span>Analyzing calibration errors <MathInline math="e_t = y_t - \hat{y}_t" /> and cumulative sum <MathInline math="S_t = \sum_{i=1}^t (e_i - \bar{e}_{\mathrm{train}})" />.</span>
    ),
    relevance: 'Tracks prediction bias and error inflation over time; detects when recalibration is required.',
    whereUsed: 'airsense.evaluation.compute_residuals, Drift & Statistics workspace.',
    whatToShow: 'Show the CUSUM trajectory bending downward in winter 2004, and the 14-month error progression table (+0.44 to -3.69 µg/m³).',
    route: '/drift-analysis',
  },
  {
    name: 'Autocorrelation & Durbin-Watson Test',
    category: 'analysis',
    categoryLabel: 'Used for analysis & diagnostics',
    meaning: (
      <span>Measuring first-order serial correlation in residuals: <MathInline math="DW = \frac{\sum_{t=2}^n (e_t - e_{t-1})^2}{\sum_{t=1}^n e_t^2} \approx 2(1 - \rho)" />.</span>
    ),
    relevance: (
      <span>Demonstrates that environmental time-series violate classical i.i.d. assumptions, causing OLS standard errors to be underestimated by <MathInline math="4.29 \times" /> (<MathInline math="\rho \approx 0.90" />).</span>
    ),
    whereUsed: 'airsense.evaluation.durbin_watson_statistic, Overview & Drift pages.',
    whatToShow: 'Show the DW statistic = 0.2062 badge on the Overview page and the autocorrelation explanation in Drift Analysis.',
    route: '/drift-analysis',
  },
  {
    name: 'Distribution Shifts & Kolmogorov-Smirnov Test',
    category: 'analysis',
    categoryLabel: 'Used for analysis & diagnostics',
    meaning: (
      <span>Non-parametric two-sample test measuring maximum divergence between reference and monitoring CDFs: <MathInline math="D = \sup_x |F_{\mathrm{ref}}(x) - F_{\mathrm{mon}}(x)|" />.</span>
    ),
    relevance: 'Quantifies sensor distribution shifts (Cohen\'s d, KS statistic) across continuous field deployment.',
    whereUsed: 'airsense.services.analyze_drift, Tab 2 of Drift & Statistics.',
    whatToShow: 'Show the cross-channel sensor drift table in Drift Analysis comparing PT08.S4 (d = -1.87) to outdoor temperature drop.',
    route: '/drift-analysis',
  },
  {
    name: 'Gaussian Elimination & RREF Player',
    category: 'lab',
    categoryLabel: 'Demonstrated in educational math lab',
    meaning: 'Systematic elementary row operations transforming a matrix into Reduced Row Echelon Form.',
    relevance: 'Educational syllabus requirement for Unit 1; demonstrates pivot selection and partial row swapping.',
    whereUsed: 'airsense.math_core.gaussian_elimination, airsense.math_core.rref, Linear Algebra Lab Step 1.',
    whatToShow: 'Show the step-by-step row reduction player in Linear Algebra Lab with Next/Previous step buttons.',
    route: '/linear-algebra',
  },
  {
    name: 'LU Decomposition with Partial Pivoting',
    category: 'lab',
    categoryLabel: 'Demonstrated in educational math lab',
    meaning: (
      <span>Factorising a square or rectangular matrix into <MathInline math="PA = LU" /> with lower and upper triangular factors.</span>
    ),
    relevance: 'Educational syllabus demonstration of matrix factorisation and pivoting stability.',
    whereUsed: 'airsense.math_core.lu_factorize, Linear Algebra Lab Step 2.',
    whatToShow: (
      <span>Show the P, L, and U factor matrices in Linear Algebra Lab verifying <MathInline math="PA = LU" /> reconstruction parity.</span>
    ),
    route: '/linear-algebra',
  },
  {
    name: 'Jacobi Eigenvalue Diagonalization',
    category: 'lab',
    categoryLabel: 'Demonstrated in educational math lab',
    meaning: 'Iterative orthogonal plane rotations eliminating off-diagonal elements in symmetric matrices.',
    relevance: (
      <span>Demonstrates how covariance matrix <MathInline math="S" /> is diagonalised into <MathInline math="V \Lambda V^\mathsf{T}" /> from first principles.</span>
    ),
    whereUsed: 'airsense.math_core.jacobi_eigenvalues_symmetric, Linear Algebra Lab Step 6.',
    whatToShow: (
      <span>Show the Jacobi spectrum in Linear Algebra Lab Step 6 matching SciPy eigh eigenvalues within <MathInline math="10^{-14}" />.</span>
    ),
    route: '/linear-algebra',
  },
];

interface VivaQA {
  question: string;
  shortAnswer: React.ReactNode;
  detailedPoints: React.ReactNode[];
  caveat: React.ReactNode;
}

const VIVA_QA_LIST: VivaQA[] = [
  {
    question: 'Why did you select the UCI Air Quality dataset for this project?',
    shortAnswer: (
      <span>It contains 13 continuous months of co-located, hourly multi-sensor array measurements paired with certified continuous gas analyzer reference data, providing real ground-truth without simulations.</span>
    ),
    detailedPoints: [
      'Recorded in an Italian urban roadside setting from March 2004 to April 2005 by CNR-ISAC (De Vito et al.).',
      'Pairs 5 low-cost metal-oxide chemical sensors (CO, NMHC, NOx, NO2, O3) with 3 meteorological sensors (T, RH, AH).',
      (
        <span key="p3">
          Provides true continuous chemical ground-truth (<MathInline math="\mathrm{C}_6\mathrm{H}_6" /> in <MathInline math="\mu\mathrm{g/m}^3" />) from a certified reference analyzer, enabling rigorous calibration and drift analysis.
        </span>
      ),
    ],
    caveat:
      'The dataset is historical, recorded at a single fixed roadside location; findings cannot be assumed to generalize directly to other sensor chemistries or climate zones.',
  },
  {
    question: 'What do the rows and columns of your matrix X represent?',
    shortAnswer: (
      <span>Each row represents one valid hourly observation at a specific timestamp; each column represents an eligible predictor feature in design matrix <MathInline math="X" />.</span>
    ),
    detailedPoints: [
      (
        <span key="p1">
          Matrix dimensions: <MathInline math="X \in \mathbb{R}^{8991 \times 8}" />, where <MathInline math="n = 8{,}991" /> is the number of valid supervised hours and <MathInline math="d = 8" /> is the feature count.
        </span>
      ),
      'Columns 1–5: Solid-state metal-oxide sensor resistance responses (PT08.S1 to PT08.S5).',
      'Columns 6–8: Environmental variables: Temperature (°C), Relative Humidity (%), and Absolute Humidity (g/m³).',
      (
        <span key="p4">
          The target vector <MathInline math="\mathbf{y} \in \mathbb{R}^{8991 \times 1}" /> contains the corresponding reference benzene measurement.
        </span>
      ),
    ],
    caveat: (
      <span>Standardized matrix <MathInline math="Z" /> has dimensionless z-scores; raw matrix <MathInline math="X" /> preserves original physical units (mV, °C, %, g/m³).</span>
    ),
  },
  {
    question: 'Why is orthogonalisation (QR factorisation) preferred over solving normal equations directly?',
    shortAnswer: (
      <span>Solving normal equations <MathInline math="X^\mathsf{T} X \beta = X^\mathsf{T} \mathbf{y}" /> squares the condition number <MathInline math="\kappa(X^\mathsf{T} X) = \kappa(X)^2" />, causing catastrophic cancellation; QR factorisation operates directly on <MathInline math="X" /> without squaring.</span>
    ),
    detailedPoints: [
      (
        <span key="p1">
          In our sensor matrix, <MathInline math="\kappa(X) \approx 1.89 \times 10^4" />. Normal equations square this to <MathInline math="\kappa(X^\mathsf{T} X) \approx 3.58 \times 10^8" />, exhausting IEEE 754 precision.
        </span>
      ),
      (
        <span key="p2">
          Modified Gram-Schmidt computes <MathInline math="X = QR" />, reducing the problem to upper-triangular back-substitution: <MathInline math="R \beta = Q^\mathsf{T} \mathbf{y}" />.
        </span>
      ),
      (
        <span key="p3">
          Our implementation confirms that Modified GS achieves <MathInline math="7.07 \times 10^6\times" /> lower cancellation error than Classical GS (<MathInline math="\|Q^\mathsf{T} Q - I_k\|_F" />).
        </span>
      ),
    ],
    caveat:
      'While MGS QR is numerically stable, Householder reflectors or SVD are theoretically even more orthogonal, though MGS is mathematically transparent for academic demonstration.',
  },
  {
    question: 'What is the fundamental difference between PCA and Linear Regression?',
    shortAnswer:
      'PCA is an unsupervised variance-maximization technique that ignores the target; Linear Regression is a supervised method that explicitly minimizes target prediction residuals.',
    detailedPoints: [
      (
        <span key="p1">
          PCA decomposes the feature matrix <MathInline math="X_c = U \Sigma V^\mathsf{T}" /> to find orthogonal directions capturing maximum feature variance.
        </span>
      ),
      (
        <span key="p2">
          Regression finds coefficients <MathInline math="\beta" /> minimizing <MathInline math="\|\mathbf{y} - X \beta\|_2^2" /> along the direction of strongest correlation with <MathInline math="\mathbf{y}" />.
        </span>
      ),
      (
        <span key="p3">
          Principal Component Regression (PCR) combines both: it projects <MathInline math="X" /> into the top <MathInline math="k" /> principal components, then performs least squares on the orthogonal scores.
        </span>
      ),
    ],
    caveat:
      'Retaining more variance in PCA does not guarantee higher calibration accuracy, because noise features can carry high variance.',
  },
  {
    question: 'Why do you need separate training, validation, and held-out test partitions?',
    shortAnswer: (
      <span>Training fits model parameters; validation tunes hyperparameters (selecting component count <MathInline math="k^*" />); held-out testing evaluates genuine out-of-time generalization.</span>
    ),
    detailedPoints: [
      (
        <span key="p1">
          If you tune <MathInline math="k" /> on the test set, your test performance is biased by data snooping.
        </span>
      ),
      (
        <span key="p2">
          In AirSense, the validation rule selected Full OLS (<MathInline math="k^* = 8" />, Val RMSE = 2.024 <MathInline math="\mu\mathrm{g/m}^3" />).
        </span>
      ),
      (
        <span key="p3">
          When evaluated on the held-out test partition 1 year later, Full OLS degraded to 2.855 <MathInline math="\mu\mathrm{g/m}^3" />.
        </span>
      ),
      (
        <span key="p4">
          The lower test error achieved by PCR (<MathInline math="k = 3" />, 2.415 <MathInline math="\mu\mathrm{g/m}^3" />) is an exploratory post-hoc finding, not the validation-selected model.
        </span>
      ),
    ],
    caveat:
      'Evaluating models across time requires contiguous chronological splits, not random k-fold cross-validation.',
  },
  {
    question: 'How do you know your results are not caused by data leakage?',
    shortAnswer:
      'We enforce contiguous chronological boundaries, freeze all preprocessing parameters on training data, and strictly exclude reference analyzer columns.',
    detailedPoints: [
      (
        <span key="p1">
          Feature standardizer means <MathInline math="\boldsymbol{\mu}" /> and standard deviations <MathInline math="\boldsymbol{\sigma}" /> are computed strictly on the 5,394 training rows and frozen.
        </span>
      ),
      (
        <span key="p2">
          Temporal boundaries enforce <MathInline math="\max(t_{\mathrm{train}}) < \min(t_{\mathrm{val}}) < \max(t_{\mathrm{val}}) < \min(t_{\mathrm{test}})" /> with zero overlap.
        </span>
      ),
      (
        <span key="p3">
          All certified reference gas analyzer columns (CO(GT), NMHC(GT), NOx(GT), NO2(GT)) are excluded from predictor matrix <MathInline math="X" />.
        </span>
      ),
    ],
    caveat:
      'Because the data is a single time-series at one geographic location, unmodelled seasonal weather cycles still confound multi-season predictions.',
  },
  {
    question: 'What does the Durbin-Watson statistic (DW = 0.2062) tell you about your calibration errors?',
    shortAnswer: (
      <span>It confirms severe positive first-order autocorrelation (<MathInline math="\rho \approx 0.90" />) in calibration residuals, violating the classical independent-and-identically-distributed (i.i.d.) assumption.</span>
    ),
    detailedPoints: [
      'The Durbin-Watson statistic ranges from 0 to 4; a value of 2.0 indicates zero autocorrelation.',
      (
        <span key="p2">
          <MathInline math="DW = 0.2062" /> indicates that consecutive hourly errors are strongly correlated: <MathInline math="e_t \approx 0.90 e_{t-1} + \varepsilon_t" />.
        </span>
      ),
      (
        <span key="p3">
          Under high autocorrelation, classical OLS standard errors severely underestimate parameter uncertainty by approximately <MathInline math="\sqrt{\frac{1+\rho}{1-\rho}} \approx 4.29\times" />.
        </span>
      ),
    ],
    caveat:
      'P-values and confidence intervals computed under standard Gaussian assumptions collapse toward zero and must be interpreted with extreme caution.',
  },
  {
    question: 'Can you claim that your analysis has proven physical sensor aging/degradation?',
    shortAnswer: (
      <span>No. Severe sensor distribution shifts (Cohen&apos;s <MathInline math="d = -1.87" /> on PT08.S4) are confounded by a simultaneous seasonal temperature drop from 23.1°C to 10.0°C (<MathInline math="d = -1.87" />).</span>
    ),
    detailedPoints: [
      'Metal-oxide semiconductor electrical conductivity is fundamentally temperature- and humidity-dependent.',
      'Outdoor temperature dropped from 23°C in summer to 10°C in winter, precisely matching the sensor resistance drop.',
      'Without periodic zero-gas and span-gas recalibration in a controlled chamber, physical catalyst aging cannot be mathematically isolated from climate confounding.',
    ],
    caveat:
      'Academic integrity requires acknowledging this attribution dilemma rather than falsely claiming proven hardware failure.',
  },
];

export const ProjectWalkthroughPage: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'walkthrough' | 'concepts' | 'viva'>('walkthrough');
  const [currentStep, setCurrentStep] = useState<number>(1);
  const [conceptFilter, setConceptFilter] = useState<'all' | 'pipeline' | 'analysis' | 'lab'>('all');
  const [expandedQA, setExpandedQA] = useState<number | null>(0);

  const activeStepData = WALKTHROUGH_STEPS[currentStep - 1];

  const filteredConcepts =
    conceptFilter === 'all'
      ? CONCEPT_MAP
      : CONCEPT_MAP.filter((c) => c.category === conceptFilter);

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* 1. Header Banner */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="text-[11px] font-bold text-teal-700 tracking-wider uppercase font-mono bg-teal-50 px-2.5 py-0.5 rounded-full border border-teal-200">
              Guided University Walkthrough
            </span>
            <span className="text-slate-300">&middot;</span>
            <span className="text-xs text-slate-500 font-medium">Viva Defense &amp; Academic Presentation</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 font-sans">
            AirSense Project Walkthrough &amp; Viva Defense
          </h1>
          <p className="text-sm text-slate-600 mt-1 max-w-3xl leading-relaxed">
            A structured, 8-step academic presentation designed to walk a professor through the research problem,
            real data matrix representation, linear algebra methods, calibration results, and limitations.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <Link
            to="/dataset"
            className="flex items-center gap-2 px-3.5 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-semibold transition-colors shadow-sm"
          >
            <Database className="w-3.5 h-3.5 text-teal-400" />
            <span>Inspect Real Matrix</span>
          </Link>
        </div>
      </div>

      {/* 2. Top Navigation Tabs */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-2 shadow-xs">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <button
            onClick={() => setActiveTab('walkthrough')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'walkthrough'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <BookOpen className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">1. 8-Step Guided Story</span>
              <span className={`text-[10px] ${activeTab === 'walkthrough' ? 'text-slate-300' : 'text-slate-500'}`}>
                Sequential research narrative for professor
              </span>
            </div>
          </button>

          <button
            onClick={() => setActiveTab('concepts')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'concepts'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <Binary className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">2. Concept-to-Code Explainer</span>
              <span className={`text-[10px] ${activeTab === 'concepts' ? 'text-slate-300' : 'text-slate-500'}`}>
                Where each mathematical concept is used
              </span>
            </div>
          </button>

          <button
            onClick={() => setActiveTab('viva')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'viva'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <Mic className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">3. Viva Scripts &amp; Q&amp;A</span>
              <span className={`text-[10px] ${activeTab === 'viva' ? 'text-slate-300' : 'text-slate-500'}`}>
                60s pitch, 3min speech &amp; examiner Q&amp;As
              </span>
            </div>
          </button>
        </div>
      </div>

      {/* 3. Tab Contents */}
      <AnimatePresence mode="wait">
        {/* =============================================================== */}
        {/* TAB 1: 8-STEP GUIDED WALKTHROUGH */}
        {/* =============================================================== */}
        {activeTab === 'walkthrough' && (
          <motion.div
            key="walkthrough_tab"
            variants={tabContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-6"
          >
            {/* Step Progress Ribbon */}
            <div className="bg-white border border-slate-200/90 rounded-2xl p-4 shadow-xs">
              <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-8 gap-2">
                {WALKTHROUGH_STEPS.map((s) => (
                  <button
                    key={s.step}
                    onClick={() => setCurrentStep(s.step)}
                    className={`p-2 rounded-xl text-left transition-all ${
                      currentStep === s.step
                        ? 'bg-slate-900 text-white shadow-xs'
                        : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
                    }`}
                  >
                    <div className="text-[10px] font-mono font-bold uppercase tracking-wider block">
                      {currentStep === s.step ? (
                        <span className="text-teal-400">Step {s.step}</span>
                      ) : (
                        <span className="text-slate-400">Step {s.step}</span>
                      )}
                    </div>
                    <div className="text-xs font-semibold truncate mt-0.5">{s.shortTitle.split('. ')[1]}</div>
                  </button>
                ))}
              </div>
            </div>

            {/* Active Step Card */}
            <Card
              title={activeStepData.title}
              subtitle={activeStepData.subtitle}
              action={
                <Link
                  to={activeStepData.route}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-teal-50 hover:bg-teal-100 text-teal-800 border border-teal-200 rounded-xl text-xs font-semibold transition-colors"
                >
                  <span>{activeStepData.routeLabel}</span>
                  <ExternalLink className="w-3.5 h-3.5 text-teal-600" />
                </Link>
              }
            >
              <div className="space-y-6">
                {/* Academic Context Summary */}
                <div className="p-4 bg-slate-50 border border-slate-200/90 rounded-xl space-y-2 text-xs leading-relaxed text-slate-700">
                  <div className="flex items-center gap-2 font-bold text-slate-900">
                    <Sparkles className="w-4 h-4 text-teal-600" />
                    <span>Academic Context &amp; Explanation</span>
                  </div>
                  <p>{activeStepData.problemSummary}</p>
                </div>

                {/* Concept -> Purpose -> AirSense Usage -> Output -> Interpretation Framework */}
                <div className="space-y-3">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-500 font-mono">
                    Structured Defense Framework (Concept &rarr; Purpose &rarr; Outcome)
                  </h4>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-sans">
                    <div className="p-3.5 bg-white border border-slate-200 rounded-xl space-y-1">
                      <span className="text-[10px] font-mono uppercase tracking-wider text-slate-400 font-bold block">
                        1. Theoretical Concept
                      </span>
                      <p className="font-semibold text-slate-900">{activeStepData.triad.concept}</p>
                    </div>

                    <div className="p-3.5 bg-white border border-slate-200 rounded-xl space-y-1">
                      <span className="text-[10px] font-mono uppercase tracking-wider text-teal-600 font-bold block">
                        2. Purpose in Project
                      </span>
                      <p className="text-slate-700">{activeStepData.triad.purpose}</p>
                    </div>

                    <div className="p-3.5 bg-white border border-slate-200 rounded-xl space-y-1">
                      <span className="text-[10px] font-mono uppercase tracking-wider text-indigo-600 font-bold block">
                        3. How AirSense Uses It
                      </span>
                      <p className="text-slate-700">{activeStepData.triad.airsenseUsage}</p>
                    </div>

                    <div className="p-3.5 bg-white border border-slate-200 rounded-xl space-y-1">
                      <span className="text-[10px] font-mono uppercase tracking-wider text-emerald-600 font-bold block">
                        4. Actual Verified Output
                      </span>
                      <p className="font-mono text-slate-800 font-semibold">{activeStepData.triad.actualOutput}</p>
                    </div>
                  </div>

                  <div className="p-3.5 bg-teal-50 border border-teal-200 rounded-xl text-xs text-teal-950 font-sans space-y-1">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-teal-800 font-bold block">
                      5. Scientific Interpretation
                    </span>
                    <p>{activeStepData.triad.interpretation}</p>
                  </div>
                </div>

                {/* Key Takeaway Alert Box */}
                <div className="p-3.5 bg-slate-900 text-white rounded-xl text-xs flex items-center justify-between gap-4 font-mono">
                  <div className="flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-teal-400 shrink-0" />
                    <span>Takeaway: {activeStepData.keyTakeaway}</span>
                  </div>
                  <Link
                    to={activeStepData.route}
                    className="text-teal-400 hover:text-teal-300 text-xs font-semibold shrink-0 flex items-center gap-1 font-sans"
                  >
                    Open Live Tool <ArrowRight className="w-3.5 h-3.5" />
                  </Link>
                </div>

                {/* Step Stepper Navigation Buttons */}
                <div className="flex items-center justify-between pt-4 border-t border-slate-100">
                  <button
                    onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
                    disabled={currentStep === 1}
                    className="inline-flex items-center gap-2 px-4 py-2 border border-slate-200 rounded-xl text-xs font-semibold text-slate-700 hover:bg-slate-50 disabled:opacity-40 transition-colors"
                  >
                    <ArrowLeft className="w-3.5 h-3.5" />
                    <span>Previous Step</span>
                  </button>

                  <span className="text-xs font-mono text-slate-400">
                    Step {currentStep} of {WALKTHROUGH_STEPS.length}
                  </span>

                  <button
                    onClick={() => setCurrentStep(Math.min(WALKTHROUGH_STEPS.length, currentStep + 1))}
                    disabled={currentStep === WALKTHROUGH_STEPS.length}
                    className="inline-flex items-center gap-2 px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-semibold disabled:opacity-40 transition-colors shadow-xs"
                  >
                    <span>Next Step</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>
              </div>
            </Card>
          </motion.div>
        )}

        {/* =============================================================== */}
        {/* TAB 2: CONCEPT-TO-IMPLEMENTATION MAP */}
        {/* =============================================================== */}
        {activeTab === 'concepts' && (
          <motion.div
            key="concepts_tab"
            variants={tabContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-6"
          >
            {/* Header & Filter Controls */}
            <Card
              title="Mathematical Concepts to AirSense Implementation Map"
              subtitle="Explicitly connecting university mathematics syllabus concepts to real code, outputs, and visualisations"
            >
              <div className="space-y-4">
                {/* Filter Pills */}
                <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-slate-50 border border-slate-200/80 rounded-xl text-xs">
                  <span className="font-semibold text-slate-700">Filter by Role in Project:</span>
                  <div className="flex flex-wrap items-center gap-2">
                    <button
                      onClick={() => setConceptFilter('all')}
                      className={`px-3 py-1 rounded-lg font-medium text-xs transition-colors ${
                        conceptFilter === 'all'
                          ? 'bg-slate-900 text-white'
                          : 'bg-white text-slate-600 hover:bg-slate-100 border border-slate-200'
                      }`}
                    >
                      All Concepts (12)
                    </button>
                    <button
                      onClick={() => setConceptFilter('pipeline')}
                      className={`px-3 py-1 rounded-lg font-medium text-xs transition-colors ${
                        conceptFilter === 'pipeline'
                          ? 'bg-teal-700 text-white'
                          : 'bg-white text-teal-800 hover:bg-teal-50 border border-teal-200'
                      }`}
                    >
                      Used Directly in Pipeline (5)
                    </button>
                    <button
                      onClick={() => setConceptFilter('analysis')}
                      className={`px-3 py-1 rounded-lg font-medium text-xs transition-colors ${
                        conceptFilter === 'analysis'
                          ? 'bg-amber-700 text-white'
                          : 'bg-white text-amber-800 hover:bg-amber-50 border border-amber-200'
                      }`}
                    >
                      Used for Analysis &amp; Diagnostics (4)
                    </button>
                    <button
                      onClick={() => setConceptFilter('lab')}
                      className={`px-3 py-1 rounded-lg font-medium text-xs transition-colors ${
                        conceptFilter === 'lab'
                          ? 'bg-indigo-700 text-white'
                          : 'bg-white text-indigo-800 hover:bg-indigo-50 border border-indigo-200'
                      }`}
                    >
                      Demonstrated in Math Lab (3)
                    </button>
                  </div>
                </div>

                {/* Concept Cards Grid */}
                <div className="space-y-4">
                  {filteredConcepts.map((item) => (
                    <div
                      key={item.name}
                      className="p-5 bg-white border border-slate-200/90 rounded-2xl shadow-xs space-y-3 hover:border-slate-300 transition-colors"
                    >
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-2.5">
                        <div className="flex items-center gap-2 flex-wrap">
                          <h3 className="font-bold text-sm text-slate-900 font-sans">{item.name}</h3>
                          <span
                            className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                              item.category === 'pipeline'
                                ? 'bg-teal-100 text-teal-800'
                                : item.category === 'analysis'
                                ? 'bg-amber-100 text-amber-800'
                                : 'bg-indigo-100 text-indigo-800'
                            }`}
                          >
                            {item.categoryLabel}
                          </span>
                        </div>

                        <Link
                          to={item.route}
                          className="inline-flex items-center gap-1 text-xs font-semibold text-teal-700 hover:text-teal-900"
                        >
                          <span>Open Workspace</span>
                          <ArrowRight className="w-3.5 h-3.5" />
                        </Link>
                      </div>

                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
                        <div className="space-y-1">
                          <span className="text-[10px] uppercase font-mono font-bold text-slate-400 block">
                            1. What Does It Mean?
                          </span>
                          <p className="text-slate-700 leading-relaxed">{item.meaning}</p>
                        </div>

                        <div className="space-y-1">
                          <span className="text-[10px] uppercase font-mono font-bold text-teal-700 block">
                            2. Why Is It Relevant?
                          </span>
                          <p className="text-slate-700 leading-relaxed">{item.relevance}</p>
                        </div>

                        <div className="space-y-1">
                          <span className="text-[10px] uppercase font-mono font-bold text-indigo-700 block">
                            3. Where Is It Used in AirSense?
                          </span>
                          <p className="text-slate-700 leading-relaxed font-mono text-[11px]">{item.whereUsed}</p>
                        </div>

                        <div className="space-y-1 bg-slate-50 p-3 rounded-xl border border-slate-200/80">
                          <span className="text-[10px] uppercase font-mono font-bold text-emerald-800 block">
                            4. What Result to Show My Professor:
                          </span>
                          <p className="text-slate-800 font-semibold leading-relaxed">{item.whatToShow}</p>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </Card>
          </motion.div>
        )}

        {/* =============================================================== */}
        {/* TAB 3: VIVA SPOKEN SCRIPTS & EXAMINER Q&A */}
        {/* =============================================================== */}
        {activeTab === 'viva' && (
          <motion.div
            key="viva_tab"
            variants={tabContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-6"
          >
            {/* Spoken Presentation Scripts */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* 60-Second Pitch */}
              <Card
                title="60-Second Project Introduction (Elevator Pitch)"
                subtitle="Clear, concise viva opening statement formatted for natural spoken delivery"
              >
                <div className="space-y-4 text-xs leading-relaxed text-slate-700">
                  <div className="flex items-center gap-2 text-slate-500 font-mono text-[11px] border-b border-slate-100 pb-2">
                    <Clock className="w-3.5 h-3.5 text-teal-600" />
                    <span>Duration: Exactly 60 seconds (150 words)</span>
                  </div>

                  <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-3 italic font-serif text-slate-800 text-sm leading-relaxed">
                    <p>
                      &ldquo;Good morning, Professor. My project is <strong>AirSense</strong>. We investigate how
                      multi-channel chemical sensor arrays can be calibrated using linear algebra and Principal Component
                      Analysis to monitor atmospheric benzene concentrations.&rdquo;
                    </p>
                    <p>
                      &ldquo;Using 13 continuous months of real urban roadside records from the UCI repository, we represent 8,991
                      hourly observations as a design matrix <strong className="not-italic"><MathInline math="X \in \mathbb{R}^{8991 \times 8}" /></strong> and target vector <strong className="not-italic"><MathInline math="\mathbf{y} \in \mathbb{R}^{8991 \times 1}" /></strong>.&rdquo;
                    </p>
                    <p>
                      &ldquo;Rather than relying on black boxes, we implement our own Modified Gram-Schmidt QR, Jacobi spectral
                      diagonalization, and SVD solvers to prevent condition number squaring. Under strict chronological zero-leakage splits,
                      our validation protocol selects Full OLS with 2.02 <span className="not-italic"><MathInline math="\mu\mathrm{g/m}^3" /></span> validation RMSE. However, over a 1-year test horizon,
                      we observe that compact PCR with <span className="not-italic"><MathInline math="k = 3" /></span> components acts as shrinkage regularization against sensor drift.&rdquo;
                    </p>
                    <p>
                      &ldquo;Finally, we demonstrate that apparent sensor drift is heavily confounded by seasonal winter cooling,
                      illustrating the difference between statistical significance and physical causality.&rdquo;
                    </p>
                  </div>
                </div>
              </Card>

              {/* 3-Minute Walkthrough Speech */}
              <Card
                title="3-Minute Comprehensive Walkthrough"
                subtitle="Structured technical narrative covering dataset, methods, results and caveats"
              >
                <div className="space-y-4 text-xs leading-relaxed text-slate-700">
                  <div className="flex items-center gap-2 text-slate-500 font-mono text-[11px] border-b border-slate-100 pb-2">
                    <Clock className="w-3.5 h-3.5 text-teal-600" />
                    <span>Duration: ~3 minutes (450 words)</span>
                  </div>

                  <div className="space-y-2.5 font-sans">
                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                      <strong className="text-slate-900 block font-mono text-[11px]">1. Problem &amp; Data Ingestion:</strong>
                      Certified benzene gas analyzers cost over €25,000, while solid-state metal-oxide sensors cost under €20. We take 9,357
                      raw records, remove 366 instrument blackout hours, and construct a clean supervised matrix of 8,991 rows and 8 predictor columns.
                    </div>

                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                      <strong className="text-slate-900 block font-mono text-[11px]">2. Linear Algebra Primitives:</strong>
                      Our feature matrix has condition number <MathInline math="\kappa(X) \approx 1.89 \times 10^4" />. Solving normal equations <MathInline math="X^\mathsf{T} X \beta = X^\mathsf{T} \mathbf{y}" /> directly squares this
                      to <MathInline math="3.58 \times 10^8" />, threatening catastrophic floating-point cancellation. We implement Modified Gram-Schmidt QR, which solves <MathInline math="R \beta = Q^\mathsf{T} \mathbf{y}" /> while preserving the original condition number.
                    </div>

                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                      <strong className="text-slate-900 block font-mono text-[11px]">3. Validation Tuning vs Out-of-Time Test:</strong>
                      We partition the 8,991 hours chronologically into 60% Train, 20% Val, and 20% Held-Out Test. Hyperparameter tuning was
                      conducted strictly on validation data, selecting Full OLS (<MathInline math="k^* = 8" />) with Val RMSE = 2.024 <MathInline math="\mu\mathrm{g/m}^3" />.
                    </div>

                    <div className="p-3 bg-slate-50 rounded-lg border border-slate-200">
                      <strong className="text-slate-900 block font-mono text-[11px]">4. The Drift &amp; Attribution Dilemma:</strong>
                      On the untouched test set 1 year later, Full OLS degrades to RMSE = 2.855 <MathInline math="\mu\mathrm{g/m}^3" />. In post-hoc analysis, compact PCR (<MathInline math="k = 3" />) achieves
                      2.415 <MathInline math="\mu\mathrm{g/m}^3" />. However, we do not claim proven sensor degradation: PT08.S4 dropped by -499 units (<MathInline math="d = -1.87" />), but outdoor temperature
                      simultaneously dropped from 23°C to 10°C (<MathInline math="d = -1.87" />).
                    </div>
                  </div>
                </div>
              </Card>
            </div>

            {/* Anticipated Examiner Questions & Answers */}
            <Card
              title="Anticipated Professor / Examiner Viva Questions & Answers"
              subtitle="Concrete, mathematically grounded answers to likely oral examination challenges"
            >
              <div className="space-y-3">
                {VIVA_QA_LIST.map((qa, idx) => {
                  const isExpanded = expandedQA === idx;

                  return (
                    <div
                      key={idx}
                      className="border border-slate-200/90 rounded-xl overflow-hidden bg-white shadow-2xs transition-colors"
                    >
                      <button
                        onClick={() => setExpandedQA(isExpanded ? null : idx)}
                        className="w-full p-4 text-left flex items-center justify-between gap-4 hover:bg-slate-50/70 transition-colors"
                      >
                        <span className="font-bold text-xs text-slate-900 font-sans flex items-center gap-2">
                          <HelpCircle className="w-4 h-4 text-teal-600 shrink-0" />
                          <span>{qa.question}</span>
                        </span>
                        {isExpanded ? (
                          <ChevronUp className="w-4 h-4 text-slate-400 shrink-0" />
                        ) : (
                          <ChevronDown className="w-4 h-4 text-slate-400 shrink-0" />
                        )}
                      </button>

                      {isExpanded && (
                        <div className="p-4 pt-0 border-t border-slate-100 space-y-3 text-xs text-slate-700 leading-relaxed font-sans bg-slate-50/30">
                          <div className="p-3 bg-teal-50/70 border border-teal-200 rounded-lg">
                            <strong className="text-teal-950 block font-mono text-[11px] mb-1">
                              Concise Answer to Speak:
                            </strong>
                            <p className="text-teal-900">{qa.shortAnswer}</p>
                          </div>

                          <div className="space-y-1.5 pl-2">
                            <strong className="text-slate-900 block font-mono text-[11px]">Key Technical Points:</strong>
                            <ul className="list-disc pl-4 space-y-1 text-slate-600">
                              {qa.detailedPoints.map((pt, pIdx) => (
                                <li key={pIdx}>{pt}</li>
                              ))}
                            </ul>
                          </div>

                          <div className="p-2.5 bg-amber-50 border border-amber-200 rounded-lg text-amber-950 text-[11px]">
                            <strong>Academic Examiner Nuance / Caveat:</strong> {qa.caveat}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
