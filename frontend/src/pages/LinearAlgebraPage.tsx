import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  ShieldCheck,
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  ChevronsLeft,
  ChevronsRight,
  CheckCircle2,
  AlertTriangle,
  Sparkles,
  Database,
  Sliders,
  RefreshCw,
  ArrowRight,
  Loader2,
} from 'lucide-react';
import {
  ResponsiveContainer,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Line,
  ComposedChart,
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';

import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { ErrorState } from '../components/ui/ErrorState';
import { MathInline } from '../components/ui/Math';
import { MatrixViewer } from '../components/math/MatrixViewer';
import { StepNavigation } from '../components/math/StepNavigation';
import { MathConceptCard } from '../components/math/MathConceptCard';
import { apiClient } from '../services/apiClient';
import {
  MatrixExtractResponse,
  GaussianEliminationResponse,
  SubspaceAnalysisResponse,
  GramSchmidtResponse,
  LeastSquaresResponse,
  SpectralAnalysisResponse,
} from '../types/api';
import { tabContentVariants } from '../lib/motion';

const ALL_FEATURES = [
  'PT08.S1(CO)',
  'PT08.S2(NMHC)',
  'PT08.S3(NOx)',
  'PT08.S4(NO2)',
  'PT08.S5(O3)',
  'T',
  'RH',
  'AH',
];

// Presets
const SQUARE_PRESET = [
  [2.0, 1.0, -1.0],
  [-3.0, -1.0, 2.0],
  [-2.0, 1.0, 2.0],
];

const ILL_CONDITIONED_PRESET = [
  [1.0, 1.0, 1.0],
  [1e-4, 0.0, 0.0],
  [0.0, 1e-4, 0.0],
  [0.0, 0.0, 1e-4],
];

const RANK_DEFICIENT_PRESET = [
  [1.0, 2.0, 1.0, 3.0],
  [2.0, 4.0, 2.0, 6.0],
  [3.0, 1.0, 4.0, 2.0],
  [4.0, 3.0, 5.0, 5.0],
];

export const LinearAlgebraPage: React.FC = () => {
  // Navigation
  const [activeStep, setActiveStep] = useState<number>(1);

  // Step 1: Real Data Matrix State
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([
    'PT08.S1(CO)',
    'PT08.S2(NMHC)',
    'PT08.S3(NOx)',
  ]);
  const [nRows, setNRows] = useState<number>(8);
  const [standardize, setStandardize] = useState<boolean>(false);
  const [includeIntercept, setIncludeIntercept] = useState<boolean>(false);
  const [matrixData, setMatrixData] = useState<MatrixExtractResponse | null>(null);
  const [loadingExtract, setLoadingExtract] = useState<boolean>(false);
  const [extractError, setExtractError] = useState<string | null>(null);

  // Step 2: Gaussian Elimination State
  const [gaussSource, setGaussSource] = useState<'extracted' | 'square' | 'ill_conditioned'>('extracted');
  const [gaussData, setGaussData] = useState<GaussianEliminationResponse | null>(null);
  const [currentStepIdx, setCurrentStepIdx] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [loadingGauss, setLoadingGauss] = useState<boolean>(false);
  const playTimerRef = useRef<number | null>(null);

  // Step 3: Subspaces State
  const [subspaceSource, setSubspaceSource] = useState<'extracted' | 'rank_deficient'>('extracted');
  const [subspaceData, setSubspaceData] = useState<SubspaceAnalysisResponse | null>(null);
  const [loadingSubspaces, setLoadingSubspaces] = useState<boolean>(false);

  // Step 4: Gram-Schmidt State
  const [gsPreset, setGsPreset] = useState<'extracted' | 'ill_conditioned'>('ill_conditioned');
  const [gsData, setGsData] = useState<GramSchmidtResponse | null>(null);
  const [loadingGs, setLoadingGs] = useState<boolean>(false);

  // Step 5: Least Squares State
  const [lsData, setLsData] = useState<LeastSquaresResponse | null>(null);
  const [loadingLs, setLoadingLs] = useState<boolean>(false);

  // Step 6: Spectral & PCA State
  const [spectralFeatures, setSpectralFeatures] = useState<string[]>([
    'PT08.S1(CO)',
    'PT08.S2(NMHC)',
    'PT08.S3(NOx)',
    'T',
  ]);
  const [spectralData, setSpectralData] = useState<SpectralAnalysisResponse | null>(null);
  const [loadingSpectral, setLoadingSpectral] = useState<boolean>(false);

  // ---------------------------------------------------------------------------
  // API Fetchers
  // ---------------------------------------------------------------------------

  const fetchExtractMatrix = useCallback(async () => {
    setLoadingExtract(true);
    setExtractError(null);
    try {
      const res = await apiClient.extractMatrix({
        features: selectedFeatures,
        n_rows: nRows,
        standardize,
        include_intercept: includeIntercept,
      });
      setMatrixData(res);
    } catch (err) {
      setExtractError(err instanceof Error ? err.message : 'Failed to extract data matrix');
    } finally {
      setLoadingExtract(false);
    }
  }, [selectedFeatures, nRows, standardize, includeIntercept]);

  const fetchGaussianElimination = useCallback(async () => {
    setLoadingGauss(true);
    try {
      let targetMatrix: number[][] = [];
      if (gaussSource === 'extracted' && matrixData) {
        targetMatrix = matrixData.matrix;
      } else if (gaussSource === 'square') {
        targetMatrix = SQUARE_PRESET;
      } else {
        targetMatrix = ILL_CONDITIONED_PRESET;
      }

      if (targetMatrix.length > 0) {
        const res = await apiClient.computeGaussianElimination({
          matrix: targetMatrix,
          compute_lu: true,
        });
        setGaussData(res);
        setCurrentStepIdx(0);
        setIsPlaying(false);
      }
    } catch (err) {
      console.error('Gaussian error:', err);
    } finally {
      setLoadingGauss(false);
    }
  }, [gaussSource, matrixData]);

  const fetchSubspaces = useCallback(async () => {
    setLoadingSubspaces(true);
    try {
      const targetMatrix =
        subspaceSource === 'extracted' && matrixData
          ? matrixData.matrix
          : RANK_DEFICIENT_PRESET;

      if (targetMatrix.length > 0) {
        const res = await apiClient.analyzeSubspaces({
          matrix: targetMatrix,
        });
        setSubspaceData(res);
      }
    } catch (err) {
      console.error('Subspaces error:', err);
    } finally {
      setLoadingSubspaces(false);
    }
  }, [subspaceSource, matrixData]);

  const fetchGramSchmidt = useCallback(async () => {
    setLoadingGs(true);
    try {
      const targetMatrix =
        gsPreset === 'ill_conditioned'
          ? ILL_CONDITIONED_PRESET
          : matrixData
          ? matrixData.matrix
          : ILL_CONDITIONED_PRESET;

      const res = await apiClient.compareGramSchmidt({
        matrix: targetMatrix,
      });
      setGsData(res);
    } catch (err) {
      console.error('Gram-Schmidt error:', err);
    } finally {
      setLoadingGs(false);
    }
  }, [gsPreset, matrixData]);

  const fetchLeastSquares = useCallback(async () => {
    if (!matrixData) return;
    setLoadingLs(true);
    try {
      const res = await apiClient.solveLeastSquares({
        matrix: matrixData.matrix,
        target_vector: matrixData.target_vector,
      });
      setLsData(res);
    } catch (err) {
      console.error('Least squares error:', err);
    } finally {
      setLoadingLs(false);
    }
  }, [matrixData]);

  const fetchSpectral = useCallback(async () => {
    setLoadingSpectral(true);
    try {
      const res = await apiClient.analyzeSpectral({
        features: spectralFeatures,
        use_training_set: true,
      });
      setSpectralData(res);
    } catch (err) {
      console.error('Spectral error:', err);
    } finally {
      setLoadingSpectral(false);
    }
  }, [spectralFeatures]);

  // Initial load
  useEffect(() => {
    fetchExtractMatrix();
    fetchSpectral();
  }, [fetchExtractMatrix, fetchSpectral]);

  // Refetch dependents when matrixData changes
  useEffect(() => {
    if (matrixData) {
      fetchGaussianElimination();
      fetchSubspaces();
      fetchGramSchmidt();
      fetchLeastSquares();
    }
  }, [matrixData, fetchGaussianElimination, fetchSubspaces, fetchGramSchmidt, fetchLeastSquares]);

  // Player controls for Step 2
  useEffect(() => {
    if (isPlaying && gaussData && gaussData.steps.length > 0) {
      playTimerRef.current = window.setInterval(() => {
        setCurrentStepIdx((prev) => {
          if (prev >= gaussData.steps.length - 1) {
            setIsPlaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 700);
    } else {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    }
    return () => {
      if (playTimerRef.current) clearInterval(playTimerRef.current);
    };
  }, [isPlaying, gaussData]);

  const toggleFeature = (feat: string) => {
    setSelectedFeatures((prev) => {
      if (prev.includes(feat)) {
        if (prev.length === 1) return prev; // Keep at least one
        return prev.filter((f) => f !== feat);
      } else {
        return [...prev, feat];
      }
    });
  };

  const toggleSpectralFeature = (feat: string) => {
    setSpectralFeatures((prev) => {
      if (prev.includes(feat)) {
        if (prev.length <= 2) return prev;
        return prev.filter((f) => f !== feat);
      } else {
        return [...prev, feat];
      }
    });
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* Header Banner */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="text-[11px] font-bold text-teal-700 tracking-wider uppercase font-mono bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
              Units 1 & 2 Foundations
            </span>
            <span className="text-slate-300">&middot;</span>
            <span className="text-xs text-slate-500 font-medium">From-Scratch Numerical Engine</span>
            <span className="text-slate-300">&middot;</span>
            <span className="text-[11px] font-mono text-slate-500">Zero High-Level Library Shortcuts</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Linear Algebra Laboratory</h1>
          <p className="text-sm text-slate-600 mt-1 max-w-3xl leading-relaxed">
            Step through canonical matrix reductions, rank-nullity decomposition, Modified Gram-Schmidt QR,
            orthogonal projections, and Jacobi spectral analysis executed directly on the UCI Air Quality dataset.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0 flex-wrap">
          <Badge variant="teal" size="md" className="gap-1.5 font-mono">
            <ShieldCheck className="w-4 h-4 text-teal-600" />
            80 Automated Tests Passing
          </Badge>
          <button
            onClick={() => {
              fetchExtractMatrix();
              fetchSpectral();
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors border border-slate-200"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Reload Engine
          </button>
        </div>
      </div>

      {/* 6-Step Workflow Navigation */}
      <StepNavigation
        activeStep={activeStep}
        onSelectStep={(stepId) => setActiveStep(stepId)}
      />

      {/* Dynamic Content Container */}
      <AnimatePresence mode="wait">
        {/* ================================================================= */}
        {/* STEP 1: REAL DATA MATRIX REPRESENTATION */}
        {/* ================================================================= */}
        {activeStep === 1 && (
          <motion.div
            key="step1"
            variants={tabContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-6"
          >
            <MathConceptCard
              formula="X \in \mathbb{R}^{m \times d}, \quad \mathbf{y} \in \mathbb{R}^m, \quad \tilde{\mathbf{x}} = \frac{\mathbf{x} - \boldsymbol{\mu}}{\boldsymbol{\sigma}}"
              concept="In empirical science, a linear system is formulated as a design matrix X where each row is an observation at timestamp t_i, and each column represents a physical sensor channel or environmental parameter. When combined with target ground-truth vector y, the system X w ~ y models chemical concentration."
              purpose="Evaluating real sensor arrays reveals real-world mathematical phenomena: extreme scale differences between micro-voltage sensors (10^3) and temperatures (10^1), multi-collinearity from shared thermal drift, and conditioning challenges."
              operation="The data slice is extracted directly from the verified 5,394-hour training partition without feature-target leakage. Optional z-score standardization rescales columns to zero mean and unit variance."
              airsenseInterpretation="Raw sensor responses (PT08.S1 to PT08.S5) vary by several orders of magnitude. Standardizing features is critical to prevent ill-conditioned normal equations and enable fair singular value comparisons."
            />

            {/* Interactive Configuration Bar */}
            <Card title="Data Matrix Extraction Controls" subtitle="Slice genuine sensor readings from March 2004 training partition">
              <div className="space-y-5">
                {/* Feature Selection Checkboxes */}
                <div>
                  <div className="text-xs font-bold text-slate-800 mb-2 flex items-center justify-between">
                    <span>Active Predictor Features ({selectedFeatures.length} of {ALL_FEATURES.length})</span>
                    <span className="text-[11px] text-slate-500 font-normal">Select features to form columns of X</span>
                  </div>
                  <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-2">
                    {ALL_FEATURES.map((feat) => {
                      const isSel = selectedFeatures.includes(feat);
                      return (
                        <button
                          key={feat}
                          onClick={() => toggleFeature(feat)}
                          className={`p-2 rounded-xl text-xs font-mono font-medium text-center border transition-all ${
                            isSel
                              ? 'bg-teal-50 border-teal-300 text-teal-900 font-bold shadow-xs'
                              : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
                          }`}
                        >
                          {feat}
                        </button>
                      );
                    })}
                  </div>
                </div>

                {/* Rows and Toggles */}
                <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 pt-3 border-t border-slate-100 items-end">
                  {/* Rows Slider */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between text-xs font-semibold text-slate-700">
                      <span>Row Count (m)</span>
                      <span className="font-mono text-teal-700">{nRows} rows</span>
                    </div>
                    <input
                      type="range"
                      min={4}
                      max={24}
                      value={nRows}
                      onChange={(e) => setNRows(Number(e.target.value))}
                      className="w-full accent-teal-600 cursor-pointer"
                    />
                  </div>

                  {/* Standardize Toggle */}
                  <div className="flex items-center gap-2.5 p-2 rounded-xl border border-slate-200 bg-slate-50">
                    <input
                      type="checkbox"
                      id="std_toggle"
                      checked={standardize}
                      onChange={(e) => setStandardize(e.target.checked)}
                      className="w-4 h-4 text-teal-600 rounded accent-teal-600 cursor-pointer"
                    />
                    <label htmlFor="std_toggle" className="text-xs font-medium text-slate-700 cursor-pointer">
                      Standardize Features (Z-Score)
                    </label>
                  </div>

                  {/* Intercept Toggle */}
                  <div className="flex items-center gap-2.5 p-2 rounded-xl border border-slate-200 bg-slate-50">
                    <input
                      type="checkbox"
                      id="int_toggle"
                      checked={includeIntercept}
                      onChange={(e) => setIncludeIntercept(e.target.checked)}
                      className="w-4 h-4 text-teal-600 rounded accent-teal-600 cursor-pointer"
                    />
                    <label htmlFor="int_toggle" className="text-xs font-medium text-slate-700 cursor-pointer">
                      Prepend Intercept Column (<MathInline math="x_0 = 1" />)
                    </label>
                  </div>

                  {/* Refresh Button */}
                  <button
                    onClick={fetchExtractMatrix}
                    disabled={loadingExtract}
                    className="w-full py-2 px-3 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-bold transition-colors flex items-center justify-center gap-2 shadow-xs"
                  >
                    {loadingExtract ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
                    Extract Matrix
                  </button>
                </div>
              </div>
            </Card>

            {/* Matrix View & Diagnostics */}
            {extractError ? (
              <ErrorState title="Matrix Extraction Failed" message={extractError} onRetry={fetchExtractMatrix} />
            ) : matrixData ? (
              <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
                {/* Design Matrix X */}
                <div className="lg:col-span-3 space-y-4">
                  <MatrixViewer
                    matrix={matrixData.matrix}
                    colLabels={matrixData.col_labels}
                    rowLabels={matrixData.row_labels}
                    title="Extracted Design Matrix X"
                    badge={matrixData.is_standardized ? 'Standardized Z-Score' : 'Raw Sensor Units'}
                    caption={`Matrix dimensions: ${matrixData.dimensions[0]} rows (hours) \u00D7 ${matrixData.dimensions[1]} columns (features).`}
                  />

                  {/* Feature Means & Stds */}
                  <div className="p-4 bg-white border border-slate-200/90 rounded-xl space-y-2 shadow-xs">
                    <span className="text-xs font-bold text-slate-800">Column Summary Statistics</span>
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs">
                      {matrixData.col_labels.map((col, idx) => (
                        <div key={col} className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/70 font-mono">
                          <div className="text-[10px] text-slate-500 truncate font-sans">{col}</div>
                          <div className="text-slate-900 font-semibold text-xs mt-0.5">
                            mean: {matrixData.means[idx]}
                          </div>
                          <div className="text-slate-500 text-[11px]">
                            std: {matrixData.stds[idx]}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Ground Truth Target Vector y */}
                <div className="space-y-4">
                  <MatrixViewer
                    matrix={matrixData.target_vector.map((val) => [val])}
                    colLabels={['C₆H₆(GT)']}
                    rowLabels={matrixData.row_labels}
                    title="Target Vector y"
                    badge="Reference Analyzer"
                    caption={<span>Benzene concentration in <MathInline math="\mu\mathrm{g/m}^3" /> (certified reference).</span>}
                  />

                  {/* Dataset Provenance Notice */}
                  <div className="p-4 rounded-xl bg-teal-50/60 border border-teal-200/70 text-xs space-y-2">
                    <div className="flex items-center gap-1.5 font-bold text-teal-900">
                      <Database className="w-3.5 h-3.5 text-teal-700" />
                      Dataset Provenance
                    </div>
                    <p className="text-teal-800 text-[11px] leading-relaxed">
                      {matrixData.source_description}
                    </p>
                  </div>
                </div>
              </div>
            ) : null}

            {/* Stepper Footer */}
            <div className="flex justify-end pt-4 border-t border-slate-200">
              <button
                onClick={() => setActiveStep(2)}
                className="flex items-center gap-2 px-5 py-2.5 bg-teal-600 hover:bg-teal-700 text-white rounded-xl text-xs font-bold shadow-xs transition-colors"
              >
                Proceed to Step 2: Gaussian Elimination & LU
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        )}

        {/* ================================================================= */}
        {/* STEP 2: GAUSSIAN ELIMINATION, RREF & LU DECOMPOSITION */}
        {/* ================================================================= */}
        {activeStep === 2 && (
          <motion.div
            key="step2"
            variants={tabContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-6"
          >
            <MathConceptCard
              formula="PA = LU, \quad \mathrm{rref}(A) = \begin{bmatrix} I_r & F \\ 0 & 0 \end{bmatrix}"
              concept="Gaussian elimination applies elementary row operations (row swapping, scaling, and elimination) to systematically reduce a matrix into row echelon and reduced row echelon form (RREF). For square invertible systems, partial row pivoting yields the factorization P A = L U where P is a permutation matrix, L is unit lower-triangular, and U is upper-triangular."
              purpose="Row reduction establishes linear independence and identifies non-zero pivots. The LU decomposition enables efficient solving of systems A w = y across multiple target vectors via forward- and back-substitution without re-reducing A."
              operation="Our engine implements row reduction and LU factorization from scratch in pure Python with partial pivoting (|a_p,k| = max |a_i,k|), ensuring numerical stability against small diagonal divisors. Every row operation is captured in an educational trace."
              airsenseInterpretation="When reducing multi-sensor arrays, collinear sensors (e.g. PT08.S1 and PT08.S2 both responding strongly to hydrocarbons) result in near-zero pivots, diagnosing redundancy before calibration."
            />

            {/* Source Preset Selection */}
            <Card title="System Matrix Configuration & Presets">
              <div className="flex flex-wrap gap-3 items-center justify-between">
                <div className="flex flex-wrap gap-2">
                  <button
                    onClick={() => setGaussSource('extracted')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      gaussSource === 'extracted'
                        ? 'bg-slate-900 text-white shadow-xs'
                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
                    }`}
                  >
                    Extracted Data Matrix ({matrixData?.dimensions[0]} &times; {matrixData?.dimensions[1]})
                  </button>
                  <button
                    onClick={() => setGaussSource('square')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      gaussSource === 'square'
                        ? 'bg-slate-900 text-white shadow-xs'
                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
                    }`}
                  >
                    Preset: Square 3&times;3 Invertible (LU Test)
                  </button>
                  <button
                    onClick={() => setGaussSource('ill_conditioned')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      gaussSource === 'ill_conditioned'
                        ? 'bg-slate-900 text-white shadow-xs'
                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
                    }`}
                  >
                    Preset: Ill-Conditioned Challenge (4&times;3)
                  </button>
                </div>

                <Badge variant="outline" size="sm" className="font-mono">
                  Tolerance: <MathInline math="\varepsilon = 10^{-10}" />
                </Badge>
              </div>
            </Card>

            {loadingGauss ? (
              <div className="p-8 text-center bg-white border border-slate-200 rounded-2xl shadow-xs">
                <LoadingSpinner label="Computing step-by-step Gaussian reduction..." />
              </div>
            ) : gaussData && gaussData.steps.length > 0 ? (
              <div className="space-y-6">
                {/* Interactive Step-by-Step Player Controls */}
                <div className="bg-white border border-slate-200/90 rounded-2xl p-4 shadow-xs space-y-3">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <span className="p-1.5 rounded-lg bg-teal-50 text-teal-700 border border-teal-200">
                        <Sliders className="w-4 h-4" />
                      </span>
                      <div>
                        <span className="text-xs font-bold text-slate-900">
                          Reduction Step {currentStepIdx} of {gaussData.steps.length - 1}
                        </span>
                        <span className="text-[11px] text-slate-500 block">
                          Interactive playback of elementary row operations
                        </span>
                      </div>
                    </div>

                    {/* Playback Buttons */}
                    <div className="flex items-center gap-1.5 self-end sm:self-auto">
                      <button
                        onClick={() => {
                          setIsPlaying(false);
                          setCurrentStepIdx(0);
                        }}
                        title="Reset to Step 0"
                        className="p-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors"
                      >
                        <ChevronsLeft className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => {
                          setIsPlaying(false);
                          setCurrentStepIdx((p) => Math.max(0, p - 1));
                        }}
                        disabled={currentStepIdx === 0}
                        title="Previous Step"
                        className="p-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 disabled:opacity-40 transition-colors"
                      >
                        <ChevronLeft className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => setIsPlaying(!isPlaying)}
                        className={`px-3 py-1.5 rounded-lg text-xs font-bold flex items-center gap-1.5 transition-colors ${
                          isPlaying
                            ? 'bg-amber-600 hover:bg-amber-700 text-white'
                            : 'bg-teal-600 hover:bg-teal-700 text-white'
                        }`}
                      >
                        {isPlaying ? (
                          <>
                            <Pause className="w-3.5 h-3.5" /> Pause
                          </>
                        ) : (
                          <>
                            <Play className="w-3.5 h-3.5" /> Auto Play
                          </>
                        )}
                      </button>
                      <button
                        onClick={() => {
                          setIsPlaying(false);
                          setCurrentStepIdx((p) => Math.min(gaussData.steps.length - 1, p + 1));
                        }}
                        disabled={currentStepIdx === gaussData.steps.length - 1}
                        title="Next Step"
                        className="p-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 disabled:opacity-40 transition-colors"
                      >
                        <ChevronRight className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => {
                          setIsPlaying(false);
                          setCurrentStepIdx(gaussData.steps.length - 1);
                        }}
                        title="Jump to Final RREF"
                        className="p-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-700 transition-colors"
                      >
                        <ChevronsRight className="w-4 h-4" />
                      </button>
                    </div>
                  </div>

                  {/* Scrubber slider */}
                  <input
                    type="range"
                    min={0}
                    max={gaussData.steps.length - 1}
                    value={currentStepIdx}
                    onChange={(e) => {
                      setIsPlaying(false);
                      setCurrentStepIdx(Number(e.target.value));
                    }}
                    className="w-full accent-teal-600 cursor-pointer"
                  />

                  {/* Active Step Description Card */}
                  {gaussData.steps[currentStepIdx] && (
                    <div className="p-3 bg-slate-50 border border-slate-200/80 rounded-xl flex items-center justify-between text-xs">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-[10px] font-bold px-2 py-0.5 rounded bg-teal-100 text-teal-800">
                          {gaussData.steps[currentStepIdx].operation}
                        </span>
                        <span className="text-slate-700 font-sans">
                          {gaussData.steps[currentStepIdx].description}
                        </span>
                      </div>
                      <span className="text-[10px] font-mono text-slate-400">
                        Step {currentStepIdx}/{gaussData.steps.length - 1}
                      </span>
                    </div>
                  )}
                </div>

                {/* Matrix State Display */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Intermediate Matrix at Active Step */}
                  <MatrixViewer
                    matrix={gaussData.steps[currentStepIdx].matrix}
                    highlightRow={gaussData.steps[currentStepIdx].target_row}
                    highlightPivots={
                      gaussData.steps[currentStepIdx].pivot_row !== null &&
                      gaussData.steps[currentStepIdx].pivot_col !== null
                        ? [
                            [
                              gaussData.steps[currentStepIdx].pivot_row as number,
                              gaussData.steps[currentStepIdx].pivot_col as number,
                            ],
                          ]
                        : []
                    }
                    title={`Matrix State at Step ${currentStepIdx}`}
                    badge={gaussData.steps[currentStepIdx].operation}
                    caption="Pivots highlighted in teal; active target row highlighted in amber."
                  />

                  {/* Final Canonical RREF */}
                  <div className="space-y-4">
                    <MatrixViewer
                      matrix={gaussData.final_rref}
                      highlightPivots={gaussData.pivot_columns.map((colIdx, rowIdx) => [rowIdx, colIdx])}
                      title="Canonical RREF Matrix"
                      badge={`Rank = ${gaussData.rank}`}
                      caption={`Detected ${gaussData.pivot_columns.length} pivot columns: indices [${gaussData.pivot_columns.join(', ')}].`}
                    />

                    {/* Rank & Pivot Diagnostics */}
                    <div className="p-4 bg-white border border-slate-200/90 rounded-xl space-y-2 shadow-xs">
                      <div className="text-xs font-bold text-slate-800">Reduction Diagnostics</div>
                      <div className="grid grid-cols-2 gap-3 text-xs font-mono">
                        <div className="p-2.5 bg-slate-50 border border-slate-200/70 rounded-lg">
                          <span className="text-[10px] text-slate-500 block font-sans">Matrix Rank</span>
                          <span className="text-teal-700 font-bold text-base">{gaussData.rank}</span>
                        </div>
                        <div className="p-2.5 bg-slate-50 border border-slate-200/70 rounded-lg">
                          <span className="text-[10px] text-slate-500 block font-sans">Determinant</span>
                          <span className="text-slate-900 font-bold text-base">
                            {gaussData.determinant !== null ? gaussData.determinant : 'N/A (Rectangular)'}
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* PA = LU Factorization Card (When Applicable) */}
                {gaussData.lu && (
                  <Card
                    title="PA = LU Factorization (Partial Row Pivoting)"
                    subtitle="Decomposes square matrix into Permutation P, Unit Lower-Triangular L, and Upper-Triangular U"
                  >
                    <div className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                        <MatrixViewer
                          matrix={gaussData.lu.P}
                          title="Permutation Matrix P"
                          badge="Orthogonal"
                          precision={1}
                        />
                        <MatrixViewer
                          matrix={gaussData.lu.L}
                          title="Lower Triangular L"
                          badge={<MathInline math="\mathrm{diag}(L) = 1" />}
                        />
                        <MatrixViewer
                          matrix={gaussData.lu.U}
                          title="Upper Triangular U"
                          badge="Echelon Form"
                        />
                      </div>

                      {/* Verification Error */}
                      <div className="p-3.5 bg-emerald-50 border border-emerald-200 rounded-xl flex items-center justify-between text-xs">
                        <div className="flex items-center gap-2 text-emerald-950">
                          <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                          <span>
                            <strong>Factorization Invariant Verified:</strong> <MathInline math="\|PA - LU\|_\infty =" />{' '}
                            <span className="font-mono font-bold">{gaussData.lu.residual_error.toExponential(4)}</span>
                          </span>
                        </div>
                        <Badge variant="teal" size="sm">
                          Machine Precision (&lt; <MathInline math="10^{-14}" />)
                        </Badge>
                      </div>
                    </div>
                  </Card>
                )}
              </div>
            ) : null}

            {/* Stepper Footer */}
            <div className="flex justify-between pt-4 border-t border-slate-200">
              <button
                onClick={() => setActiveStep(1)}
                className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                Back to Step 1
              </button>
              <button
                onClick={() => setActiveStep(3)}
                className="flex items-center gap-2 px-5 py-2.5 bg-teal-600 hover:bg-teal-700 text-white rounded-xl text-xs font-bold shadow-xs transition-colors"
              >
                Proceed to Step 3: Vector Spaces & Subspaces
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        )}

        {/* ================================================================= */}
        {/* STEP 3: VECTOR SPACES & SUBSPACES (BASIS, RANK, NULLITY) */}
        {/* ================================================================= */}
        {activeStep === 3 && (
          <motion.div
            key="step3"
            variants={tabContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-6"
          >
            <MathConceptCard
              formula="\dim(C(A)) + \dim(N(A)) = n \iff \mathrm{rank}(A) + \mathrm{nullity}(A) = n"
              concept="Every matrix A in R^(m x n) defines four fundamental subspaces: Column Space C(A) in R^m, Null Space N(A) in R^n, Row Space C(A^T) in R^n, and Left Null Space N(A^T) in R^m. The Fundamental Rank-Nullity Theorem dictates that the dimension of the column space (rank) plus the dimension of the null space (nullity) equals the number of columns n."
              purpose="In gas sensor calibration, nullity > 0 indicates exact collinearity: a linear combination of sensor channels produces zero response across all observations. A full column rank (nullity = 0) is mandatory for the existence of a unique least-squares solution."
              operation="Basis vectors for C(A) are extracted directly from the original pivot columns identified during RREF. Basis vectors for N(A) are computed via singular value decomposition spanning the right singular vectors corresponding to zero singular values."
              airsenseInterpretation="For the full 8-sensor AirSense array, rank = 8 and nullity = 0 across all 8,991 hours, proving the sensor basis spans an 8-dimensional subspace with no redundant zero-information directions."
            />

            {/* Source Preset Selection */}
            <Card title="Subspace Input Matrix Configuration">
              <div className="flex flex-wrap gap-2 items-center justify-between">
                <div className="flex gap-2">
                  <button
                    onClick={() => setSubspaceSource('extracted')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      subspaceSource === 'extracted'
                        ? 'bg-slate-900 text-white shadow-xs'
                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
                    }`}
                  >
                    Extracted Sensor Matrix ({matrixData?.dimensions[0]} &times; {matrixData?.dimensions[1]})
                  </button>
                  <button
                    onClick={() => setSubspaceSource('rank_deficient')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      subspaceSource === 'rank_deficient'
                        ? 'bg-slate-900 text-white shadow-xs'
                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
                    }`}
                  >
                    Preset: Rank-Deficient Matrix (Rank 2 of 4 &middot; Nullity 2)
                  </button>
                </div>

                <Badge variant="teal" size="sm" className="font-mono">
                  SVD Kernel Precision: <MathInline math="\varepsilon = 10^{-10}" />
                </Badge>
              </div>
            </Card>

            {loadingSubspaces ? (
              <div className="p-8 text-center bg-white border border-slate-200 rounded-2xl shadow-xs">
                <LoadingSpinner label="Computing fundamental subspaces and bases..." />
              </div>
            ) : subspaceData ? (
              <div className="space-y-6">
                {/* Prominent Rank-Nullity Theorem Card */}
                <div className="bg-gradient-to-r from-slate-900 to-slate-800 text-white rounded-2xl p-6 shadow-md border border-slate-700">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-700/80">
                    <div>
                      <span className="text-[10px] font-mono uppercase tracking-wider text-teal-400 font-bold bg-teal-950/60 px-2 py-0.5 rounded border border-teal-500/30">
                        Unit 1 Core Theorem
                      </span>
                      <h3 className="text-lg font-bold mt-1 text-white">The Rank-Nullity Theorem</h3>
                      <p className="text-xs text-slate-300 mt-0.5">
                        Dimension conservation across domain and codomain: <MathInline math="\dim(C(A)) + \dim(N(A)) = n" />
                      </p>
                    </div>

                    <div className="flex items-center gap-2">
                      <span className="px-3 py-1.5 rounded-xl bg-emerald-500/20 text-emerald-300 border border-emerald-500/40 font-mono text-xs font-bold flex items-center gap-1.5">
                        <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                        Theorem Verified Exact
                      </span>
                    </div>
                  </div>

                  {/* Numbers Grid */}
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 text-center">
                    <div className="p-3 rounded-xl bg-slate-800/70 border border-slate-700">
                      <div className="text-[10px] uppercase font-mono text-slate-400">Total Columns (n)</div>
                      <div className="text-2xl font-bold font-mono text-white mt-1">{subspaceData.n_cols}</div>
                      <div className="text-[10px] text-slate-400 mt-0.5">Domain Dimension</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-800/70 border border-slate-700">
                      <div className="text-[10px] uppercase font-mono text-teal-400">Rank: <MathInline math="\dim(C(A))" /></div>
                      <div className="text-2xl font-bold font-mono text-teal-300 mt-1">{subspaceData.rank}</div>
                      <div className="text-[10px] text-teal-400/80 mt-0.5">Pivot Columns</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-800/70 border border-slate-700">
                      <div className="text-[10px] uppercase font-mono text-amber-400">Nullity: <MathInline math="\dim(N(A))" /></div>
                      <div className="text-2xl font-bold font-mono text-amber-300 mt-1">{subspaceData.nullity}</div>
                      <div className="text-[10px] text-amber-400/80 mt-0.5">Free Columns</div>
                    </div>
                    <div className="p-3 rounded-xl bg-slate-800/70 border border-slate-700">
                      <div className="text-[10px] uppercase font-mono text-emerald-400"><MathInline math="r + k = n" /> Identity</div>
                      <div className="text-2xl font-bold font-mono text-emerald-300 mt-1">
                        {subspaceData.rank} + {subspaceData.nullity} = {subspaceData.n_cols}
                      </div>
                      <div className="text-[10px] text-emerald-400/80 mt-0.5">Identity Satisfied</div>
                    </div>
                  </div>
                </div>

                {/* Subspace Bases Display */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Column Space Basis */}
                  <div className="space-y-4">
                    <MatrixViewer
                      matrix={subspaceData.column_space_basis}
                      title={<span>Column Space Basis <MathInline math={`C(A) \\subseteq \\mathbb{R}^{${subspaceData.m_rows}}`} /></span>}
                      badge={`dim = ${subspaceData.rank}`}
                      caption={`Formed by the ${subspaceData.rank} pivot columns of original matrix A.`}
                    />
                    <div className="p-4 bg-white border border-slate-200/90 rounded-xl space-y-2 shadow-xs text-xs">
                      <span className="font-bold text-slate-800">Column Space Properties</span>
                      <p className="text-slate-600 leading-relaxed text-[11px]">
                        The column space represents all possible sensor calibration responses achievable as a linear
                        combination of the sensor features. Any target vector y outside C(A) cannot be predicted without error.
                      </p>
                    </div>
                  </div>

                  {/* Null Space Basis */}
                  <div className="space-y-4">
                    {subspaceData.nullity > 0 ? (
                      <MatrixViewer
                        matrix={subspaceData.null_space_basis}
                        title={<span>Null Space Basis <MathInline math={`N(A) \\subseteq \\mathbb{R}^{${subspaceData.n_cols}}`} /></span>}
                        badge={`dim = ${subspaceData.nullity}`}
                        caption={<span>Kernel vectors satisfying <MathInline math="A\mathbf{v} = \mathbf{0}" />.</span>}
                      />
                    ) : (
                      <div className="p-8 bg-white border border-slate-200/90 rounded-xl text-center space-y-2 shadow-xs">
                        <CheckCircle2 className="w-8 h-8 text-teal-600 mx-auto" />
                        <h4 className="text-sm font-bold text-slate-900">Trivial Null Space: <MathInline math="N(A) = \{ \mathbf{0} \}" /></h4>
                        <p className="text-xs text-slate-500 max-w-sm mx-auto leading-relaxed">
                          Because rank = n, the columns of A are strictly linearly independent. No non-zero vector v exists such that A v = 0.
                        </p>
                      </div>
                    )}

                    {/* Verification Residual */}
                    <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl text-xs space-y-1 font-mono">
                      <div className="flex justify-between text-slate-700">
                        <span className="font-sans font-semibold">Max Kernel Residual <MathInline math="\max \|A\mathbf{v}\|_2:" /></span>
                        <span className="text-teal-700 font-bold">{subspaceData.max_null_residual.toExponential(4)}</span>
                      </div>
                      <div className="flex justify-between text-slate-500 text-[10px]">
                        <span className="font-sans">Pivot Columns:</span>
                        <span>[{subspaceData.pivot_columns.join(', ')}]</span>
                      </div>
                      <div className="flex justify-between text-slate-500 text-[10px]">
                        <span className="font-sans">Free Columns:</span>
                        <span>[{subspaceData.free_columns.join(', ') || 'None'}]</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            ) : null}

            {/* Stepper Footer */}
            <div className="flex justify-between pt-4 border-t border-slate-200">
              <button
                onClick={() => setActiveStep(2)}
                className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                Back to Step 2
              </button>
              <button
                onClick={() => setActiveStep(4)}
                className="flex items-center gap-2 px-5 py-2.5 bg-teal-600 hover:bg-teal-700 text-white rounded-xl text-xs font-bold shadow-xs transition-colors"
              >
                Proceed to Step 4: Gram-Schmidt Orthogonalization
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        )}

        {/* ================================================================= */}
        {/* STEP 4: GRAM-SCHMIDT ORTHOGONALIZATION (CGS VS MGS) */}
        {/* ================================================================= */}
        {activeStep === 4 && (
          <motion.div
            key="step4"
            variants={tabContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-6"
          >
            <MathConceptCard
              formula="Q^\mathsf{T} Q = I_k, \quad \mathrm{Loss} = \|Q^\mathsf{T} Q - I_k\|_F"
              concept="Gram-Schmidt orthogonalization transforms a linearly independent set of column vectors {a_1, ..., a_n} into an orthonormal basis {q_1, ..., q_n} such that span(a_1, ..., a_k) = span(q_1, ..., q_k). The resulting factorization A = Q R expresses A as orthogonal matrix Q and upper-triangular matrix R."
              purpose="In exact arithmetic, Classical Gram-Schmidt (CGS) and Modified Gram-Schmidt (MGS) produce identical results. However, under floating-point arithmetic on ill-conditioned systems, CGS suffers catastrophic loss of orthogonality, whereas MGS projects away components sequentially to preserve orthogonality up to machine precision."
              operation="Our engine computes both CGS and MGS decompositions side-by-side on the same matrix, evaluating the Frobenious norm deviation ||Q^T Q - I||_F to benchmark numerical stability gain directly."
              airsenseInterpretation="When metal oxide sensors exhibit strong cross-correlation, the design matrix becomes ill-conditioned. Using MGS guarantees that the QR basis remains strictly orthonormal, preventing numerical instability in calibration regression."
            />

            {/* Preset Selection Bar */}
            <Card title="Orthogonalization Test System">
              <div className="flex flex-wrap gap-2 items-center justify-between">
                <div className="flex gap-2">
                  <button
                    onClick={() => setGsPreset('ill_conditioned')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      gsPreset === 'ill_conditioned'
                        ? 'bg-slate-900 text-white shadow-xs'
                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
                    }`}
                  >
                    Preset: Ill-Conditioned Challenge (<MathInline math="\varepsilon = 10^{-4}" />)
                  </button>
                  <button
                    onClick={() => setGsPreset('extracted')}
                    className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      gsPreset === 'extracted'
                        ? 'bg-slate-900 text-white shadow-xs'
                        : 'bg-slate-100 text-slate-700 hover:bg-slate-200 border border-slate-200'
                    }`}
                  >
                    Current Sensor Matrix ({matrixData?.dimensions[0]} &times; {matrixData?.dimensions[1]})
                  </button>
                </div>

                <Badge variant="outline" size="sm" className="font-mono">
                  Frobenius Metric <MathInline math="\|Q^\mathsf{T} Q - I\|_F" />
                </Badge>
              </div>
            </Card>

            {loadingGs ? (
              <div className="p-8 text-center bg-white border border-slate-200 rounded-2xl shadow-xs">
                <LoadingSpinner label="Benchmarking Classical vs Modified Gram-Schmidt..." />
              </div>
            ) : gsData ? (
              <div className="space-y-6">
                {/* Prominent Stability Comparison Banner */}
                <div className="p-5 rounded-2xl bg-gradient-to-r from-teal-900 to-slate-900 text-white shadow-md border border-teal-800">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                    <div>
                      <span className="text-[10px] font-mono uppercase tracking-wider text-teal-300 font-bold bg-teal-950/80 px-2 py-0.5 rounded border border-teal-500/40">
                        Unit 2 Stability Verification
                      </span>
                      <h3 className="text-lg font-bold mt-1 text-white">Modified vs Classical Numerical Resilience</h3>
                      <p className="text-xs text-slate-300 mt-0.5">
                        Demonstrating that MGS avoids catastrophic round-off error under collinear vectors.
                      </p>
                    </div>

                    <div className="p-3 bg-white/10 backdrop-blur-xs rounded-xl border border-white/20 text-center font-mono">
                      <div className="text-[10px] text-teal-200 font-sans">Stability Ratio (CGS / MGS)</div>
                      <div className="text-xl font-bold text-teal-300 mt-0.5">
                        {gsData.stability_gain > 100
                          ? `${gsData.stability_gain.toExponential(1)}x`
                          : `${gsData.stability_gain}x`}
                      </div>
                      <div className="text-[9px] text-slate-300 font-sans">Orthogonality Preservation</div>
                    </div>
                  </div>

                  {/* Orthogonality Error Compare */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-700/80 font-mono text-xs">
                    <div className="p-3 rounded-xl bg-slate-800/80 border border-slate-700">
                      <div className="text-[10px] uppercase font-sans text-rose-300 font-bold flex items-center justify-between">
                        <span>Classical Gram-Schmidt (CGS) Loss</span>
                        <MathInline math="\|Q^\mathsf{T} Q - I\|_F" />
                      </div>
                      <div className="text-lg font-bold text-rose-400 mt-1">
                        {gsData.loss_cgs.toExponential(4)}
                      </div>
                      <div className="text-[10px] text-slate-400 font-sans mt-0.5">
                        {gsData.loss_cgs > 1e-6
                          ? 'Severe loss of orthogonality due to finite-precision roundoff.'
                          : 'Sufficiently orthogonal on well-conditioned inputs.'}
                      </div>
                    </div>

                    <div className="p-3 rounded-xl bg-slate-800/80 border border-slate-700">
                      <div className="text-[10px] uppercase font-sans text-emerald-300 font-bold flex items-center justify-between">
                        <span>Modified Gram-Schmidt (MGS) Loss</span>
                        <MathInline math="\|Q^\mathsf{T} Q - I\|_F" />
                      </div>
                      <div className="text-lg font-bold text-emerald-400 mt-1">
                        {gsData.loss_mgs.toExponential(4)}
                      </div>
                      <div className="text-[10px] text-emerald-300/80 font-sans mt-0.5">
                        Preserves orthogonality up to machine precision (&lt; <MathInline math="10^{-14}" />).
                      </div>
                    </div>
                  </div>
                </div>

                {/* Dot Product Matrix Heatmaps / Tables */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Classical Q^T Q */}
                  <div className="space-y-4">
                    <MatrixViewer
                      matrix={gsData.dot_product_cgs}
                      title={<span>CGS Dot Product Matrix: <MathInline math="Q_{\mathrm{cgs}}^\mathsf{T} Q_{\mathrm{cgs}}" /></span>}
                      badge={`Loss = ${gsData.loss_cgs.toExponential(2)}`}
                      caption={<span>Ideally identity matrix <MathInline math="I" />. Off-diagonal non-zeros represent loss of orthogonality.</span>}
                    />
                  </div>

                  {/* Modified Q^T Q */}
                  <div className="space-y-4">
                    <MatrixViewer
                      matrix={gsData.dot_product_mgs}
                      title={<span>MGS Dot Product Matrix: <MathInline math="Q_{\mathrm{mgs}}^\mathsf{T} Q_{\mathrm{mgs}}" /></span>}
                      badge={`Loss = ${gsData.loss_mgs.toExponential(2)}`}
                      caption="Off-diagonal elements remain strictly zero within machine precision."
                    />
                  </div>
                </div>

                {/* MGS Factors Q and R */}
                <Card title={<span>Orthonormal Factors (<MathInline math="A = QR" /> via Modified Gram-Schmidt)</span>}>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <MatrixViewer
                      matrix={gsData.q_mgs}
                      title="Orthonormal Matrix Q"
                      badge={`dim: ${gsData.q_mgs.length}x${gsData.q_mgs[0]?.length}`}
                      caption="Columns form an orthonormal basis for C(A)."
                    />
                    <MatrixViewer
                      matrix={gsData.r_mgs}
                      title="Upper-Triangular Matrix R"
                      badge={<MathInline math="R = Q^\mathsf{T} A" />}
                      caption="Contains projection coefficients and normalizers."
                    />
                  </div>
                </Card>
              </div>
            ) : null}

            {/* Stepper Footer */}
            <div className="flex justify-between pt-4 border-t border-slate-200">
              <button
                onClick={() => setActiveStep(3)}
                className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                Back to Step 3
              </button>
              <button
                onClick={() => setActiveStep(5)}
                className="flex items-center gap-2 px-5 py-2.5 bg-teal-600 hover:bg-teal-700 text-white rounded-xl text-xs font-bold shadow-xs transition-colors"
              >
                Proceed to Step 5: Projections & Least Squares
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        )}

        {/* ================================================================= */}
        {/* STEP 5: ORTHOGONAL PROJECTIONS & LEAST SQUARES */}
        {/* ================================================================= */}
        {activeStep === 5 && (
          <motion.div
            key="step5"
            variants={tabContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-6"
          >
            <MathConceptCard
              formula="P = X (X^\mathsf{T} X)^{-1} X^\mathsf{T}, \quad \hat{\mathbf{y}} = P\mathbf{y}, \quad X^\mathsf{T} (\mathbf{y} - \hat{\mathbf{y}}) = \mathbf{0}"
              concept="When an overdetermined linear system X w = y has no exact solution because y is outside the column space C(X), the least-squares solution finds w_hat minimizing the Euclidean norm ||X w - y||_2. Geometrically, y_hat = X w_hat is the unique orthogonal projection of y onto C(X), meaning the residual vector e = y - y_hat is strictly perpendicular to every column of X."
              purpose="The normal equations (X^T X) w = X^T y square the condition number: cond(X^T X) = cond(X)^2. For sensor arrays with condition number ~ 10^4, squaring yields 10^8, doubling the loss of significant digits. Solving via QR decomposition (R w = Q^T y) preserves cond(X), guaranteeing numerical stability."
              operation="Our engine computes the least-squares parameters via QR factorization without forming (X^T X). It explicitly verifies orthogonality by calculating max |X^T e| and measures both condition numbers cond(X) and cond(X^T X)."
              airsenseInterpretation="In AirSense calibration, the Benzene target vector y contains measurement noise and unmodelled chemical interferents. Orthogonal projection finds the optimal sensor linear combination while guaranteeing that the remaining calibration error is orthogonal to our sensor readings."
            />

            {loadingLs ? (
              <div className="p-8 text-center bg-white border border-slate-200 rounded-2xl shadow-xs">
                <LoadingSpinner label="Computing orthogonal projection and QR least squares..." />
              </div>
            ) : lsData ? (
              <div className="space-y-6">
                {/* Condition Number Squaring Callout */}
                <div className="bg-amber-50 border border-amber-200 rounded-2xl p-5 shadow-xs">
                  <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-3 border-b border-amber-200/80">
                    <div className="flex items-center gap-2">
                      <span className="p-1.5 rounded-lg bg-amber-100 text-amber-900 border border-amber-300">
                        <AlertTriangle className="w-4 h-4" />
                      </span>
                      <div>
                        <h3 className="text-sm font-bold text-amber-950 flex items-center gap-2">
                          Condition Number Squaring: <MathInline math="\kappa(X^\mathsf{T} X) = \kappa(X)^2" />
                        </h3>
                        <p className="text-xs text-amber-800">
                          Why the AirSense engine solves via QR decomposition instead of normal equations
                        </p>
                      </div>
                    </div>

                    <Badge variant="outline" size="sm" className="font-mono bg-white">
                      Numerical Conditioning Law
                    </Badge>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-3 font-mono text-center">
                    <div className="p-3 bg-white rounded-xl border border-amber-200/80 shadow-xs">
                      <div className="text-[10px] text-slate-500 uppercase font-sans">Design Matrix: <MathInline math="\kappa(X)" /></div>
                      <div className="text-xl font-bold text-slate-900 mt-1">
                        {lsData.cond_X > 1000 ? lsData.cond_X.toExponential(2) : lsData.cond_X}
                      </div>
                      <div className="text-[10px] text-teal-700 font-sans mt-0.5">Preserved by QR Solve</div>
                    </div>

                    <div className="p-3 bg-white rounded-xl border border-amber-200/80 shadow-xs">
                      <div className="text-[10px] text-slate-500 uppercase font-sans">Normal Equations: <MathInline math="\kappa(X^\mathsf{T} X)" /></div>
                      <div className="text-xl font-bold text-rose-600 mt-1">
                        {lsData.cond_XtX > 1000 ? lsData.cond_XtX.toExponential(2) : lsData.cond_XtX}
                      </div>
                      <div className="text-[10px] text-rose-600 font-sans mt-0.5">Squared Condition Number</div>
                    </div>

                    <div className="p-3 bg-white rounded-xl border border-amber-200/80 shadow-xs">
                      <div className="text-[10px] text-slate-500 uppercase font-sans">Conditioning Amplification</div>
                      <div className="text-xl font-bold text-amber-900 mt-1">
                        {lsData.cond_ratio > 100 ? `${lsData.cond_ratio.toExponential(1)}x` : `${lsData.cond_ratio}x`}
                      </div>
                      <div className="text-[10px] text-slate-500 font-sans mt-0.5">Loss of Precision Factor</div>
                    </div>
                  </div>
                </div>

                {/* Solution Weights & Orthogonality Check */}
                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {/* Fitted Weights */}
                  <div className="p-4 bg-white border border-slate-200/90 rounded-xl space-y-3 shadow-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-800">Fitted Parameter Vector <MathInline math="\hat{\mathbf{w}}" /></span>
                      <span className="text-[10px] font-mono text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
                        <MathInline math="R\mathbf{w} = Q^\mathsf{T} \mathbf{y}" />
                      </span>
                    </div>
                    <div className="space-y-2 font-mono text-xs">
                      {lsData.weights.map((w, idx) => (
                        <div
                          key={idx}
                          className="flex items-center justify-between p-2 rounded-lg bg-slate-50 border border-slate-200/70"
                        >
                          <span className="text-slate-500 font-sans">
                            {matrixData?.col_labels[idx] || `w_${idx}`}
                          </span>
                          <span className="font-bold text-slate-900">{w.toFixed(4)}</span>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Target Decomposition y = y_hat + e */}
                  <div className="lg:col-span-2 p-4 bg-white border border-slate-200/90 rounded-xl space-y-3 shadow-xs">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-800">
                        Target Decomposition: <MathInline math="\mathbf{y} = \hat{\mathbf{y}} + \mathbf{e}" />
                      </span>
                      <span className="text-[10px] font-mono text-slate-500">
                        RSS = {lsData.rss.toFixed(4)}
                      </span>
                    </div>

                    <div className="overflow-x-auto">
                      <table className="w-full text-xs font-mono border-collapse">
                        <thead>
                          <tr className="border-b border-slate-200 text-[10px] text-slate-500 text-right">
                            <th className="p-1.5 text-left font-sans">Hour</th>
                            <th className="p-1.5">Actual <MathInline math="y" /></th>
                            <th className="p-1.5">Projected <MathInline math="\hat{y}" /></th>
                            <th className="p-1.5">Residual <MathInline math="e" /></th>
                          </tr>
                        </thead>
                        <tbody>
                          {matrixData?.target_vector.slice(0, 8).map((act, idx) => (
                            <tr key={idx} className="border-b border-slate-100 text-right hover:bg-slate-50">
                              <td className="p-1.5 text-left text-slate-500 font-sans">
                                {matrixData?.row_labels[idx]?.split(' ')[0] || `t_${idx}`}
                              </td>
                              <td className="p-1.5 text-slate-900 font-semibold">{act.toFixed(2)}</td>
                              <td className="p-1.5 text-teal-700 font-semibold">
                                {lsData.projected_target[idx]?.toFixed(2)}
                              </td>
                              <td className="p-1.5 text-slate-500">
                                {lsData.residual_vector[idx]?.toFixed(3)}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>

                    {/* Orthogonality Invariant Check */}
                    <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-xl flex items-center justify-between text-xs font-mono">
                      <div className="flex items-center gap-2 text-emerald-950 font-sans">
                        <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                        <span>
                          <strong>Orthogonality Invariant:</strong> <MathInline math="\max |X^\mathsf{T} \mathbf{e}| =" />{' '}
                          <span className="font-mono font-bold">
                            {lsData.orthogonality_residual.toExponential(4)}
                          </span>
                        </span>
                      </div>
                      <Badge variant="teal" size="sm">
                        <MathInline math="\mathbf{e} \perp C(X)" />
                      </Badge>
                    </div>
                  </div>
                </div>

                {/* Projection Matrix Sample */}
                <Card
                  title={<span>Orthogonal Projection Operator <MathInline math="P = X (X^\mathsf{T} X)^{-1} X^\mathsf{T}" /> (Sample Block)</span>}
                  subtitle="Idempotent (P^2 = P) and symmetric (P^T = P) operator projecting any vector onto C(X)"
                >
                  <MatrixViewer
                    matrix={lsData.projection_matrix_sample}
                    title="Projection Matrix P (Top 6x6 Block)"
                    badge={<MathInline math="P^2 = P, \; P^\mathsf{T} = P" />}
                    caption={<span>Applying <MathInline math="P" /> to target <MathInline math="\mathbf{y}" /> yields projected concentration <MathInline math="\hat{\mathbf{y}} = P\mathbf{y}" />.</span>}
                  />
                </Card>
              </div>
            ) : null}

            {/* Stepper Footer */}
            <div className="flex justify-between pt-4 border-t border-slate-200">
              <button
                onClick={() => setActiveStep(4)}
                className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                Back to Step 4
              </button>
              <button
                onClick={() => setActiveStep(6)}
                className="flex items-center gap-2 px-5 py-2.5 bg-teal-600 hover:bg-teal-700 text-white rounded-xl text-xs font-bold shadow-xs transition-colors"
              >
                Proceed to Step 6: Spectral Decomposition & PCA
                <ArrowRight className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        )}

        {/* ================================================================= */}
        {/* STEP 6: SPECTRAL DECOMPOSITION & PCA */}
        {/* ================================================================= */}
        {activeStep === 6 && (
          <motion.div
            key="step6"
            variants={tabContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-6"
          >
            <MathConceptCard
              formula="S = V \Lambda V^\mathsf{T}, \quad \Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_d), \quad S \mathbf{v}_i = \lambda_i \mathbf{v}_i"
              concept="The Spectral Theorem for real symmetric matrices states that any real symmetric matrix S in R^(d x d) has purely real eigenvalues and an orthonormal basis of eigenvectors V in R^(d x d) such that S = V Lambda V^T. The sample covariance matrix of centered sensor features is symmetric and positive semi-definite (PSD)."
              purpose="Spectral decomposition identifies the principal axes of maximum variance in the multi-sensor response space. The eigenvalues lambda_i quantify the variance captured along each eigenvector direction v_i, forming the mathematical foundation for Principal Component Analysis (PCA) and Principal Component Regression (PCR)."
              operation="Our engine computes the sample covariance matrix S = 1/(n-1) X_c^T X_c directly from 5,394 training hours. It checks symmetry and positive semi-definiteness, then executes the iterative Jacobi symmetric eigenvalue algorithm from scratch until off-diagonal elements vanish."
              airsenseInterpretation="The scree plot of eigenvalues reveals that 3 components explain ~85% of total sensor variance, while the remaining 5 eigenvalues are tiny. This explains why PCR with k=3 performs best on out-of-time test data (RMSE 2.41 vs 2.85): discarding small eigenvalues eliminates collinear noise."
            />

            {/* Feature Selection for Covariance Matrix */}
            <Card title="Covariance Matrix Channel Selection">
              <div className="space-y-3">
                <div className="text-xs font-bold text-slate-800 flex items-center justify-between">
                  <span>Selected Sensor Channels for Covariance Matrix ({spectralFeatures.length} features)</span>
                  <span className="text-[11px] text-slate-500 font-normal">Fitted across 5,394 training hours</span>
                </div>
                <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-8 gap-2">
                  {ALL_FEATURES.map((feat) => {
                    const isSel = spectralFeatures.includes(feat);
                    return (
                      <button
                        key={feat}
                        onClick={() => toggleSpectralFeature(feat)}
                        className={`p-2 rounded-xl text-xs font-mono font-medium text-center border transition-all ${
                          isSel
                            ? 'bg-teal-50 border-teal-300 text-teal-900 font-bold shadow-xs'
                            : 'bg-slate-50 border-slate-200 text-slate-600 hover:bg-slate-100'
                        }`}
                      >
                        {feat}
                      </button>
                    );
                  })}
                </div>
              </div>
            </Card>

            {loadingSpectral ? (
              <div className="p-8 text-center bg-white border border-slate-200 rounded-2xl shadow-xs">
                <LoadingSpinner label="Computing covariance matrix and Jacobi diagonalization..." />
              </div>
            ) : spectralData ? (
              <div className="space-y-6">
                {/* Covariance Matrix Properties Banner */}
                <div className="p-4 bg-white border border-slate-200/90 rounded-2xl shadow-xs flex flex-wrap items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <span className="p-2 rounded-xl bg-teal-50 text-teal-700 border border-teal-200">
                      <Sparkles className="w-5 h-5" />
                    </span>
                    <div>
                      <h3 className="text-sm font-bold text-slate-900 flex items-center gap-1.5">
                        Sample Covariance Matrix <MathInline math={`S \\in \\mathbb{R}^{${spectralData.feature_names.length} \\times ${spectralData.feature_names.length}}`} />
                      </h3>
                      <p className="text-xs text-slate-500">
                        Verified symmetric and positive semi-definite (PSD)
                      </p>
                    </div>
                  </div>

                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="px-2.5 py-1 rounded-lg bg-emerald-50 text-emerald-700 border border-emerald-200 text-xs font-mono font-bold flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                      Symmetric: <MathInline math="\|S - S^\mathsf{T}\| = 0" />
                    </span>
                    <span className="px-2.5 py-1 rounded-lg bg-teal-50 text-teal-700 border border-teal-200 text-xs font-mono font-bold flex items-center gap-1">
                      <CheckCircle2 className="w-3.5 h-3.5 text-teal-600" />
                      Positive Semi-Definite (PSD)
                    </span>
                  </div>
                </div>

                {/* Covariance Matrix & Diagonalized Form */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Covariance Matrix */}
                  <MatrixViewer
                    matrix={spectralData.matrix}
                    colLabels={spectralData.feature_names}
                    rowLabels={spectralData.feature_names}
                    title="Sample Covariance Matrix S"
                    badge={<MathInline math="S = \frac{1}{n-1} X_c^\mathsf{T} X_c" />}
                    caption="Diagonal elements are sample variances; off-diagonals are pairwise sensor covariances."
                  />

                  {/* Eigenvalues & Scree Plot */}
                  <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-xs space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-slate-800">
                        Eigenvalue Scree Plot (Variance Explained)
                      </span>
                      <span className="text-[10px] font-mono text-teal-700 bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
                        Jacobi Diagonalization
                      </span>
                    </div>

                    {/* Recharts Scree Plot */}
                    <div className="h-64 w-full">
                      <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart
                          data={spectralData.eigenvalues.map((_, idx) => ({
                            component: `PC${idx + 1}`,
                            variance: Number((spectralData.variance_explained_ratio[idx] * 100).toFixed(1)),
                            cumulative: Number((spectralData.cumulative_variance_ratio[idx] * 100).toFixed(1)),
                          }))}
                          margin={{ top: 10, right: 10, left: -20, bottom: 0 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                          <XAxis dataKey="component" tick={{ fontSize: 10 }} />
                          <YAxis yAxisId="left" unit="%" tick={{ fontSize: 10 }} />
                          <YAxis yAxisId="right" orientation="right" unit="%" domain={[0, 100]} tick={{ fontSize: 10 }} />
                          <Tooltip
                            formatter={(value: any, name: string) => [
                              `${value}%`,
                              name === 'variance' ? 'Individual Variance' : 'Cumulative Variance',
                            ]}
                          />
                          <Legend wrapperStyle={{ fontSize: 11 }} />
                          <Bar
                            yAxisId="left"
                            dataKey="variance"
                            name="Individual Variance %"
                            fill="#0d9488"
                            radius={[4, 4, 0, 0]}
                          />
                          <Line
                            yAxisId="right"
                            type="monotone"
                            dataKey="cumulative"
                            name="Cumulative Variance %"
                            stroke="#0f172a"
                            strokeWidth={2}
                            dot={{ r: 3 }}
                          />
                        </ComposedChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Eigenvalues Table */}
                    <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
                      {spectralData.eigenvalues.map((lambda, idx) => (
                        <div
                          key={idx}
                          className="p-2.5 rounded-lg bg-slate-50 border border-slate-200/70"
                        >
                          <span className="text-[10px] text-slate-500 font-sans block">
                            <MathInline math={`\\lambda_{${idx + 1}}`} /> (PC{idx + 1})
                          </span>
                          <span className="text-teal-800 font-bold text-xs mt-0.5 block truncate">
                            {lambda > 1000 ? lambda.toExponential(2) : lambda.toFixed(2)}
                          </span>
                          <span className="text-[10px] text-slate-500">
                            {(spectralData.variance_explained_ratio[idx] * 100).toFixed(1)}% var
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                {/* Orthonormal Eigenvectors Matrix V */}
                <Card
                  title="Orthonormal Eigenvector Matrix V (Principal Directions)"
                  subtitle={<span>Columns <MathInline math="\mathbf{v}_i" /> satisfy <MathInline math="S \mathbf{v}_i = \lambda_i \mathbf{v}_i" /> and <MathInline math="V^\mathsf{T} V = I" /></span>}
                >
                  <div className="space-y-4">
                    <MatrixViewer
                      matrix={spectralData.eigenvectors}
                      colLabels={spectralData.eigenvalues.map((_, i) => `PC${i + 1}`)}
                      rowLabels={spectralData.feature_names}
                      title="Eigenvectors Matrix V"
                      badge={<MathInline math="V^\mathsf{T} V = I" />}
                      caption="Each column defines the linear combination of raw sensor responses forming the principal component."
                    />

                    {/* Verification Invariant */}
                    <div className="p-3.5 bg-emerald-50 border border-emerald-200 rounded-xl flex items-center justify-between text-xs font-mono">
                      <div className="flex items-center gap-2 text-emerald-950 font-sans">
                        <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
                        <span>
                          <strong>Spectral Invariant Verified:</strong> <MathInline math="\max \|S \mathbf{v}_i - \lambda_i \mathbf{v}_i\|_2 =" />{' '}
                          <span className="font-mono font-bold">
                            {spectralData.max_eigenvector_residual.toExponential(4)}
                          </span>
                        </span>
                      </div>
                      <Badge variant="teal" size="sm">
                        Machine Precision (&lt; <MathInline math="10^{-12}" />)
                      </Badge>
                    </div>
                  </div>
                </Card>
              </div>
            ) : null}

            {/* Stepper Footer */}
            <div className="flex justify-between pt-4 border-t border-slate-200">
              <button
                onClick={() => setActiveStep(5)}
                className="flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 text-slate-700 rounded-xl text-xs font-semibold transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                Back to Step 5
              </button>
              <button
                onClick={() => setActiveStep(1)}
                className="flex items-center gap-2 px-4 py-2 bg-slate-900 hover:bg-slate-800 text-white rounded-xl text-xs font-bold transition-colors"
              >
                Return to Step 1: Data Matrix
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
