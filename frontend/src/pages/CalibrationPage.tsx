import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  Cell,
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';
import {
  CheckCircle2,
  Sliders,
  Play,
  RotateCcw,
  ChevronDown,
  ChevronUp,
  Cpu,
  ShieldCheck,
  Zap,
  Sparkles,
} from 'lucide-react';
import { apiClient } from '../services/apiClient';
import {
  ModelBenchmarkRow,
  CalibrationRunResponse,
} from '../types/api';
import { Card } from '../components/ui/Card';
import { MetricCard } from '../components/ui/MetricCard';
import { Badge } from '../components/ui/Badge';
import { Alert } from '../components/ui/Alert';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { ErrorState } from '../components/ui/ErrorState';
import { MathInline } from '../components/ui/Math';
import { tabContentVariants, accordionVariants } from '../lib/motion';

const ALL_PREDICTORS = [
  { id: 'PT08.S1(CO)', label: 'PT08.S1 (CO)', category: 'sensor', desc: 'Carbon Monoxide response' },
  { id: 'PT08.S2(NMHC)', label: 'PT08.S2 (NMHC)', category: 'sensor', desc: 'Non-Methane Hydrocarbons (primary target proxy)' },
  { id: 'PT08.S3(NOx)', label: 'PT08.S3 (NOx)', category: 'sensor', desc: 'Nitrogen Oxides response' },
  { id: 'PT08.S4(NO2)', label: 'PT08.S4 (NO2)', category: 'sensor', desc: 'Nitrogen Dioxide response' },
  { id: 'PT08.S5(O3)', label: 'PT08.S5 (O3)', category: 'sensor', desc: 'Ozone response' },
  { id: 'T', label: 'T (°C)', category: 'env', desc: 'Ambient Temperature' },
  { id: 'RH', label: 'RH (%)', category: 'env', desc: 'Relative Humidity' },
  { id: 'AH', label: 'AH (g/m³)', category: 'env', desc: 'Absolute Humidity' },
];

export const CalibrationPage: React.FC = () => {
  // Experiment configuration state
  const [modelType, setModelType] = useState<'OLS' | 'PCR'>('OLS');
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(
    ALL_PREDICTORS.map((p) => p.id)
  );
  const [kComponents, setKComponents] = useState<number>(3);

  // Results & UI state
  const [calibrationResult, setCalibrationResult] = useState<CalibrationRunResponse | null>(null);
  const [benchmarkMatrix, setBenchmarkMatrix] = useState<ModelBenchmarkRow[]>([]);
  const [activeTab, setActiveTab] = useState<'scatter' | 'timeline' | 'residuals' | 'weights' | 'variance'>('scatter');
  const [mathOpen, setMathOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isComputing, setIsComputing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Clamped components count
  const effectiveK = Math.min(Math.max(1, kComponents), Math.max(1, selectedFeatures.length));

  // Toggle feature selection
  const handleFeatureToggle = (featureId: string) => {
    setSelectedFeatures((prev) => {
      if (prev.includes(featureId)) {
        if (prev.length <= 1) return prev; // Keep at least one feature
        return prev.filter((id) => id !== featureId);
      } else {
        return [...prev, featureId];
      }
    });
  };

  // Run calibration model via backend API
  const runModel = async (overrideType?: 'OLS' | 'PCR', overrideFeatures?: string[], overrideK?: number) => {
    setIsComputing(true);
    setError(null);
    try {
      const type = overrideType || modelType;
      const features = overrideFeatures || selectedFeatures;
      const k = overrideK !== undefined ? overrideK : effectiveK;

      const result = await apiClient.runCalibration({
        model_type: type,
        selected_features: features,
        k_components: type === 'PCR' ? k : undefined,
      });
      setCalibrationResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Calibration calculation failed on backend.');
    } finally {
      setIsComputing(false);
      setIsLoading(false);
    }
  };

  // Initial load: fetch benchmark matrix and run baseline OLS
  useEffect(() => {
    const initialize = async () => {
      try {
        const matrixData = await apiClient.getBenchmarkMatrix();
        setBenchmarkMatrix(matrixData.models);
        await runModel('OLS', ALL_PREDICTORS.map((p) => p.id));
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to connect to AirSense calibration gateway.');
        setIsLoading(false);
      }
    };
    initialize();
  }, []);

  // Quick reset to baseline
  const handleResetBaseline = () => {
    const all = ALL_PREDICTORS.map((p) => p.id);
    setModelType('OLS');
    setSelectedFeatures(all);
    setKComponents(8);
    runModel('OLS', all, 8);
  };

  // Quick preset: Compact PCR (k=3)
  const handlePresetPCR3 = () => {
    const all = ALL_PREDICTORS.map((p) => p.id);
    setModelType('PCR');
    setSelectedFeatures(all);
    setKComponents(3);
    runModel('PCR', all, 3);
  };

  if (isLoading) {
    return <LoadingSpinner label="Loading sensor calibration laboratory & running baseline QR fit..." />;
  }

  if (error && !calibrationResult) {
    return (
      <ErrorState
        title="Calibration Gateway Disconnected"
        message={error}
        onRetry={() => runModel()}
      />
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* 1. Scientific Header */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-[11px] font-bold text-teal-700 tracking-wider uppercase font-mono bg-teal-50 px-2.5 py-0.5 rounded-full border border-teal-200">
              Sensor Calibration Laboratory
            </span>
            <span className="text-slate-300">·</span>
            <span className="text-xs text-slate-500 font-medium">QR Factorization &amp; PCR Subspace Projection</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 font-sans">Calibration &amp; PCA Regression</h1>
          <p className="text-sm text-slate-600 mt-1 max-w-3xl leading-relaxed">
            Translate raw semiconductor potential measurements into certified benzene concentrations (<MathInline math="\mathrm{C}_6\mathrm{H}_6" />).
            Compare Full Ordinary Least Squares (OLS) with Principal Component Regression (PCR) on 5,394 training hours.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <div className="flex items-center gap-2 px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs text-slate-700">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <span className="font-medium">Test Set Sealed (Zero Leakage)</span>
          </div>
          <Badge variant="teal" size="md" className="gap-1.5 font-mono">
            <CheckCircle2 className="w-4 h-4 text-teal-600" />
            Backend Python Engine Ready
          </Badge>
        </div>
      </div>

      {/* 2. Interactive Control Panel */}
      <Card
        title={
          <div className="flex items-center gap-2.5">
            <Sliders className="w-4 h-4 text-teal-600" />
            <span>Interactive Experiment Configuration</span>
          </div>
        }
        subtitle="Select predictor subset and model architecture. The mathematical solver executes in real time on the Python engine."
      >
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            {/* Left: Model Architecture & Components Slider (5 cols) */}
            <div className="lg:col-span-5 space-y-5 border-b lg:border-b-0 lg:border-r border-slate-200/80 pb-6 lg:pb-0 lg:pr-6">
              <div>
                <label className="text-xs font-bold uppercase tracking-wider text-slate-600 mb-2 block font-mono">
                  Model Architecture
                </label>
                <div className="grid grid-cols-2 gap-3">
                  <button
                    type="button"
                    onClick={() => setModelType('OLS')}
                    className={`p-3.5 rounded-xl border text-left transition-all cursor-pointer ${
                      modelType === 'OLS'
                        ? 'bg-teal-50/80 border-teal-500 ring-2 ring-teal-500/20 text-teal-950 shadow-xs'
                        : 'bg-white border-slate-200 hover:border-slate-300 text-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-sm">Full OLS</span>
                      {modelType === 'OLS' && <CheckCircle2 className="w-4 h-4 text-teal-600" />}
                    </div>
                    <p className="text-[11px] text-slate-500 leading-snug">
                      Modified Gram-Schmidt QR decomposition on all selected predictors.
                    </p>
                  </button>

                  <button
                    type="button"
                    onClick={() => setModelType('PCR')}
                    className={`p-3.5 rounded-xl border text-left transition-all cursor-pointer ${
                      modelType === 'PCR'
                        ? 'bg-teal-50/80 border-teal-500 ring-2 ring-teal-500/20 text-teal-950 shadow-xs'
                        : 'bg-white border-slate-200 hover:border-slate-300 text-slate-700'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-bold text-sm">PCR (SVD)</span>
                      {modelType === 'PCR' && <CheckCircle2 className="w-4 h-4 text-teal-600" />}
                    </div>
                    <p className="text-[11px] text-slate-500 leading-snug">
                      Thin SVD dimensionality reduction followed by orthogonal QR regression.
                    </p>
                  </button>
                </div>
              </div>

              {/* PCR Components Slider */}
              <div className={`space-y-3 transition-opacity duration-200 ${modelType === 'PCR' ? 'opacity-100' : 'opacity-40 pointer-events-none'}`}>
                <div className="flex items-center justify-between">
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-600 font-mono">
                    Principal Components (k): <span className="font-mono text-teal-700 text-sm font-bold">{effectiveK}</span>
                  </label>
                  <span className="text-[11px] text-slate-400 font-mono">
                    Max: {selectedFeatures.length} (d)
                  </span>
                </div>

                <input
                  type="range"
                  min="1"
                  max={selectedFeatures.length}
                  value={effectiveK}
                  onChange={(e) => setKComponents(parseInt(e.target.value, 10))}
                  className="w-full h-2 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-teal-600"
                />

                <div className="flex items-center justify-between text-[11px] text-slate-500">
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => setKComponents(1)}
                      className="px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 border border-slate-200 font-mono text-[10px] cursor-pointer"
                    >
                      k=1 (Coarse)
                    </button>
                    <button
                      type="button"
                      onClick={() => setKComponents(Math.min(3, selectedFeatures.length))}
                      className="px-2 py-0.5 rounded bg-teal-50 hover:bg-teal-100 border border-teal-200 text-teal-700 font-mono text-[10px] font-semibold cursor-pointer"
                    >
                      k=3 (Regularized)
                    </button>
                    <button
                      type="button"
                      onClick={() => setKComponents(selectedFeatures.length)}
                      className="px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 border border-slate-200 font-mono text-[10px] cursor-pointer"
                    >
                      k={selectedFeatures.length} (Full)
                    </button>
                  </div>
                  <span className="text-[11px] text-slate-500 font-medium">
                    {effectiveK === selectedFeatures.length ? 'Full Rank Projection' : 'Subspace Compression'}
                  </span>
                </div>
              </div>

              {/* Protocol Badges */}
              <div className="p-3.5 bg-slate-50 rounded-xl border border-slate-200/80 space-y-1.5 text-[11px] text-slate-600">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-700">Data Partitioning</span>
                  <Badge variant="teal" size="sm">Strict Chronological</Badge>
                </div>
                <div className="flex items-center justify-between text-slate-500 font-mono text-[10px] flex-wrap gap-1">
                  <span>Train: 5,394 hrs (60%)</span>
                  <span>&middot;</span>
                  <span>Val: 1,798 hrs (20%)</span>
                  <span>&middot;</span>
                  <span>Held-Out Test: 1,799 hrs (20%)</span>
                </div>
              </div>
            </div>

            {/* Right: Predictor Feature Selection Matrix (7 cols) */}
            <div className="lg:col-span-7 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <label className="text-xs font-bold uppercase tracking-wider text-slate-600 block font-mono">
                    Active Predictor Subset ({selectedFeatures.length} / {ALL_PREDICTORS.length})
                  </label>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Toggle individual sensor and environmental channels to test robustness against channel dropouts.
                  </p>
                </div>

                <div className="flex items-center gap-2">
                  <button
                    type="button"
                    onClick={() => setSelectedFeatures(ALL_PREDICTORS.map((p) => p.id))}
                    className="text-[11px] text-teal-600 hover:text-teal-700 font-semibold px-2 py-1 rounded hover:bg-teal-50 transition-colors cursor-pointer"
                  >
                    Select All
                  </button>
                  <span className="text-slate-300">|</span>
                  <button
                    type="button"
                    onClick={() => setSelectedFeatures(ALL_PREDICTORS.filter((p) => p.category === 'sensor').map((p) => p.id))}
                    className="text-[11px] text-slate-600 hover:text-slate-800 font-semibold px-2 py-1 rounded hover:bg-slate-100 transition-colors cursor-pointer"
                  >
                    Sensors Only
                  </button>
                </div>
              </div>

              {/* Feature Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                {ALL_PREDICTORS.map((predictor) => {
                  const isChecked = selectedFeatures.includes(predictor.id);
                  return (
                    <button
                      type="button"
                      key={predictor.id}
                      onClick={() => handleFeatureToggle(predictor.id)}
                      className={`flex items-start gap-3 p-3 rounded-xl border text-left transition-all cursor-pointer ${
                        isChecked
                          ? 'bg-teal-50/40 border-teal-300 ring-1 ring-teal-500/10 shadow-2xs'
                          : 'bg-slate-50/50 border-slate-200/80 opacity-60 hover:opacity-100'
                      }`}
                    >
                      <input
                        type="checkbox"
                        checked={isChecked}
                        onChange={() => {}} // handled by button
                        className="mt-0.5 rounded border-slate-300 text-teal-600 focus:ring-teal-500 cursor-pointer pointer-events-none"
                      />
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center justify-between">
                          <span className="font-mono text-xs font-bold text-slate-800 truncate">
                            {predictor.id}
                          </span>
                          <span className={`text-[9px] uppercase px-1.5 py-0.2 rounded font-semibold ${
                            predictor.category === 'sensor'
                              ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                              : 'bg-sky-50 text-sky-700 border border-sky-200'
                          }`}>
                            {predictor.category === 'sensor' ? 'MOX' : 'Env'}
                          </span>
                        </div>
                        <p className="text-[10px] text-slate-500 truncate mt-0.5">{predictor.desc}</p>
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Actions & Warnings */}
              <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-t border-slate-100">
                <div className="text-[11px] text-slate-500 flex items-center gap-2">
                  <Zap className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                  <span>
                    Certified reference analyzer columns <code className="bg-slate-100 px-1 py-0.5 rounded text-rose-600 font-mono">(GT)</code> are excluded to prevent target leakage.
                  </span>
                </div>

                <div className="flex items-center gap-2 shrink-0">
                  <button
                    type="button"
                    onClick={handlePresetPCR3}
                    className="flex items-center gap-1.5 px-3 py-2 border border-sky-200 hover:border-sky-300 rounded-xl text-xs font-semibold text-sky-800 hover:bg-sky-50 transition-colors cursor-pointer shadow-2xs"
                  >
                    <Sparkles className="w-3.5 h-3.5 text-sky-600" />
                    Compact PCR (k=3)
                  </button>

                  <button
                    type="button"
                    onClick={handleResetBaseline}
                    className="flex items-center gap-1.5 px-3 py-2 border border-slate-200 hover:border-slate-300 rounded-xl text-xs font-semibold text-slate-700 hover:bg-slate-50 transition-colors cursor-pointer shadow-2xs"
                  >
                    <RotateCcw className="w-3.5 h-3.5 text-slate-400" />
                    Reset Baseline
                  </button>

                  <button
                    type="button"
                    disabled={isComputing || selectedFeatures.length === 0}
                    onClick={() => runModel()}
                    className="flex items-center gap-2 px-4 py-2 bg-teal-600 hover:bg-teal-700 active:scale-95 disabled:bg-teal-300 text-white rounded-xl text-xs font-bold shadow-sm hover:shadow transition-all cursor-pointer"
                  >
                    {isComputing ? (
                      <>
                        <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        <span>Solving QR...</span>
                      </>
                    ) : (
                      <>
                        <Play className="w-3.5 h-3.5 fill-current" />
                        <span>Run Calibration Fit</span>
                      </>
                    )}
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Warnings emitted by service */}
          {calibrationResult?.warnings && calibrationResult.warnings.length > 0 && (
            <div className="space-y-2 pt-2">
              {calibrationResult.warnings.map((w, idx) => (
                <Alert key={idx} variant="warning" title="Numerical / Collinearity Warning">
                  {w}
                </Alert>
              ))}
            </div>
          )}
        </div>
      </Card>

      {/* 3. Validation Performance KPI Strip */}
      {calibrationResult && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-5">
          <MetricCard
            label="Validation RMSE"
            value={calibrationResult.val_metrics.rmse.toFixed(3)}
            unit={<MathInline math="\mu\mathrm{g/m}^3" />}
            subtitle={<span>Train RMSE: {calibrationResult.train_metrics.rmse.toFixed(3)} <MathInline math="\mu\mathrm{g/m}^3" /></span>}
            badge={{
              text: calibrationResult.val_metrics.rmse <= 2.05 ? 'Optimal (Val)' : 'Sub-optimal',
              variant: calibrationResult.val_metrics.rmse <= 2.05 ? 'teal' : 'default',
            }}
          />

          <MetricCard
            label="Validation MAE"
            value={calibrationResult.val_metrics.mae.toFixed(3)}
            unit={<MathInline math="\mu\mathrm{g/m}^3" />}
            subtitle={<span>Train MAE: {calibrationResult.train_metrics.mae.toFixed(3)} <MathInline math="\mu\mathrm{g/m}^3" /></span>}
            badge={{ text: 'L1 Norm Error', variant: 'default' }}
          />

          <MetricCard
            label={<MathInline math="\text{Validation } R^2" />}
            value={`${(calibrationResult.val_metrics.r2 * 100).toFixed(1)}%`}
            subtitle={<span>Train <MathInline math="R^2" />: {(calibrationResult.train_metrics.r2 * 100).toFixed(1)}%</span>}
            badge={{
              text: calibrationResult.val_metrics.r2 >= 0.9 ? 'Strong Fit' : 'Moderate',
              variant: calibrationResult.val_metrics.r2 >= 0.9 ? 'success' : 'warning',
            }}
          />

          <MetricCard
            label="Validation Mean Bias (MBE)"
            value={calibrationResult.val_metrics.mbe.toFixed(3)}
            unit={<MathInline math="\mu\mathrm{g/m}^3" />}
            subtitle={
              Math.abs(calibrationResult.val_metrics.mbe) < 0.2
                ? 'Unbiased Mean'
                : calibrationResult.val_metrics.mbe > 0
                ? 'Overestimating'
                : 'Underestimating'
            }
            badge={{
              text: Math.abs(calibrationResult.val_metrics.mbe) < 0.5 ? 'Well Centered' : 'Drift Offset',
              variant: Math.abs(calibrationResult.val_metrics.mbe) < 0.5 ? 'teal' : 'warning',
            }}
          />
        </div>
      )}

      {/* 4. Interactive Visualization Tabs */}
      {calibrationResult && (
        <Card>
          {/* Tab Navigation Header */}
          <div className="flex flex-wrap items-center justify-between border-b border-slate-200/80 pb-4 mb-6 gap-4">
            <div className="flex flex-wrap items-center gap-1.5 p-1 bg-slate-100/80 rounded-xl">
              <button
                type="button"
                onClick={() => setActiveTab('scatter')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  activeTab === 'scatter'
                    ? 'bg-white text-teal-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Measured vs. Predicted
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('timeline')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  activeTab === 'timeline'
                    ? 'bg-white text-teal-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Validation Timeline
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('residuals')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  activeTab === 'residuals'
                    ? 'bg-white text-teal-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Residual Distribution
              </button>
              <button
                type="button"
                onClick={() => setActiveTab('weights')}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                  activeTab === 'weights'
                    ? 'bg-white text-teal-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                Feature Sensitivities
              </button>
              {calibrationResult.pca_variance && (
                <button
                  type="button"
                  onClick={() => setActiveTab('variance')}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all cursor-pointer ${
                    activeTab === 'variance'
                      ? 'bg-white text-teal-900 shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  PCA Scree Plot
                </button>
              )}
            </div>

            <div className="text-[11px] text-slate-500 font-mono">
              Experiment ID: <span className="font-bold text-slate-700">{calibrationResult.experiment_id}</span>
            </div>
          </div>

          <AnimatePresence mode="wait">
            <motion.div
              key={activeTab}
              variants={tabContentVariants}
              initial="initial"
              animate="animate"
              exit="exit"
            >
              {/* TAB 1: Measured vs Predicted Scatter */}
              {activeTab === 'scatter' && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-xs text-slate-500">
                    <span>
                      Validation observations (downsampled to 180 representative hours for responsive rendering).
                      Dashed line represents ideal 1:1 agreement (<MathInline math="y = \hat{y}" />).
                    </span>
                    <span className="font-mono text-[11px] text-teal-700 font-semibold">
                      <MathInline math="R^2 =" /> {(calibrationResult.val_metrics.r2 * 100).toFixed(2)}%
                    </span>
                  </div>

                  <div className="h-[380px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis
                          type="number"
                          dataKey="actual"
                          name="Reference C6H6"
                          unit=" &mu;g/m&sup3;"
                          domain={[0, 60]}
                          tick={{ fill: '#64748b', fontSize: 11 }}
                          stroke="#cbd5e1"
                          label={{
                            value: 'Certified Reference Analyzer: C6H6(GT) [µg/m³]',
                            position: 'insideBottom',
                            offset: -12,
                            fill: '#475569',
                            fontSize: 11,
                          }}
                        />
                        <YAxis
                          type="number"
                          dataKey="predicted"
                          name="Calibrated Prediction"
                          unit=" &mu;g/m&sup3;"
                          domain={[0, 60]}
                          tick={{ fill: '#64748b', fontSize: 11 }}
                          stroke="#cbd5e1"
                          label={{
                            value: 'Calibrated Sensor Output: y_hat [µg/m³]',
                            angle: -90,
                            position: 'insideLeft',
                            offset: 10,
                            fill: '#475569',
                            fontSize: 11,
                          }}
                        />
                        <Tooltip
                          cursor={{ strokeDasharray: '3 3' }}
                          content={({ payload }) => {
                            if (!payload || payload.length === 0) return null;
                            const data = payload[0].payload;
                            return (
                              <div className="bg-slate-900 text-white p-3 rounded-xl shadow-xl text-xs space-y-1 font-mono border border-slate-800">
                                <div className="text-slate-400 text-[10px] pb-1 border-b border-slate-700">
                                  {data.timestamp}
                                </div>
                                <div className="text-teal-300 font-bold">
                                  Actual: {data.actual} <MathInline math="\mu\mathrm{g/m}^3" />
                                </div>
                                <div className="text-sky-300">
                                  Predicted: {data.predicted} <MathInline math="\mu\mathrm{g/m}^3" />
                                </div>
                                <div className="text-slate-300 text-[11px]">
                                  Residual: {data.residual} <MathInline math="\mu\mathrm{g/m}^3" />
                                </div>
                              </div>
                            );
                          }}
                        />
                        <ReferenceLine
                          segment={[{ x: 0, y: 0 }, { x: 60, y: 60 }]}
                          stroke="#94a3b8"
                          strokeDasharray="4 4"
                          strokeWidth={1.5}
                          label={{ value: '1:1 Agreement', position: 'insideTopLeft', fill: '#94a3b8', fontSize: 10 }}
                        />
                        <Scatter
                          name="Validation Points"
                          data={calibrationResult.scatter_sample}
                          fill="#0d9488"
                          opacity={0.65}
                        />
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* TAB 2: Validation Trajectory Timeline */}
              {activeTab === 'timeline' && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-xs text-slate-500">
                    <span>
                      Oct 27, 2004 to Jan 16, 2005 (1,798 hours contiguous validation partition).
                    </span>
                    <div className="flex items-center gap-4 text-[11px] font-semibold">
                      <span className="flex items-center gap-1.5 text-slate-700">
                        <span className="w-2.5 h-2.5 rounded-full bg-slate-700" /> Reference Benzene
                      </span>
                      <span className="flex items-center gap-1.5 text-teal-600">
                        <span className="w-2.5 h-2.5 rounded-full bg-teal-500" /> Calibrated Output
                      </span>
                    </div>
                  </div>

                  <div className="h-[380px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={calibrationResult.timeline_sample} margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis
                          dataKey="timestamp"
                          tick={{ fill: '#64748b', fontSize: 10 }}
                          stroke="#cbd5e1"
                          minTickGap={40}
                        />
                        <YAxis
                          tick={{ fill: '#64748b', fontSize: 11 }}
                          stroke="#cbd5e1"
                          unit=" &mu;g/m&sup3;"
                        />
                        <Tooltip
                          content={({ payload }) => {
                            if (!payload || payload.length === 0) return null;
                            const data = payload[0].payload;
                            return (
                              <div className="bg-slate-900 text-white p-3 rounded-xl shadow-xl text-xs space-y-1 font-mono border border-slate-800">
                                <div className="text-slate-400 text-[10px] pb-1 border-b border-slate-700">
                                  {data.timestamp}
                                </div>
                                <div className="text-slate-200">
                                  Reference: <span className="font-bold">{data.actual} <MathInline math="\mu\mathrm{g/m}^3" /></span>
                                </div>
                                <div className="text-teal-400">
                                  Predicted: <span className="font-bold">{data.predicted} <MathInline math="\mu\mathrm{g/m}^3" /></span>
                                </div>
                                <div className="text-slate-400 text-[10px]">
                                  Error: {(data.actual - data.predicted).toFixed(2)} <MathInline math="\mu\mathrm{g/m}^3" />
                                </div>
                              </div>
                            );
                          }}
                        />
                        <Line
                          type="monotone"
                          dataKey="actual"
                          stroke="#334155"
                          strokeWidth={1.5}
                          dot={false}
                          name="Reference C6H6"
                        />
                        <Line
                          type="monotone"
                          dataKey="predicted"
                          stroke="#0d9488"
                          strokeWidth={1.5}
                          dot={false}
                          name="Model Prediction"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* TAB 3: Residual Distribution */}
              {activeTab === 'residuals' && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-xs text-slate-500">
                    <span>
                      Validation Residuals <MathInline math="e = y - \hat{y}" /> grouped into 20 histogram bins vs. fitted theoretical Normal distribution.
                    </span>
                    <span className="font-mono text-[11px] text-slate-700">
                      Mean Bias: {calibrationResult.val_metrics.mbe.toFixed(3)} <MathInline math="\mu\mathrm{g/m}^3" />
                    </span>
                  </div>

                  <div className="h-[380px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={calibrationResult.residuals_distribution} margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis
                          dataKey="bin_center"
                          tick={{ fill: '#64748b', fontSize: 10 }}
                          stroke="#cbd5e1"
                          unit=" &mu;g/m&sup3;"
                          label={{
                            value: 'Residual Error (e = y - y_hat) [µg/m³]',
                            position: 'insideBottom',
                            offset: -12,
                            fill: '#475569',
                            fontSize: 11,
                          }}
                        />
                        <YAxis
                          tick={{ fill: '#64748b', fontSize: 11 }}
                          stroke="#cbd5e1"
                          name="Density"
                        />
                        <Tooltip
                          content={({ payload }) => {
                            if (!payload || payload.length === 0) return null;
                            const data = payload[0].payload;
                            return (
                              <div className="bg-slate-900 text-white p-3 rounded-xl shadow-xl text-xs space-y-1 font-mono border border-slate-800">
                                <div className="text-slate-400 text-[10px]">
                                  Bin: [{data.bin_start}, {data.bin_end}] &mu;g/m&sup3;
                                </div>
                                <div className="text-teal-300 font-bold">
                                  Count: {data.count} observations
                                </div>
                                <div className="text-slate-300 text-[11px]">
                                  Observed Density: {data.density}
                                </div>
                                <div className="text-sky-300 text-[11px]">
                                  Gaussian Normal: {data.normal_density}
                                </div>
                              </div>
                            );
                          }}
                        />
                        <ReferenceLine x={0} stroke="#f43f5e" strokeWidth={1.5} label={{ value: 'Zero Error', fill: '#f43f5e', fontSize: 10 }} />
                        <Bar dataKey="density" fill="#0d9488" opacity={0.7} radius={[4, 4, 0, 0]} name="Residual Density" />
                        <Line type="monotone" dataKey="normal_density" stroke="#0284c7" strokeWidth={2} dot={false} name="Theoretical Normal" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}

              {/* TAB 4: Feature Sensitivities */}
              {activeTab === 'weights' && (
                <div className="space-y-6">
                  <div className="flex items-center justify-between text-xs text-slate-500">
                    <span>
                      Physical weights back-projected via{' '}
                      <span className="bg-slate-100 px-1 py-0.5 rounded text-teal-700 font-mono font-bold">
                        <MathInline math="\mathbf{w}_{\mathrm{feature}} = D_\sigma^{-1} V_k \mathbf{w}_{\mathrm{pca}}" />
                      </span>
                      . Represents physical <MathInline math="\mu\mathrm{g/m}^3" /> change per sensor reading unit.
                    </span>
                    <span className="font-mono text-[11px] text-slate-700">
                      Model Intercept: {calibrationResult.intercept.toFixed(4)} <MathInline math="\mu\mathrm{g/m}^3" />
                    </span>
                  </div>

                  <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                    {/* Bar chart of feature weights */}
                    <div className="lg:col-span-7 h-[340px]">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={calibrationResult.feature_sensitivities}
                          layout="vertical"
                          margin={{ top: 10, right: 30, bottom: 10, left: 90 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                          <XAxis type="number" stroke="#cbd5e1" tick={{ fill: '#64748b', fontSize: 10 }} />
                          <YAxis
                            type="category"
                            dataKey="feature"
                            stroke="#cbd5e1"
                            tick={{ fill: '#334155', fontSize: 11, fontWeight: 600 }}
                          />
                          <Tooltip
                            content={({ payload }) => {
                              if (!payload || payload.length === 0) return null;
                              const data = payload[0].payload;
                              return (
                                <div className="bg-slate-900 text-white p-3 rounded-xl shadow-xl text-xs space-y-1 font-mono border border-slate-800">
                                  <div className="font-bold text-teal-300">{data.feature}</div>
                                  <div>Weight: {data.weight.toFixed(6)}</div>
                                  <div className="text-[10px] text-slate-400">{data.unit}</div>
                                </div>
                              );
                            }}
                          />
                          <ReferenceLine x={0} stroke="#94a3b8" />
                          <Bar dataKey="weight" radius={[0, 4, 4, 0]}>
                            {calibrationResult.feature_sensitivities.map((entry, index) => (
                              <Cell
                                key={`cell-${index}`}
                                fill={entry.weight >= 0 ? '#0d9488' : '#f43f5e'}
                                opacity={0.8}
                              />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    {/* Table of feature weights */}
                    <div className="lg:col-span-5 bg-slate-50/70 border border-slate-200/80 rounded-xl overflow-hidden text-xs">
                      <div className="px-4 py-3 bg-slate-100/80 border-b border-slate-200 font-bold text-slate-700 flex items-center justify-between">
                        <span>Feature Attribution Matrix</span>
                        <Badge variant="teal" size="sm">{calibrationResult.feature_count} Inputs</Badge>
                      </div>
                      <div className="divide-y divide-slate-100 max-h-[300px] overflow-y-auto">
                        {calibrationResult.feature_sensitivities.map((item) => (
                          <div key={item.feature} className="p-3 flex items-center justify-between hover:bg-white transition-colors">
                            <div>
                              <div className="font-mono font-bold text-slate-800">{item.feature}</div>
                              <div className="text-[10px] text-slate-400">{item.unit}</div>
                            </div>
                            <div className="text-right font-mono">
                              <span className={`font-bold ${item.weight >= 0 ? 'text-teal-700' : 'text-rose-600'}`}>
                                {item.weight >= 0 ? `+${item.weight.toFixed(6)}` : item.weight.toFixed(6)}
                              </span>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* TAB 5: PCA Explained Variance Spectrum */}
              {activeTab === 'variance' && calibrationResult.pca_variance && (
                <div className="space-y-4">
                  <div className="flex items-center justify-between text-xs text-slate-500">
                    <span>
                      Scree plot of eigenvalues computed via Jacobi/SVD on standardized training covariance matrix.
                    </span>
                    <span className="font-mono text-[11px] text-teal-700 font-semibold">
                      Retained k={effectiveK}: captures{' '}
                      {calibrationResult.pca_variance[effectiveK - 1]?.cum_variance_pct}% variance
                    </span>
                  </div>

                  <div className="h-[380px] w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={calibrationResult.pca_variance} margin={{ top: 20, right: 30, bottom: 20, left: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis
                          dataKey="k"
                          tick={{ fill: '#64748b', fontSize: 11 }}
                          stroke="#cbd5e1"
                          label={{
                            value: 'Principal Component Index (k)',
                            position: 'insideBottom',
                            offset: -12,
                            fill: '#475569',
                            fontSize: 11,
                          }}
                        />
                        <YAxis
                          tick={{ fill: '#64748b', fontSize: 11 }}
                          stroke="#cbd5e1"
                          unit="%"
                        />
                        <Tooltip
                          content={({ payload }) => {
                            if (!payload || payload.length === 0) return null;
                            const data = payload[0].payload;
                            return (
                              <div className="bg-slate-900 text-white p-3 rounded-xl shadow-xl text-xs space-y-1 font-mono border border-slate-800">
                                <div className="font-bold text-teal-300">PC {data.k}</div>
                                <div>Component Variance: {data.variance_pct}%</div>
                                <div className="text-sky-300">Cumulative: {data.cum_variance_pct}%</div>
                                <div className="text-[10px] text-slate-400">
                                  {data.k <= effectiveK ? 'RETAINED in calibration' : 'DISCARDED (Regularized noise)'}
                                </div>
                              </div>
                            );
                          }}
                        />
                        <Legend />
                        <Bar
                          dataKey="variance_pct"
                          name="Component Variance (%)"
                          radius={[4, 4, 0, 0]}
                        >
                          {calibrationResult.pca_variance.map((entry) => (
                            <Cell
                              key={`var-cell-${entry.k}`}
                              fill={entry.k <= effectiveK ? '#0d9488' : '#cbd5e1'}
                            />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </Card>
      )}

      {/* 5. Sealed Historical Benchmark Matrix Reference */}
      <Card
        title={
          <div className="flex items-center gap-2.5">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <span>Frozen Benchmark Matrix (Historical Protocol Record)</span>
          </div>
        }
        subtitle="Pre-recorded benchmark from reports/experiment_results.csv. Interactive exploration above NEVER touches or modifies the final test partition."
      >
        <div className="space-y-4">
          <Alert variant="info" title="Academic Evaluation Protocol">
            Under strict chronological experimental rules, <strong><MathInline math="k^* = 8" /> (Full OLS)</strong> was the model selected based on lowest
            validation error (RMSE = 2.024 <MathInline math="\mu\mathrm{g/m}^3" />). The lower test error of <strong><MathInline math="k = 3" /></strong> (RMSE = 2.415 <MathInline math="\mu\mathrm{g/m}^3" />) is a post-hoc
            regularization observation demonstrating how discarding noisy collinear components provides resistance against seasonal drift.
          </Alert>

          <div className="overflow-x-auto border border-slate-200/80 rounded-xl">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 border-b border-slate-200 font-semibold text-slate-700 uppercase tracking-wider">
                <tr>
                  <th className="py-3 px-4">Model (k)</th>
                  <th className="py-3 px-4">Cum. Var %</th>
                  <th className="py-3 px-4">Train RMSE</th>
                  <th className="py-3 px-4">Val RMSE</th>
                  <th className="py-3 px-4">Test RMSE</th>
                  <th className="py-3 px-4">Test <MathInline math="R^2" /></th>
                  <th className="py-3 px-4">Experimental Role</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100 font-mono">
                {benchmarkMatrix.map((row) => {
                  const isSelected = row.k === 8;
                  const isPostHoc = row.k === 3;
                  return (
                    <tr
                      key={row.k}
                      className={`hover:bg-slate-50/80 transition-colors ${
                        isSelected ? 'bg-teal-50/50 font-bold' : isPostHoc ? 'bg-sky-50/40' : ''
                      }`}
                    >
                      <td className="py-3 px-4 font-sans font-medium flex items-center gap-2">
                        {row.k === 8 ? 'Full OLS / PCR (k=8)' : `PCR (k=${row.k})`}
                        {isSelected && <Badge variant="teal" size="sm">Selected k*</Badge>}
                        {isPostHoc && <Badge variant="info" size="sm">Post-Hoc Best</Badge>}
                      </td>
                      <td className="py-3 px-4">{row.cum_variance_pct.toFixed(1)}%</td>
                      <td className="py-3 px-4 text-slate-600">{row.train_rmse.toFixed(3)}</td>
                      <td className="py-3 px-4 text-slate-800">{row.val_rmse.toFixed(3)}</td>
                      <td className="py-3 px-4 font-bold text-slate-900">{row.test_rmse.toFixed(3)}</td>
                      <td className="py-3 px-4">{(row.test_r2 * 100).toFixed(1)}%</td>
                      <td className="py-3 px-4 font-sans">
                        <span className="text-[11px] text-slate-500">{row.role}</span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      </Card>

      {/* 6. Collapsible Mathematical Rigor & Formulations */}
      <div className="bg-white border border-slate-200/90 rounded-2xl overflow-hidden shadow-xs">
        <button
          type="button"
          onClick={() => setMathOpen(!mathOpen)}
          className="w-full px-6 py-4.5 flex items-center justify-between text-left hover:bg-slate-50 transition-colors cursor-pointer"
        >
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-teal-50 border border-teal-200 rounded-xl text-teal-700">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-slate-900 text-sm">Mathematical Formulations &amp; Algorithmic Derivations</h3>
              <p className="text-xs text-slate-500 mt-0.5">
                Inspect QR factorization, 5-step PCR derivation, physical weight back-projection, and viva defense concepts.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2 text-slate-400">
            <span className="text-xs font-semibold">{mathOpen ? 'Hide Derivations' : 'View Derivations'}</span>
            {mathOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </div>
        </button>

        <AnimatePresence>
          {mathOpen && (
            <motion.div
              variants={accordionVariants}
              initial="collapsed"
              animate="expanded"
              exit="collapsed"
              className="overflow-hidden"
            >
              <div className="px-6 py-5 border-t border-slate-100 space-y-6 text-xs text-slate-600 leading-relaxed bg-slate-50/30">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Card 1: QR vs Normal Equations */}
                  <div className="bg-white border border-slate-200/90 rounded-xl p-4.5 space-y-3 shadow-2xs">
                    <h4 className="font-bold text-slate-900 text-xs uppercase tracking-wider text-teal-800 font-mono">
                      1. QR Decomposition vs. Normal Equations
                    </h4>
                    <p>
                      Solving least-squares via normal equations requires forming the Gram matrix <MathInline math="X^\mathsf{T} X" />.
                      This squares the matrix condition number: <MathInline math="\kappa(X^\mathsf{T} X) = \kappa(X)^2" />,
                      causing catastrophic cancellation on collinear sensor arrays.
                    </p>
                    <div className="p-3 bg-slate-900 text-teal-300 font-mono text-[11px] rounded-xl border border-slate-800 space-y-1">
                      <div><MathInline math="X = QR \quad \text{(Thin QR via Modified Gram-Schmidt)}" /></div>
                      <div><MathInline math="R\mathbf{w} = Q^\mathsf{T} \mathbf{y} \quad \text{(Back-substitution in } \mathcal{O}(d^2)\text{)}" /></div>
                      <div className="text-slate-400 text-[10px] mt-1 pt-1 border-t border-slate-800">
                        <MathInline math="\kappa(R) = \kappa(X) \implies" /> Zero condition squaring
                      </div>
                    </div>
                  </div>

                  {/* Card 2: 5-Step PCR Derivation */}
                  <div className="bg-white border border-slate-200/90 rounded-xl p-4.5 space-y-3 shadow-2xs">
                    <h4 className="font-bold text-slate-900 text-xs uppercase tracking-wider text-teal-800 font-mono">
                      2. 5-Step Principal Component Regression (PCR)
                    </h4>
                    <p>
                      PCR decomposes standardized training features via Thin Singular Value Decomposition (SVD):
                    </p>
                    <div className="p-3 bg-slate-900 text-sky-300 font-mono text-[11px] rounded-xl border border-slate-800 space-y-1.5">
                      <div><MathInline math="1.\; Z = (X - \mathbf{1}\boldsymbol{\mu}^\mathsf{T}) D_\sigma^{-1}" /></div>
                      <div><MathInline math="2.\; Z = U \Sigma V^\mathsf{T} \quad \text{(Thin SVD)}" /></div>
                      <div><MathInline math="3.\; T_k = Z V_k \quad \text{(Orthogonal Principal Scores)}" /></div>
                      <div><MathInline math="4.\; \hat{\mathbf{y}} = w_0 + T_k \mathbf{w}_{\mathrm{pca}} \quad \text{(QR Least Squares)}" /></div>
                      <div className="text-teal-300 font-bold pt-1 border-t border-slate-800">
                        <MathInline math="5.\; \mathbf{w}_{\mathrm{feature}} = D_\sigma^{-1} V_k \mathbf{w}_{\mathrm{pca}} \in \mathbb{R}^d" />
                      </div>
                    </div>
                    <p className="text-[11px]">
                      Step 5 maps abstract latent weights back into physical sensor resistance units (<MathInline math="\mu\mathrm{g/m}^3" /> per unit).
                    </p>
                  </div>

                  {/* Card 3: Concept -> Purpose -> Outcome */}
                  <div className="bg-white border border-slate-200/90 rounded-xl p-4.5 space-y-3 shadow-2xs">
                    <h4 className="font-bold text-slate-900 text-xs uppercase tracking-wider text-teal-800 font-mono">
                      3. Concept &rarr; Purpose &rarr; Outcome Triad
                    </h4>
                    <div className="space-y-2">
                      <div>
                        <span className="font-bold text-slate-800">Concept:</span> Orthogonal subspace projection via spectral decomposition of covariance matrix.
                      </div>
                      <div>
                        <span className="font-bold text-slate-800">Purpose:</span> Decouple highly correlated MOX sensor channels (<MathInline math="r > 0.85" />) without discarding physical sensors.
                      </div>
                      <div>
                        <span className="font-bold text-slate-800">Outcome:</span> Stable, regularized weights that do not amplify high-frequency measurement noise when ambient weather drifts.
                      </div>
                    </div>
                  </div>

                  {/* Card 4: Academic Viva Rule */}
                  <div className="bg-white border border-slate-200/90 rounded-xl p-4.5 space-y-3 shadow-2xs">
                    <h4 className="font-bold text-slate-900 text-xs uppercase tracking-wider text-teal-800 font-mono">
                      4. The Academic Model Selection Protocol
                    </h4>
                    <p>
                      Why did we select <MathInline math="k^* = 8" /> if <MathInline math="k = 3" /> had a lower test error?
                    </p>
                    <p className="text-[11px] bg-amber-50 border border-amber-200/80 p-3 rounded-xl text-amber-900">
                      <strong>The Examiner Question:</strong> If you use test set error to select <MathInline math="k = 3" />, you commit <em>test-set data snooping</em> (target leakage).
                      Under strict forward validation, <MathInline math="k^* = 8" /> had the lowest validation RMSE (2.0242). The superior generalization of <MathInline math="k = 3" /> on the test set is a post-hoc
                      empirical discovery illustrating that discarding trailing collinear components regularizes against seasonal winter drift.
                    </p>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};
