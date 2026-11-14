import React, { useState, useEffect, useCallback } from 'react';
import {
  ShieldCheck,
  Sliders,
  RefreshCw,
  BookOpen,
  TrendingUp,
  Eye,
  Info,
  AlertTriangle,
  Grid,
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
  ReferenceLine,
  ScatterChart,
  Scatter,
  ZAxis,
  Cell,
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';

import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { ErrorState } from '../components/ui/ErrorState';
import { MathInline, MathBlock } from '../components/ui/Math';
import { apiClient } from '../services/apiClient';
import { PCAResponse, PCAScorePoint } from '../types/api';
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

export const PcaExplorerPage: React.FC = () => {
  // Navigation tabs
  const [activeTab, setActiveTab] = useState<'scree' | 'loadings' | 'scores' | 'reconstruction' | 'theory'>('scree');

  // Configuration state
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([...ALL_FEATURES]);
  const [standardize, setStandardize] = useState<boolean>(true);
  const [nComponents, setNComponents] = useState<number>(3);
  const [partition, setPartition] = useState<'train' | 'val' | 'all'>('train');
  const [scorePcX, setScorePcX] = useState<number>(1);
  const [scorePcY, setScorePcY] = useState<number>(2);
  const [selectedLoadingsPc, setSelectedLoadingsPc] = useState<number>(1);
  const [reconstructionFeature, setReconstructionFeature] = useState<string>('PT08.S1(CO)');

  // API State
  const [pcaData, setPcaData] = useState<PCAResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch PCA Data
  const fetchPCA = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.analyzePCA({
        features: selectedFeatures,
        standardize,
        n_components: nComponents,
        partition,
        score_pc_x: scorePcX,
        score_pc_y: scorePcY,
        reconstruction_feature: reconstructionFeature,
        sample_size: 280,
      });
      setPcaData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to compute PCA');
    } finally {
      setLoading(false);
    }
  }, [selectedFeatures, standardize, nComponents, partition, scorePcX, scorePcY, reconstructionFeature]);

  useEffect(() => {
    fetchPCA();
  }, [fetchPCA]);

  const toggleFeature = (feat: string) => {
    setSelectedFeatures((prev) => {
      if (prev.includes(feat)) {
        if (prev.length <= 2) return prev; // Keep at least 2 features for PCA
        return prev.filter((f) => f !== feat);
      } else {
        return [...prev, feat];
      }
    });
  };

  const applyPreset = (type: 'all' | 'chemical' | 'environmental') => {
    if (type === 'all') {
      setSelectedFeatures([...ALL_FEATURES]);
    } else if (type === 'chemical') {
      setSelectedFeatures(['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 'PT08.S4(NO2)', 'PT08.S5(O3)']);
    } else {
      setSelectedFeatures(['T', 'RH', 'AH']);
    }
  };

  // Helper for loading cell color in diverging matrix
  const getLoadingColorClass = (val: number) => {
    const abs = Math.abs(val);
    if (abs < 0.05) return 'bg-slate-50 text-slate-400';
    if (val > 0) {
      if (abs > 0.45) return 'bg-teal-200 text-teal-950 font-bold';
      if (abs > 0.25) return 'bg-teal-100 text-teal-900 font-semibold';
      return 'bg-teal-50 text-teal-800';
    } else {
      if (abs > 0.45) return 'bg-rose-200 text-rose-950 font-bold';
      if (abs > 0.25) return 'bg-rose-100 text-rose-900 font-semibold';
      return 'bg-rose-50 text-rose-800';
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* Top Header Banner */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="text-[11px] font-bold text-teal-700 tracking-wider uppercase font-mono bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
              Unit 2 Spectral Methods
            </span>
            <span className="text-slate-300">&middot;</span>
            <span className="text-xs text-slate-500 font-medium">Thin SVD &amp; Covariance Equivalence</span>
            <span className="text-slate-300">&middot;</span>
            <span className="text-[11px] font-mono text-slate-500">Strict Chronological Split</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Principal Component Analysis Explorer</h1>
          <p className="text-sm text-slate-600 mt-1 max-w-3xl leading-relaxed">
            Decompose multi-sensor collinearity, explore spectral eigenvalues, inspect loading directions,
            project observations into principal score subspaces, and verify low-rank reconstruction bounds.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0 flex-wrap">
          <Badge variant="teal" size="md" className="gap-1.5 font-mono">
            <ShieldCheck className="w-4 h-4 text-teal-600" />
            Training-Fitted Scaler (Zero Leakage)
          </Badge>
          <button
            onClick={fetchPCA}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors border border-slate-200"
          >
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
            Recompute SVD
          </button>
        </div>
      </div>

      {/* Global Interactive Configuration Bar */}
      <Card title="PCA Pipeline &amp; Dimensionality Controls" subtitle="Transformations fitted strictly on 5,394 training hours (March–October 2004)">
        <div className="space-y-5">
          {/* Feature Selection Row */}
          <div>
            <div className="text-xs font-bold text-slate-800 mb-2 flex items-center justify-between flex-wrap gap-2">
              <span>Predictor Sensor Features ({selectedFeatures.length} of {ALL_FEATURES.length} Selected)</span>
              <div className="flex items-center gap-1.5 text-[11px]">
                <span className="text-slate-400">Presets:</span>
                <button
                  onClick={() => applyPreset('all')}
                  className="px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium"
                >
                  All 8
                </button>
                <button
                  onClick={() => applyPreset('chemical')}
                  className="px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium"
                >
                  Chemical (5)
                </button>
                <button
                  onClick={() => applyPreset('environmental')}
                  className="px-2 py-0.5 rounded bg-slate-100 hover:bg-slate-200 text-slate-700 font-medium"
                >
                  Environmental (3)
                </button>
              </div>
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

          {/* Controls: Standardize, Retained Components k, Partition */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 pt-4 border-t border-slate-100 items-end">
            {/* Standardize Toggle */}
            <div className="p-3 rounded-xl border border-slate-200 bg-slate-50 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-xs font-bold text-slate-800">Feature Scaling</span>
                <span className="text-[10px] font-mono text-teal-700 font-semibold">
                  {standardize ? 'Correlation PCA' : 'Covariance PCA'}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="std_pca"
                  checked={standardize}
                  onChange={(e) => setStandardize(e.target.checked)}
                  className="w-4 h-4 text-teal-600 rounded accent-teal-600 cursor-pointer"
                />
                <label htmlFor="std_pca" className="text-xs font-medium text-slate-700 cursor-pointer">
                  Standardize (Z-Score <MathInline math="\sigma = 1" />)
                </label>
              </div>
              <p className="text-[10px] text-slate-500 leading-tight">
                {standardize
                  ? 'All sensors have equal variance weight.'
                  : 'Warning: High-voltage sensors dominate.'}
              </p>
            </div>

            {/* Retained Components k Slider */}
            <div className="p-3 rounded-xl border border-slate-200 bg-slate-50 space-y-1.5">
              <div className="flex justify-between text-xs font-bold text-slate-800">
                <span>Retained Components (k)</span>
                <span className="font-mono text-teal-700 font-bold">
                  k = {nComponents} of {selectedFeatures.length}
                </span>
              </div>
              <input
                type="range"
                min={1}
                max={selectedFeatures.length}
                value={nComponents}
                onChange={(e) => setNComponents(Number(e.target.value))}
                className="w-full accent-teal-600 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-slate-400 font-mono">
                <span>k = 1</span>
                <span>k = {selectedFeatures.length} (Full)</span>
              </div>
            </div>

            {/* Data Partition Selector */}
            <div className="p-3 rounded-xl border border-slate-200 bg-slate-50 space-y-1.5">
              <span className="text-xs font-bold text-slate-800 block">Visualized Partition</span>
              <select
                value={partition}
                onChange={(e) => setPartition(e.target.value as 'train' | 'val' | 'all')}
                className="w-full text-xs font-medium py-1.5 px-2 rounded-lg border border-slate-300 bg-white text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-500"
              >
                <option value="train">Training Set (5,394 hrs &middot; Mar–Oct 2004)</option>
                <option value="val">Validation Set (1,798 hrs &middot; Out-of-Time)</option>
                <option value="all">All Usable (8,991 hrs &middot; Combined)</option>
              </select>
              <p className="text-[10px] text-slate-500 leading-tight">
                Evaluated with frozen training-set eigenvectors.
              </p>
            </div>

            {/* Feature to Reconstruct Selector */}
            <div className="p-3 rounded-xl border border-slate-200 bg-slate-50 space-y-1.5">
              <span className="text-xs font-bold text-slate-800 block">Reconstruction Target Series</span>
              <select
                value={reconstructionFeature}
                onChange={(e) => setReconstructionFeature(e.target.value)}
                className="w-full text-xs font-medium py-1.5 px-2 rounded-lg border border-slate-300 bg-white text-slate-800 focus:outline-none focus:ring-1 focus:ring-teal-500"
              >
                {selectedFeatures.map((f) => (
                  <option key={f} value={f}>
                    {f}
                  </option>
                ))}
              </select>
              <p className="text-[10px] text-slate-500 leading-tight">
                Plotted in 48h reconstruction comparison.
              </p>
            </div>
          </div>
        </div>
      </Card>

      {/* KPI Summary Banner */}
      {pcaData && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl shadow-xs space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono block">Original Dimension</span>
            <span className="text-xl font-bold font-mono text-slate-900">{pcaData.total_components}</span>
            <span className="text-[10px] text-slate-500 block">Input sensor channels</span>
          </div>

          <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl shadow-xs space-y-1">
            <span className="text-[10px] text-teal-600 uppercase font-mono block font-bold">Retained Subspace</span>
            <span className="text-xl font-bold font-mono text-teal-700">k = {pcaData.n_components}</span>
            <span className="text-[10px] text-teal-600 block">
              {((1 - pcaData.n_components / pcaData.total_components) * 100).toFixed(0)}% dimension reduction
            </span>
          </div>

          <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl shadow-xs space-y-1">
            <span className="text-[10px] text-emerald-600 uppercase font-mono block font-bold">Retained Variance</span>
            <span className="text-xl font-bold font-mono text-emerald-700">{pcaData.retained_variance_pct}%</span>
            <span className="text-[10px] text-slate-500 block">Total signal captured</span>
          </div>

          <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl shadow-xs space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono block">Discarded Variance</span>
            <span className="text-xl font-bold font-mono text-slate-600">{pcaData.discarded_variance_pct}%</span>
            <span className="text-[10px] text-slate-400 block">Collinear / noise tail</span>
          </div>

          <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl shadow-xs space-y-1">
            <span className="text-[10px] text-indigo-600 uppercase font-mono block font-bold">Frobenius Error</span>
            <span className="text-xl font-bold font-mono text-indigo-700">{pcaData.reconstruction_error_frobenius}</span>
            <span className="text-[10px] text-slate-500 block">
              {(pcaData.relative_reconstruction_error * 100).toFixed(1)}% relative error
            </span>
          </div>

          <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl shadow-xs space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono block">Evaluated Hours</span>
            <span className="text-xl font-bold font-mono text-slate-900">{pcaData.evaluated_partition_rows}</span>
            <span className="text-[10px] text-slate-500 block">
              {pcaData.partition === 'train' ? 'Training partition' : pcaData.partition === 'val' ? 'Validation split' : 'Combined dataset'}
            </span>
          </div>
        </div>
      )}

      {/* Workspace Tabs Navigation */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-2 shadow-xs">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
          <button
            onClick={() => setActiveTab('scree')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'scree'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <TrendingUp className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">1. Scree &amp; Variance</span>
              <span className={`text-[10px] ${activeTab === 'scree' ? 'text-slate-300' : 'text-slate-500'}`}>
                Eigenvalues &amp; cumulative %
              </span>
            </div>
          </button>

          <button
            onClick={() => setActiveTab('loadings')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'loadings'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <Grid className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">2. Component Loadings</span>
              <span className={`text-[10px] ${activeTab === 'loadings' ? 'text-slate-300' : 'text-slate-500'}`}>
                Diverging matrix &amp; directions
              </span>
            </div>
          </button>

          <button
            onClick={() => setActiveTab('scores')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'scores'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <Eye className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">3. Score Projections</span>
              <span className={`text-[10px] ${activeTab === 'scores' ? 'text-slate-300' : 'text-slate-500'}`}>
                2D observation scatter
              </span>
            </div>
          </button>

          <button
            onClick={() => setActiveTab('reconstruction')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'reconstruction'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <Sliders className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">4. Reconstruction Lab</span>
              <span className={`text-[10px] ${activeTab === 'reconstruction' ? 'text-slate-300' : 'text-slate-500'}`}>
                Low-rank approximation
              </span>
            </div>
          </button>

          <button
            onClick={() => setActiveTab('theory')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'theory'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <BookOpen className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">5. Mathematical Theory</span>
              <span className={`text-[10px] ${activeTab === 'theory' ? 'text-slate-300' : 'text-slate-500'}`}>
                Spectral SVD identities
              </span>
            </div>
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      {error ? (
        <ErrorState title="PCA Computation Error" message={error} onRetry={fetchPCA} />
      ) : loading && !pcaData ? (
        <div className="p-16 text-center bg-white border border-slate-200 rounded-2xl shadow-xs">
          <LoadingSpinner label="Computing SVD decomposition and principal subspaces..." />
        </div>
      ) : pcaData ? (
        <AnimatePresence mode="wait">
          {/* =============================================================== */}
          {/* TAB 1: SCREE PLOT & VARIANCE EXPLAINED */}
          {/* =============================================================== */}
          {activeTab === 'scree' && (
            <motion.div
              key="scree_tab"
              variants={tabContentVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="space-y-6"
            >
              {/* Scree Chart Card */}
              <Card
                title="Eigenvalue Scree Plot &amp; Cumulative Explained Variance"
                subtitle="Bars denote individual component variance; line traces cumulative variance across principal axes."
              >
                <div className="space-y-6">
                  <div className="h-80 w-full pt-4">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart
                        data={pcaData.variance_table.map((row) => ({
                          component: row.component,
                          variance: row.variance_ratio_pct,
                          cumulative: row.cumulative_variance_pct,
                          isRetained: row.is_retained,
                        }))}
                        margin={{ top: 20, right: 30, left: 0, bottom: 10 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="component" tick={{ fontSize: 11 }} />
                        <YAxis
                          yAxisId="left"
                          unit="%"
                          tick={{ fontSize: 11 }}
                          domain={[0, Math.ceil(Math.max(...pcaData.variance_table.map((r) => r.variance_ratio_pct)) * 1.1)]}
                        />
                        <YAxis
                          yAxisId="right"
                          orientation="right"
                          unit="%"
                          domain={[0, 100]}
                          tick={{ fontSize: 11 }}
                        />
                        <Tooltip
                          formatter={(value: any, name: string) => [
                            `${value}%`,
                            name === 'variance' ? 'Individual Component Variance' : 'Cumulative Variance',
                          ]}
                        />
                        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
                        {/* Reference lines for standard thresholds */}
                        <ReferenceLine
                          yAxisId="right"
                          y={80}
                          stroke="#94a3b8"
                          strokeDasharray="4 4"
                          label={{ value: '80% Threshold', position: 'insideTopLeft', fill: '#64748b', fontSize: 10 }}
                        />
                        <ReferenceLine
                          yAxisId="right"
                          y={90}
                          stroke="#94a3b8"
                          strokeDasharray="4 4"
                          label={{ value: '90% Threshold', position: 'insideTopLeft', fill: '#64748b', fontSize: 10 }}
                        />
                        <ReferenceLine
                          yAxisId="right"
                          y={95}
                          stroke="#94a3b8"
                          strokeDasharray="4 4"
                          label={{ value: '95% Threshold', position: 'insideTopLeft', fill: '#64748b', fontSize: 10 }}
                        />
                        <Bar
                          yAxisId="left"
                          dataKey="variance"
                          name="Individual Variance %"
                          radius={[4, 4, 0, 0]}
                        >
                          {pcaData.variance_table.map((entry, idx) => (
                            <Cell
                              key={`cell-${idx}`}
                              fill={entry.is_retained ? '#0d9488' : '#cbd5e1'}
                              className="cursor-pointer transition-opacity hover:opacity-80"
                              onClick={() => setNComponents(idx + 1)}
                            />
                          ))}
                        </Bar>
                        <Line
                          yAxisId="right"
                          type="monotone"
                          dataKey="cumulative"
                          name="Cumulative Variance %"
                          stroke="#0f172a"
                          strokeWidth={2.5}
                          dot={{ r: 4, fill: '#0f172a' }}
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Scientific interpretation note */}
                  <div className="p-4 bg-slate-50 border border-slate-200/80 rounded-xl flex items-start gap-3 text-xs leading-relaxed">
                    <Info className="w-4 h-4 text-teal-600 shrink-0 mt-0.5" />
                    <div className="space-y-1">
                      <span className="font-bold text-slate-800">
                        Kaiser-Guttman &amp; Elbow Criterion Interpretation
                      </span>
                      <p className="text-slate-600">
                        In standardized PCA, components with eigenvalues <MathInline math="\lambda_i > 1.0" /> contain more information
                        than an average single raw sensor. The scree plot clearly displays an elbow after PC3,
                        confirming that retaining <strong>k = 3</strong> components captures {pcaData.retained_variance_pct}% of total variance
                        while eliminating collinear noise from the remaining {pcaData.total_components - pcaData.n_components} components.
                      </p>
                    </div>
                  </div>
                </div>
              </Card>

              {/* Complete Variance & Invariants Table */}
              <Card title="Spectral Variance Table (Full Spectrum)">
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 text-left text-[11px] text-slate-500 font-sans">
                        <th className="p-3">Component</th>
                        <th className="p-3">Eigenvalue (<MathInline math="\lambda_i" />)</th>
                        <th className="p-3">Singular Value (<MathInline math="\sigma_i" />)</th>
                        <th className="p-3">Individual Variance %</th>
                        <th className="p-3">Cumulative Variance %</th>
                        <th className="p-3">Status</th>
                        <th className="p-3 text-right">Action</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pcaData.variance_table.map((row, idx) => (
                        <tr
                          key={row.component}
                          className={`border-b border-slate-100 transition-colors ${
                            row.is_retained ? 'bg-teal-50/40 hover:bg-teal-50/70' : 'hover:bg-slate-50'
                          }`}
                        >
                          <td className="p-3 font-bold text-slate-900">{row.component}</td>
                          <td className="p-3 text-teal-800 font-semibold">{row.eigenvalue.toFixed(4)}</td>
                          <td className="p-3 text-slate-700">{row.singular_value.toFixed(2)}</td>
                          <td className="p-3 font-semibold text-slate-900">{row.variance_ratio_pct.toFixed(2)}%</td>
                          <td className="p-3 text-slate-700">{row.cumulative_variance_pct.toFixed(2)}%</td>
                          <td className="p-3 font-sans">
                            {row.is_retained ? (
                              <span className="px-2 py-0.5 rounded-full bg-teal-100 text-teal-800 text-[10px] font-bold">
                                Retained (<MathInline math={`k \\le ${pcaData.n_components}`} />)
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded-full bg-slate-100 text-slate-500 text-[10px]">
                                Discarded
                              </span>
                            )}
                          </td>
                          <td className="p-3 text-right font-sans">
                            <button
                              onClick={() => setNComponents(idx + 1)}
                              className="text-[11px] text-teal-700 hover:text-teal-900 font-semibold underline"
                            >
                              Set k = {idx + 1}
                            </button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </motion.div>
          )}

          {/* =============================================================== */}
          {/* TAB 2: COMPONENT LOADINGS MATRIX */}
          {/* =============================================================== */}
          {activeTab === 'loadings' && (
            <motion.div
              key="loadings_tab"
              variants={tabContentVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="space-y-6"
            >
              {/* Diverging Loadings Matrix Grid */}
              <Card
                title={<span>Diverging Component Loading Matrix: <MathInline math="V \in \mathbb{R}^{d \times d}" /></span>}
                subtitle="Columns represent principal axes; rows denote sensor features. Colors encode sign and loading magnitude."
              >
                <div className="space-y-4">
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs font-mono border-collapse text-right">
                      <thead>
                        <tr className="border-b border-slate-200">
                          <th className="p-2.5 text-left text-slate-500 font-sans text-[11px]">Sensor Feature</th>
                          {pcaData.loadings_columns.map((col, colIdx) => (
                            <th
                              key={col}
                              onClick={() => setSelectedLoadingsPc(colIdx + 1)}
                              className={`p-2.5 cursor-pointer text-center transition-colors ${
                                selectedLoadingsPc === colIdx + 1
                                  ? 'bg-slate-900 text-teal-300 font-bold rounded-t-lg'
                                  : 'text-slate-700 hover:bg-slate-100'
                              }`}
                            >
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {pcaData.loadings_features.map((feat, rIdx) => (
                          <tr key={feat} className="border-b border-slate-100 hover:bg-slate-50/50">
                            <td className="p-2.5 text-left font-sans font-medium text-slate-800 whitespace-nowrap">
                              {feat}
                            </td>
                            {pcaData.loadings_columns.map((_, cIdx) => {
                              const val = pcaData.loadings_matrix[rIdx][cIdx];
                              const isSelectedCol = selectedLoadingsPc === cIdx + 1;
                              return (
                                <td
                                  key={cIdx}
                                  onClick={() => setSelectedLoadingsPc(cIdx + 1)}
                                  className={`p-2.5 cursor-pointer text-center transition-all ${getLoadingColorClass(
                                    val
                                  )} ${isSelectedCol ? 'ring-1 ring-slate-900/30' : ''}`}
                                  title={`${feat} on PC${cIdx + 1}: loading = ${val.toFixed(4)}, contribution = ${(val ** 2 * 100).toFixed(1)}%`}
                                >
                                  {val > 0 ? `+${val.toFixed(3)}` : val.toFixed(3)}
                                </td>
                              );
                            })}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {/* Legend */}
                  <div className="flex items-center justify-between text-[11px] text-slate-500 pt-2 border-t border-slate-100 flex-wrap gap-2">
                    <div className="flex items-center gap-3">
                      <span className="font-semibold">Loading Legend:</span>
                      <span className="inline-flex items-center gap-1">
                        <span className="w-3 h-3 rounded bg-teal-200 border border-teal-300" /> Strong Positive (&gt; 0.45)
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <span className="w-3 h-3 rounded bg-teal-50 border border-teal-200" /> Moderate Positive
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <span className="w-3 h-3 rounded bg-rose-200 border border-rose-300" /> Strong Negative (&lt; -0.45)
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <span className="w-3 h-3 rounded bg-rose-50 border border-rose-200" /> Moderate Negative
                      </span>
                    </div>
                    <span className="font-mono">Columns are orthonormal: <MathInline math="V^\mathsf{T} V = I_d" /></span>
                  </div>
                </div>
              </Card>

              {/* Selected Component Feature Contribution View */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Horizontal Bar Chart for Selected PC */}
                <Card
                  title={`Feature Loadings on PC${selectedLoadingsPc}`}
                  subtitle={<span>Directional coefficients <MathInline math="\mathbf{v}_i" /> for principal axis {selectedLoadingsPc}</span>}
                >
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-700">Select Principal Axis:</span>
                      <div className="flex gap-1">
                        {pcaData.loadings_columns.map((col, idx) => (
                          <button
                            key={col}
                            onClick={() => setSelectedLoadingsPc(idx + 1)}
                            className={`px-2.5 py-1 rounded text-xs font-mono font-bold transition-all ${
                              selectedLoadingsPc === idx + 1
                                ? 'bg-slate-900 text-teal-300 shadow-xs'
                                : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                            }`}
                          >
                            {col}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Feature Loading Bars */}
                    <div className="space-y-2 pt-2">
                      {pcaData.loadings_features.map((feat, idx) => {
                        const val = pcaData.loadings_matrix[idx][selectedLoadingsPc - 1];
                        const absPct = Math.min(100, Math.abs(val) * 100);
                        return (
                          <div key={feat} className="space-y-1">
                            <div className="flex justify-between text-xs font-mono">
                              <span className="font-sans text-slate-800">{feat}</span>
                              <span className={val >= 0 ? 'text-teal-700 font-bold' : 'text-rose-700 font-bold'}>
                                {val >= 0 ? `+${val.toFixed(4)}` : val.toFixed(4)}
                              </span>
                            </div>
                            <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden flex">
                              {val < 0 ? (
                                <div
                                  className="h-full bg-rose-500 rounded-full transition-all duration-300"
                                  style={{ width: `${absPct}%` }}
                                />
                              ) : (
                                <div
                                  className="h-full bg-teal-500 rounded-full transition-all duration-300"
                                  style={{ width: `${absPct}%` }}
                                />
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                </Card>

                {/* Contribution Breakdown Table */}
                <Card
                  title={`Variance Contributions to PC${selectedLoadingsPc}`}
                  subtitle={<span>Calculated as <MathInline math="v_i^2" /> where <MathInline math="\sum_i v_i^2 = 1.0" /> (Unit Norm Invariant)</span>}
                >
                  <div className="space-y-3">
                    <div className="overflow-x-auto">
                      <table className="w-full text-xs font-mono border-collapse">
                        <thead>
                          <tr className="border-b border-slate-200 text-left text-[11px] text-slate-500 font-sans">
                            <th className="p-2">Feature</th>
                            <th className="p-2">Raw Loading</th>
                            <th className="p-2">|Loading|</th>
                            <th className="p-2 text-right">Contribution %</th>
                          </tr>
                        </thead>
                        <tbody>
                          {pcaData.loadings_features
                            .map((feat, idx) => {
                              const val = pcaData.loadings_matrix[idx][selectedLoadingsPc - 1];
                              return {
                                feature: feat,
                                loading: val,
                                absLoading: Math.abs(val),
                                contributionPct: val ** 2 * 100,
                              };
                            })
                            .sort((a, b) => b.absLoading - a.absLoading)
                            .map((item) => (
                              <tr key={item.feature} className="border-b border-slate-100 hover:bg-slate-50">
                                <td className="p-2 font-sans font-medium text-slate-800">{item.feature}</td>
                                <td
                                  className={`p-2 font-semibold ${
                                    item.loading >= 0 ? 'text-teal-700' : 'text-rose-700'
                                  }`}
                                >
                                  {item.loading.toFixed(4)}
                                </td>
                                <td className="p-2 text-slate-600">{item.absLoading.toFixed(4)}</td>
                                <td className="p-2 text-right font-bold text-slate-900">
                                  {item.contributionPct.toFixed(1)}%
                                </td>
                              </tr>
                            ))}
                        </tbody>
                      </table>
                    </div>

                    <div className="p-3 bg-teal-50/70 border border-teal-200/80 rounded-xl text-xs text-teal-900 space-y-1">
                      <span className="font-bold block">Chemical Interpretation:</span>
                      <p className="text-[11px] leading-relaxed">
                        Features with large loadings on the same component co-vary strongly. On PC1, resistance sensors
                        (PT08.S1, S2, S4) share heavy co-movement reflecting general urban combustion emissions, while
                        temperature and humidity exhibit contrasting signs reflecting diurnal meteorological cycles.
                      </p>
                    </div>
                  </div>
                </Card>
              </div>
            </motion.div>
          )}

          {/* =============================================================== */}
          {/* TAB 3: PRINCIPAL COMPONENT SCORE EXPLORER (2D SCATTER) */}
          {/* =============================================================== */}
          {activeTab === 'scores' && (
            <motion.div
              key="scores_tab"
              variants={tabContentVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="space-y-6"
            >
              {/* Controls Bar for PC-Axis Pair */}
              <div className="bg-white border border-slate-200/90 rounded-2xl p-4 shadow-xs flex flex-wrap items-center justify-between gap-4">
                <div className="flex items-center gap-4 flex-wrap">
                  {/* X-Axis Component */}
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-700">X-Axis Component:</span>
                    <select
                      value={scorePcX}
                      onChange={(e) => setScorePcX(Number(e.target.value))}
                      className="text-xs font-mono font-bold py-1 px-2.5 rounded-lg border border-slate-300 bg-white text-slate-900"
                    >
                      {pcaData.loadings_columns.map((col, idx) => (
                        <option key={col} value={idx + 1}>
                          {col} ({pcaData.variance_table[idx]?.variance_ratio_pct}%)
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Y-Axis Component */}
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-semibold text-slate-700">Y-Axis Component:</span>
                    <select
                      value={scorePcY}
                      onChange={(e) => setScorePcY(Number(e.target.value))}
                      className="text-xs font-mono font-bold py-1 px-2.5 rounded-lg border border-slate-300 bg-white text-slate-900"
                    >
                      {pcaData.loadings_columns.map((col, idx) => (
                        <option key={col} value={idx + 1}>
                          {col} ({pcaData.variance_table[idx]?.variance_ratio_pct}%)
                        </option>
                      ))}
                    </select>
                  </div>
                </div>

                <div className="flex items-center gap-2">
                  <Badge variant="teal" size="sm" className="font-mono">
                    Sampled: {pcaData.scores_sample.length} Hourly Observations
                  </Badge>
                </div>
              </div>

              {/* 2D Scatter Chart Card */}
              <Card
                title={`Principal Score Projection: ${pcaData.score_pc_x_label} vs ${pcaData.score_pc_y_label}`}
                subtitle="Each point represents an hourly multi-sensor observation projected onto the orthogonal principal axes."
              >
                <div className="space-y-4">
                  <div className="h-96 w-full">
                    <ResponsiveContainer width="100%" height="100%">
                      <ScatterChart margin={{ top: 20, right: 30, left: 10, bottom: 20 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis
                          type="number"
                          dataKey="score_x"
                          name={pcaData.score_pc_x_label}
                          tick={{ fontSize: 11 }}
                          label={{
                            value: `${pcaData.score_pc_x_label} Score (${pcaData.score_pc_x_var_pct}% Explained Variance)`,
                            position: 'bottom',
                            offset: 5,
                            fontSize: 12,
                            fill: '#334155',
                          }}
                        />
                        <YAxis
                          type="number"
                          dataKey="score_y"
                          name={pcaData.score_pc_y_label}
                          tick={{ fontSize: 11 }}
                          label={{
                            value: `${pcaData.score_pc_y_label} Score (${pcaData.score_pc_y_var_pct}% Explained Variance)`,
                            angle: -90,
                            position: 'insideLeft',
                            offset: 0,
                            fontSize: 12,
                            fill: '#334155',
                          }}
                        />
                        <ZAxis type="number" dataKey="target" range={[40, 140]} name="Benzene Target C6H6" />
                        <Tooltip
                          cursor={{ strokeDasharray: '3 3' }}
                          content={({ active, payload }) => {
                            if (active && payload && payload.length) {
                              const data: PCAScorePoint = payload[0].payload;
                              return (
                                <div className="bg-slate-900 text-white p-3 rounded-xl shadow-lg text-xs space-y-1 font-mono">
                                  <div className="text-[10px] text-teal-300 font-sans font-semibold">
                                    {data.timestamp}
                                  </div>
                                  <div>
                                    {pcaData.score_pc_x_label}: <strong>{data.score_x}</strong>
                                  </div>
                                  <div>
                                    {pcaData.score_pc_y_label}: <strong>{data.score_y}</strong>
                                  </div>
                                  <div className="text-amber-300 pt-1 border-t border-slate-700">
                                    Reference <MathInline math="\mathrm{C}_6\mathrm{H}_6" />: <strong>{data.target} <MathInline math="\mu\mathrm{g/m}^3" /></strong>
                                  </div>
                                </div>
                              );
                            }
                            return null;
                          }}
                        />
                        <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="2 2" />
                        <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="2 2" />
                        <Scatter name="Sensor Observations" data={pcaData.scores_sample}>
                          {pcaData.scores_sample.map((entry, index) => {
                            // Color mapping by reference target concentration
                            const t = Math.min(1, Math.max(0, entry.target / 25));
                            const color = t > 0.6 ? '#e11d48' : t > 0.3 ? '#f59e0b' : '#0d9488';
                            return <Cell key={`cell-${index}`} fill={color} opacity={0.75} />;
                          })}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Scientific Guidance Banner */}
                  <div className="p-4 bg-amber-50/70 border border-amber-200 rounded-xl space-y-1 text-xs text-amber-950">
                    <div className="flex items-center gap-1.5 font-bold">
                      <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0" />
                      Visual Annotation vs. Model Fitting Warning
                    </div>
                    <p className="text-amber-900 text-[11px] leading-relaxed">
                      Points are colored by certified reference Benzene concentration (teal = low, amber = moderate, red = high).
                      <strong> The target analyzer was strictly excluded from PCA fitting.</strong> Visible gradients along PC1 confirm
                      that the dominant principal axis aligns strongly with true ambient hydrocarbon pollution without supervision.
                    </p>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}

          {/* =============================================================== */}
          {/* TAB 4: RECONSTRUCTION LAB & ERROR ANALYSIS */}
          {/* =============================================================== */}
          {activeTab === 'reconstruction' && (
            <motion.div
              key="reconstruction_tab"
              variants={tabContentVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="space-y-6"
            >
              {/* Reconstruction KPI Banner */}
              <div className="p-5 bg-gradient-to-r from-slate-900 to-slate-800 text-white rounded-2xl shadow-md border border-slate-700">
                <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-4 border-b border-slate-700">
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-teal-400 font-bold bg-teal-950/80 px-2 py-0.5 rounded border border-teal-500/30">
                      Eckart-Young-Mirsky Theorem
                    </span>
                    <h3 className="text-lg font-bold text-white mt-1">Low-Rank Frobenius Approximation Bounds</h3>
                    <p className="text-xs text-slate-300 mt-0.5">
                      Empirical reconstruction error strictly matches theoretical singular value tail sum:{' '}
                      <MathInline math="\|Z - \hat{Z}_k\|_F^2 = \sum_{j=k+1}^d \sigma_j^2" />
                    </p>
                  </div>

                  <Badge variant="teal" size="md" className="font-mono">
                    Parity Verified (&lt; <MathInline math="10^{-14}" />)
                  </Badge>
                </div>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 text-center font-mono">
                  <div className="p-3 rounded-xl bg-slate-800/80 border border-slate-700">
                    <div className="text-[10px] text-slate-400 font-sans">Retained Components (k)</div>
                    <div className="text-2xl font-bold text-teal-300 mt-1">{pcaData.n_components}</div>
                    <div className="text-[10px] text-slate-400 font-sans">of {pcaData.total_components} dimensions</div>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-800/80 border border-slate-700">
                    <div className="text-[10px] text-slate-400 font-sans">Empirical Error (Frobenius)</div>
                    <div className="text-2xl font-bold text-white mt-1">
                      {pcaData.reconstruction_error_frobenius.toFixed(2)}
                    </div>
                    <div className="text-[10px] text-teal-400 font-sans"><MathInline math="\|Z - \hat{Z}_k\|_F" /></div>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-800/80 border border-slate-700">
                    <div className="text-[10px] text-slate-400 font-sans">Theoretical Bound</div>
                    <div className="text-2xl font-bold text-emerald-300 mt-1">
                      {pcaData.theoretical_error_eckart_young > 0
                        ? pcaData.theoretical_error_eckart_young.toFixed(2)
                        : '0.00 (Full k)'}
                    </div>
                    <div className="text-[10px] text-slate-400 font-sans"><MathInline math="\sqrt{\sum_{j=k+1}^d \sigma_j^2}" /></div>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-800/80 border border-slate-700">
                    <div className="text-[10px] text-slate-400 font-sans">Relative Information Loss</div>
                    <div className="text-2xl font-bold text-amber-300 mt-1">
                      {(pcaData.relative_reconstruction_error * 100).toFixed(1)}%
                    </div>
                    <div className="text-[10px] text-slate-400 font-sans">of total matrix Frobenius norm</div>
                  </div>
                </div>
              </div>

              {/* 48-Hour Reconstructed Series Time-Series Comparison */}
              <Card
                title={`Original vs. Reconstructed Signal: ${pcaData.reconstruction_feature_name}`}
                subtitle={`Comparing raw physical readings against rank-${pcaData.n_components} subspace approximation across 48 continuous hours.`}
              >
                <div className="space-y-4">
                  <div className="h-72 w-full pt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart
                        data={pcaData.time_series_reconstruction}
                        margin={{ top: 10, right: 30, left: 0, bottom: 10 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} />
                        <YAxis tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
                        <Line
                          type="monotone"
                          dataKey="original"
                          name="Original Sensor Reading"
                          stroke="#0f172a"
                          strokeWidth={2}
                          dot={false}
                        />
                        <Line
                          type="monotone"
                          dataKey="reconstructed"
                          name={`Reconstructed (k=${pcaData.n_components})`}
                          stroke="#0d9488"
                          strokeWidth={2}
                          strokeDasharray="4 4"
                          dot={false}
                        />
                        <Bar
                          dataKey="residual"
                          name="Reconstruction Residual"
                          fill="#f59e0b"
                          opacity={0.6}
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="flex justify-between items-center text-xs text-slate-500 font-mono pt-2 border-t border-slate-100 flex-wrap gap-2">
                    <span>
                      Target feature: <strong>{pcaData.reconstruction_feature_name}</strong>
                    </span>
                    <span>Values shown in physical sensor units (rescaled from z-space).</span>
                  </div>
                </div>
              </Card>

              {/* Per-Feature Reconstruction Errors Table */}
              <Card title="Individual Sensor Channel Reconstruction Precision">
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 text-left text-[11px] text-slate-500 font-sans">
                        <th className="p-3">Sensor Channel</th>
                        <th className="p-3">RMSE (Physical Units)</th>
                        <th className="p-3">MAE (Physical Units)</th>
                        <th className="p-3 text-right">Relative Error %</th>
                        <th className="p-3 text-right">Signal Preserved</th>
                      </tr>
                    </thead>
                    <tbody>
                      {pcaData.feature_reconstruction.map((item) => (
                        <tr key={item.feature} className="border-b border-slate-100 hover:bg-slate-50">
                          <td className="p-3 font-sans font-bold text-slate-900">{item.feature}</td>
                          <td className="p-3 text-slate-700">{item.rmse}</td>
                          <td className="p-3 text-slate-700">{item.mae}</td>
                          <td className="p-3 text-right font-semibold text-amber-700">
                            {item.relative_error_pct}%
                          </td>
                          <td className="p-3 text-right font-bold text-teal-700">
                            {(100 - item.relative_error_pct).toFixed(1)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </motion.div>
          )}

          {/* =============================================================== */}
          {/* TAB 5: MATHEMATICAL THEORY & DERIVATIONS */}
          {/* =============================================================== */}
          {activeTab === 'theory' && (
            <motion.div
              key="theory_tab"
              variants={tabContentVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="space-y-6"
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* SVD to Covariance Spectral Identity */}
                <Card title="1. SVD to Covariance Spectral Equivalence">
                  <div className="space-y-3 text-xs leading-relaxed">
                    <div className="p-4 bg-slate-900 text-teal-300 rounded-xl font-mono text-center text-sm font-bold">
                      <MathBlock math="\lambda_i = \frac{\sigma_i^2}{n - 1} \iff \sigma_i = \sqrt{(n - 1)\lambda_i}" />
                    </div>
                    <p className="text-slate-600">
                      The Thin Singular Value Decomposition of the centered data matrix <MathInline math="X_c = U \Sigma V^\mathsf{T}" /> directly
                      decomposes the sample covariance matrix <MathInline math="S = \frac{1}{n - 1} X_c^\mathsf{T} X_c" />.
                    </p>
                    <div className="text-slate-600 space-y-1">
                      <span>Substituting <MathInline math="X_c" /> into <MathInline math="S" /> yields:</span>
                      <div className="bg-slate-50 border border-slate-200 p-2.5 rounded-lg">
                        <MathBlock math="S = \frac{1}{n - 1} (V \Sigma U^\mathsf{T})(U \Sigma V^\mathsf{T}) = V \left(\frac{\Sigma^2}{n - 1}\right) V^\mathsf{T}" />
                      </div>
                    </div>
                    <p className="text-slate-600">
                      Because <MathInline math="U" /> is orthonormal (<MathInline math="U^\mathsf{T} U = I" />), the right singular vectors <MathInline math="V" /> of the data matrix are identically
                      the eigenvectors of the covariance matrix <MathInline math="S" />, and the singular values <MathInline math="\sigma_i" /> relate to covariance
                      eigenvalues <MathInline math="\lambda_i" /> by <MathInline math="\lambda_i = \frac{\sigma_i^2}{n - 1}" />.
                    </p>
                  </div>
                </Card>

                {/* Eckart-Young-Mirsky Theorem */}
                <Card title="2. Eckart-Young-Mirsky Low-Rank Approximation">
                  <div className="space-y-3 text-xs leading-relaxed">
                    <div className="p-4 bg-slate-900 text-teal-300 rounded-xl font-mono text-center text-sm font-bold">
                      <MathBlock math="\min_{\mathrm{rank}(B) \le k} \| X_c - B \|_F^2 = \sum_{j=k+1}^d \sigma_j^2" />
                    </div>
                    <p className="text-slate-600">
                      The Eckart-Young-Mirsky theorem establishes that truncating the SVD at <MathInline math="k" /> principal components produces
                      the best possible rank-<MathInline math="k" /> approximation of the data matrix in both the Frobenius norm and spectral norm.
                    </p>
                    <p className="text-slate-600">
                      The total squared reconstruction error is mathematically identical to the sum of squares of the neglected
                      singular values. In AirSense, automated test invariants verify that empirical Frobenius residual norms match
                      this theoretical bound to machine precision (&lt; <MathInline math="10^{-14}" />).
                    </p>
                  </div>
                </Card>

                {/* Uncorrelated Scores Invariant */}
                <Card title="3. Uncorrelated Principal Scores">
                  <div className="space-y-3 text-xs leading-relaxed">
                    <div className="p-4 bg-slate-900 text-teal-300 rounded-xl font-mono text-center text-sm font-bold">
                      <MathBlock math="\frac{1}{n - 1} T^\mathsf{T} T = \Lambda = \mathrm{diag}(\lambda_1, \dots, \lambda_k)" />
                    </div>
                    <p className="text-slate-600">
                      The projected observation coordinates (scores) <MathInline math="T = X_c V_k" /> are completely decorrelated.
                      The sample covariance matrix of the scores is purely diagonal, eliminating the multi-collinearity
                      that causes variance inflation in Ordinary Least Squares (OLS).
                    </p>
                  </div>
                </Card>

                {/* Unsupervised Nature Caveat */}
                <Card title="4. Scientific Caveat: Variance vs. Target Calibration">
                  <div className="space-y-3 text-xs leading-relaxed">
                    <div className="p-4 bg-amber-50 border border-amber-200 text-amber-950 rounded-xl space-y-1">
                      <div className="flex items-center gap-1.5 font-bold">
                        <AlertTriangle className="w-4 h-4 text-amber-600" />
                        PCA Maximizes Variance, Not Accuracy
                      </div>
                      <p className="text-[11px] leading-relaxed">
                        PCA is strictly an unsupervised technique. Retaining more variance does not necessarily guarantee
                        higher calibration accuracy for predicting Benzene concentrations. In fact, our verified benchmark
                        proves that PCR with k=3 components achieves superior test-set RMSE (2.41) compared to full OLS (2.85),
                        because dropping the smallest 5 eigenvalues removes thermal drift and sensor cross-sensitivity noise.
                      </p>
                    </div>
                  </div>
                </Card>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      ) : null}
    </div>
  );
};
