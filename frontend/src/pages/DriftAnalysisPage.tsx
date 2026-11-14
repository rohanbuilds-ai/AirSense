import React, { useState, useEffect, useCallback } from 'react';
import {
  AlertTriangle,
  TrendingUp,
  Activity,
  Layers,
  Eye,
  Clock,
  RefreshCw,
  FileText,
  CheckCircle2,
  Calendar,
  ArrowDownRight,
  ArrowUpRight,
  Loader2,
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
  BarChart,
} from 'recharts';
import { motion, AnimatePresence } from 'motion/react';

import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { Alert } from '../components/ui/Alert';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { ErrorState } from '../components/ui/ErrorState';
import { MathInline, MathBlock } from '../components/ui/Math';
import { apiClient } from '../services/apiClient';
import { DriftAnalysisResponse } from '../types/api';
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

export const DriftAnalysisPage: React.FC = () => {
  // Navigation Tabs
  const [activeTab, setActiveTab] = useState<'console' | 'features' | 'residuals' | 'statistics' | 'findings'>('console');

  // Interactive Configuration
  const [modelType, setModelType] = useState<'OLS' | 'PCR'>('OLS');
  const [referencePartition, setReferencePartition] = useState<'train' | 'val'>('train');
  const [monitoringPartition, setMonitoringPartition] = useState<'val' | 'test' | 'monitoring'>('test');
  const [selectedFeature, setSelectedFeature] = useState<string>('PT08.S4(NO2)');
  const [rollingWindowHours, setRollingWindowHours] = useState<number>(72);

  // Time-Series Series Toggles
  const [showActual, setShowActual] = useState<boolean>(true);
  const [showPredicted, setShowPredicted] = useState<boolean>(true);
  const [showResidual, setShowResidual] = useState<boolean>(true);
  const [showRollingBias, setShowRollingBias] = useState<boolean>(true);
  const [showRollingRmse, setShowRollingRmse] = useState<boolean>(true);
  const [showCusum, setShowCusum] = useState<boolean>(false);
  const [showControlLimits, setShowControlLimits] = useState<boolean>(true);

  // API State
  const [driftData, setDriftData] = useState<DriftAnalysisResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch Drift Analysis Data
  const fetchDrift = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.analyzeDrift({
        model_type: modelType,
        k_components: 3,
        reference_partition: referencePartition,
        monitoring_partition: monitoringPartition,
        selected_feature: selectedFeature,
        rolling_window_hours: rollingWindowHours,
        timeline_subsample: 260,
      });
      setDriftData(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to compute drift analysis');
    } finally {
      setLoading(false);
    }
  }, [modelType, referencePartition, monitoringPartition, selectedFeature, rollingWindowHours]);

  useEffect(() => {
    fetchDrift();
  }, [fetchDrift]);

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* Top Header Banner */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="text-[11px] font-bold text-teal-700 tracking-wider uppercase font-mono bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
              Retrospective Monitoring
            </span>
            <span className="text-slate-300">&middot;</span>
            <span className="text-xs text-slate-500 font-medium">13 Continuous Months (March 2004 – April 2005)</span>
            <span className="text-slate-300">&middot;</span>
            <span className="text-[11px] font-mono text-slate-500">Historical UCI Telemetry</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Sensor Drift &amp; Calibration Monitoring</h1>
          <p className="text-sm text-slate-600 mt-1 max-w-3xl leading-relaxed">
            Investigate measurement shifts, track calibration prediction bias over time, examine rolling error dynamics,
            and evaluate statistical evidence distinguishing physical sensor degradation from seasonal weather confounding.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0 flex-wrap">
          <Badge variant="warning" size="md" className="gap-1.5 font-mono">
            <AlertTriangle className="w-4 h-4 text-amber-600" />
            DW = 0.2062 (Severe Serial Autocorrelation)
          </Badge>
          <button
            onClick={fetchDrift}
            disabled={loading}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-slate-100 hover:bg-slate-200 rounded-lg transition-colors border border-slate-200"
          >
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RefreshCw className="w-3.5 h-3.5" />}
            Refresh Monitoring
          </button>
        </div>
      </div>

      {/* Global Interactive Configuration Console */}
      <Card
        title="Drift Monitoring Configuration &amp; Window Controls"
        subtitle="Compare historical baseline calibration against out-of-time evaluation periods"
      >
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 items-end">
          {/* Model Selector */}
          <div className="space-y-1.5 p-3 rounded-xl border border-slate-200 bg-slate-50">
            <span className="text-xs font-bold text-slate-800 block">Monitored Calibrator</span>
            <select
              value={modelType}
              onChange={(e) => setModelType(e.target.value as 'OLS' | 'PCR')}
              className="w-full text-xs font-mono font-bold py-1.5 px-2.5 rounded-lg border border-slate-300 bg-white text-slate-900"
            >
              <option value="OLS">Full OLS (8 Features)</option>
              <option value="PCR">Compact PCR (k=3 Components)</option>
            </select>
            <p className="text-[10px] text-slate-500">
              {modelType === 'OLS' ? 'Prone to severe winter bias' : 'Higher noise resilience'}
            </p>
          </div>

          {/* Reference Partition */}
          <div className="space-y-1.5 p-3 rounded-xl border border-slate-200 bg-slate-50">
            <span className="text-xs font-bold text-slate-800 block">Reference Baseline</span>
            <select
              value={referencePartition}
              onChange={(e) => setReferencePartition(e.target.value as 'train' | 'val')}
              className="w-full text-xs font-medium py-1.5 px-2 rounded-lg border border-slate-300 bg-white text-slate-800"
            >
              <option value="train">Training (5,394 hrs &middot; Mar–Oct 2004)</option>
              <option value="val">Validation (1,798 hrs &middot; Oct 2004–Jan 2005)</option>
            </select>
            <p className="text-[10px] text-slate-500">Declared baseline for comparisons</p>
          </div>

          {/* Monitoring Partition */}
          <div className="space-y-1.5 p-3 rounded-xl border border-slate-200 bg-slate-50">
            <span className="text-xs font-bold text-slate-800 block">Monitoring Period</span>
            <select
              value={monitoringPartition}
              onChange={(e) => setMonitoringPartition(e.target.value as 'val' | 'test' | 'monitoring')}
              className="w-full text-xs font-medium py-1.5 px-2 rounded-lg border border-slate-300 bg-white text-slate-800"
            >
              <option value="test">Held-Out Test (1,799 hrs &middot; Jan–Apr 2005)</option>
              <option value="val">Validation (1,798 hrs &middot; Oct 2004–Jan 2005)</option>
              <option value="monitoring">Full Out-of-Time (3,597 hrs &middot; Oct–Apr)</option>
            </select>
            <p className="text-[10px] text-slate-500">Target window evaluated for drift</p>
          </div>

          {/* Rolling Window Size */}
          <div className="space-y-1.5 p-3 rounded-xl border border-slate-200 bg-slate-50">
            <span className="text-xs font-bold text-slate-800 block">Rolling Window</span>
            <select
              value={rollingWindowHours}
              onChange={(e) => setRollingWindowHours(Number(e.target.value))}
              className="w-full text-xs font-medium py-1.5 px-2 rounded-lg border border-slate-300 bg-white text-slate-800"
            >
              <option value={24}>24 Hours (Diurnal Filter)</option>
              <option value={72}>72 Hours (3 Days &middot; Default)</option>
              <option value={168}>168 Hours (Weekly Horizon)</option>
              <option value={720}>720 Hours (Monthly Horizon)</option>
            </select>
            <p className="text-[10px] text-slate-500">Smooths high-frequency variance</p>
          </div>

          {/* Feature Selector for Distribution Drift */}
          <div className="space-y-1.5 p-3 rounded-xl border border-slate-200 bg-slate-50">
            <span className="text-xs font-bold text-slate-800 block">Target Sensor Channel</span>
            <select
              value={selectedFeature}
              onChange={(e) => setSelectedFeature(e.target.value)}
              className="w-full text-xs font-mono font-medium py-1.5 px-2 rounded-lg border border-slate-300 bg-white text-slate-800"
            >
              {ALL_FEATURES.map((feat) => (
                <option key={feat} value={feat}>
                  {feat}
                </option>
              ))}
            </select>
            <p className="text-[10px] text-slate-500">Investigated in feature drift panel</p>
          </div>
        </div>
      </Card>

      {/* KPI Metrics Summary Strip */}
      {driftData && (
        <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
          {/* Reference RMSE */}
          <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl shadow-xs space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono block">Baseline RMSE</span>
            <span className="text-xl font-bold font-mono text-slate-900">{driftData.overall_metrics.ref_rmse}</span>
            <span className="text-[10px] text-slate-500 block"><MathInline math="\mu\mathrm{g/m}^3" /> in reference window</span>
          </div>

          {/* Monitoring RMSE & Inflation */}
          <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl shadow-xs space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-amber-700 uppercase font-mono block font-bold">Monitoring RMSE</span>
              <span className="text-[10px] font-bold text-rose-700 flex items-center font-mono">
                <ArrowUpRight className="w-3 h-3" />+{driftData.overall_metrics.rmse_inflation_pct.toFixed(0)}%
              </span>
            </div>
            <span className="text-xl font-bold font-mono text-amber-800">{driftData.overall_metrics.mon_rmse}</span>
            <span className="text-[10px] text-slate-500 block"><MathInline math="\mu\mathrm{g/m}^3" /> in monitoring period</span>
          </div>

          {/* Baseline Bias MBE */}
          <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl shadow-xs space-y-1">
            <span className="text-[10px] text-slate-400 uppercase font-mono block">Baseline Bias (MBE)</span>
            <span className="text-xl font-bold font-mono text-slate-900">
              {driftData.overall_metrics.ref_mbe >= 0 ? `+${driftData.overall_metrics.ref_mbe}` : driftData.overall_metrics.ref_mbe}
            </span>
            <span className="text-[10px] text-slate-500 block"><MathInline math="\mu\mathrm{g/m}^3" /> reference bias</span>
          </div>

          {/* Monitoring Bias & Net Shift */}
          <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl shadow-xs space-y-1">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-rose-700 uppercase font-mono block font-bold">Monitoring Bias</span>
              <span className="text-[10px] font-bold text-rose-700 flex items-center font-mono">
                <ArrowDownRight className="w-3 h-3" />
                {driftData.overall_metrics.bias_shift.toFixed(2)}
              </span>
            </div>
            <span className="text-xl font-bold font-mono text-rose-700">
              {driftData.overall_metrics.mon_mbe >= 0 ? `+${driftData.overall_metrics.mon_mbe}` : driftData.overall_metrics.mon_mbe}
            </span>
            <span className="text-[10px] text-slate-500 block"><MathInline math="\mu\mathrm{g/m}^3" /> prediction shift</span>
          </div>

          {/* Autocorrelation Diagnostic */}
          <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl shadow-xs space-y-1">
            <span className="text-[10px] text-amber-700 uppercase font-mono block font-bold">Durbin-Watson</span>
            <span className="text-xl font-bold font-mono text-amber-800">{driftData.durbin_watson_monitoring}</span>
            <span className="text-[10px] text-slate-500 block"><MathInline math={`\\rho \\approx ${(1.0 - driftData.durbin_watson_monitoring / 2.0).toFixed(2)}`} /> (High Serial Dependency)</span>
          </div>

          {/* Monitored Observations */}
          <div className="p-3.5 bg-white border border-slate-200/90 rounded-xl shadow-xs space-y-1">
            <span className="text-[10px] text-teal-700 uppercase font-mono block font-bold">Observation Count</span>
            <span className="text-xl font-bold font-mono text-slate-900">{driftData.monitoring_rows}</span>
            <span className="text-[10px] text-slate-500 block">Out-of-time hourly records</span>
          </div>
        </div>
      )}

      {/* Workspace Tabs Navigation */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-2 shadow-xs">
        <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
          <button
            onClick={() => setActiveTab('console')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'console'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <TrendingUp className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">1. Calibration Timeline</span>
              <span className={`text-[10px] ${activeTab === 'console' ? 'text-slate-300' : 'text-slate-500'}`}>
                Predictions &amp; rolling errors
              </span>
            </div>
          </button>

          <button
            onClick={() => setActiveTab('features')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'features'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <Activity className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">2. Feature Drift</span>
              <span className={`text-[10px] ${activeTab === 'features' ? 'text-slate-300' : 'text-slate-500'}`}>
                Distributions &amp; shift stats
              </span>
            </div>
          </button>

          <button
            onClick={() => setActiveTab('residuals')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'residuals'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <Layers className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">3. Residuals &amp; CUSUM</span>
              <span className={`text-[10px] ${activeTab === 'residuals' ? 'text-slate-300' : 'text-slate-500'}`}>
                Control bounds &amp; monthly progression
              </span>
            </div>
          </button>

          <button
            onClick={() => setActiveTab('statistics')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'statistics'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <FileText className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">4. Statistical Evidence</span>
              <span className={`text-[10px] ${activeTab === 'statistics' ? 'text-slate-300' : 'text-slate-500'}`}>
                KS &amp; Welch tests &amp; caveats
              </span>
            </div>
          </button>

          <button
            onClick={() => setActiveTab('findings')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'findings'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <Eye className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">5. Findings &amp; Attribution</span>
              <span className={`text-[10px] ${activeTab === 'findings' ? 'text-slate-300' : 'text-slate-500'}`}>
                Physical vs climate dilemma
              </span>
            </div>
          </button>
        </div>
      </div>

      {/* Main Workspace Content */}
      {error ? (
        <ErrorState title="Drift Analysis Error" message={error} onRetry={fetchDrift} />
      ) : loading && !driftData ? (
        <div className="p-16 text-center bg-white border border-slate-200 rounded-2xl shadow-xs">
          <LoadingSpinner label="Computing retrospective calibration drift and distribution shifts..." />
        </div>
      ) : driftData ? (
        <AnimatePresence mode="wait">
          {/* =============================================================== */}
          {/* TAB 1: CALIBRATION TIMELINE & PERFORMANCE OVER TIME */}
          {/* =============================================================== */}
          {activeTab === 'console' && (
            <motion.div
              key="console_tab"
              variants={tabContentVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="space-y-6"
            >
              {/* Primary Chart Card */}
              <Card
                title={`Calibration Performance Across Time: ${driftData.model_type}`}
                subtitle={`Tracking ground-truth C6H6 against predictions and ${rollingWindowHours}h rolling error metrics across 13 months.`}
              >
                <div className="space-y-4">
                  {/* Series Toggles Bar */}
                  <div className="flex flex-wrap items-center justify-between gap-3 p-3 bg-slate-50 border border-slate-200/80 rounded-xl text-xs">
                    <span className="font-semibold text-slate-700">Display Series:</span>
                    <div className="flex flex-wrap items-center gap-4">
                      <label className="inline-flex items-center gap-1.5 cursor-pointer text-slate-800">
                        <input
                          type="checkbox"
                          checked={showActual}
                          onChange={(e) => setShowActual(e.target.checked)}
                          className="w-3.5 h-3.5 rounded text-teal-600 accent-teal-600"
                        />
                        <span className="w-2.5 h-2.5 rounded-full bg-teal-600 inline-block" />
                        Observed Benzene (<MathInline math="\mathrm{C}_6\mathrm{H}_6" />)
                      </label>

                      <label className="inline-flex items-center gap-1.5 cursor-pointer text-slate-800">
                        <input
                          type="checkbox"
                          checked={showPredicted}
                          onChange={(e) => setShowPredicted(e.target.checked)}
                          className="w-3.5 h-3.5 rounded text-indigo-600 accent-indigo-600"
                        />
                        <span className="w-2.5 h-2.5 rounded-full bg-indigo-600 inline-block" />
                        Model Prediction
                      </label>

                      <label className="inline-flex items-center gap-1.5 cursor-pointer text-slate-800">
                        <input
                          type="checkbox"
                          checked={showResidual}
                          onChange={(e) => setShowResidual(e.target.checked)}
                          className="w-3.5 h-3.5 rounded text-amber-600 accent-amber-600"
                        />
                        <span className="w-2.5 h-2.5 rounded-full bg-amber-500 inline-block" />
                        Residual Error (<MathInline math="e_t" />)
                      </label>

                      <label className="inline-flex items-center gap-1.5 cursor-pointer text-slate-800">
                        <input
                          type="checkbox"
                          checked={showRollingBias}
                          onChange={(e) => setShowRollingBias(e.target.checked)}
                          className="w-3.5 h-3.5 rounded text-rose-600 accent-rose-600"
                        />
                        <span className="w-2.5 h-2.5 rounded-full bg-rose-600 inline-block" />
                        Rolling Bias ({rollingWindowHours}h)
                      </label>

                      <label className="inline-flex items-center gap-1.5 cursor-pointer text-slate-800">
                        <input
                          type="checkbox"
                          checked={showRollingRmse}
                          onChange={(e) => setShowRollingRmse(e.target.checked)}
                          className="w-3.5 h-3.5 rounded text-purple-600 accent-purple-600"
                        />
                        <span className="w-2.5 h-2.5 rounded-full bg-purple-600 inline-block" />
                        Rolling RMSE ({rollingWindowHours}h)
                      </label>

                      <label className="inline-flex items-center gap-1.5 cursor-pointer text-slate-800">
                        <input
                          type="checkbox"
                          checked={showCusum}
                          onChange={(e) => setShowCusum(e.target.checked)}
                          className="w-3.5 h-3.5 rounded text-rose-600 accent-rose-600"
                        />
                        <span className="w-2.5 h-2.5 rounded-full bg-rose-600 inline-block" />
                        CUSUM Drift Curve
                      </label>

                      <label className="inline-flex items-center gap-1.5 cursor-pointer text-slate-800">
                        <input
                          type="checkbox"
                          checked={showControlLimits}
                          onChange={(e) => setShowControlLimits(e.target.checked)}
                          className="w-3.5 h-3.5 rounded text-slate-600 accent-slate-600"
                        />
                        <span className="w-2.5 h-2.5 rounded-full bg-slate-400 inline-block" />
                        <MathInline math="\pm 3\sigma" /> Control Limits
                      </label>
                    </div>
                  </div>

                  {/* Recharts Timeline */}
                  <div className="h-96 w-full pt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart
                        data={driftData.timeline_series}
                        margin={{ top: 10, right: 30, left: 0, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} />
                        <YAxis tick={{ fontSize: 11 }} unit=" &mu;g/m&sup3;" />
                        <Tooltip />
                        <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
                        <ReferenceLine y={0} stroke="#94a3b8" strokeDasharray="2 2" />

                        {showControlLimits && (
                          <>
                            <ReferenceLine
                              y={driftData.timeline_series[0]?.upper_bound_3sigma}
                              stroke="#e11d48"
                              strokeDasharray="3 3"
                              label={{ value: '+3σ UCL', fill: '#e11d48', fontSize: 10, position: 'insideTopRight' }}
                            />
                            <ReferenceLine
                              y={driftData.timeline_series[0]?.lower_bound_3sigma}
                              stroke="#e11d48"
                              strokeDasharray="3 3"
                              label={{ value: '-3σ LCL', fill: '#e11d48', fontSize: 10, position: 'insideBottomRight' }}
                            />
                          </>
                        )}

                        {showActual && (
                          <Line
                            type="monotone"
                            dataKey="actual"
                            name="Observed Benzene"
                            stroke="#0d9488"
                            strokeWidth={1.5}
                            dot={false}
                          />
                        )}

                        {showPredicted && (
                          <Line
                            type="monotone"
                            dataKey="predicted"
                            name="Model Predicted"
                            stroke="#4f46e5"
                            strokeWidth={1.5}
                            strokeDasharray="3 3"
                            dot={false}
                          />
                        )}

                        {showResidual && (
                          <Bar
                            dataKey="residual"
                            name="Residual Error (y - y_hat)"
                            fill="#f59e0b"
                            opacity={0.4}
                          />
                        )}

                        {showRollingBias && (
                          <Line
                            type="monotone"
                            dataKey="rolling_bias"
                            name={`Rolling Bias (${rollingWindowHours}h)`}
                            stroke="#e11d48"
                            strokeWidth={2.5}
                            dot={false}
                          />
                        )}

                        {showRollingRmse && (
                          <Line
                            type="monotone"
                            dataKey="rolling_rmse"
                            name={`Rolling RMSE (${rollingWindowHours}h)`}
                            stroke="#9333ea"
                            strokeWidth={2}
                            dot={false}
                          />
                        )}

                        {showCusum && (
                          <Line
                            type="monotone"
                            dataKey="cusum"
                            name="CUSUM Drift Curve"
                            stroke="#e11d48"
                            strokeWidth={2}
                            dot={false}
                          />
                        )}
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>

                  {/* Partition demarcation callout */}
                  <div className="flex items-center justify-between text-[11px] text-slate-500 font-mono pt-2 border-t border-slate-100 flex-wrap gap-2">
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-teal-500" />
                      Training Partition: Mar 10 – Oct 27, 2004 (5,394 hrs)
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-amber-500" />
                      Validation Partition: Oct 27, 2004 – Jan 16, 2005 (1,798 hrs)
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span className="w-2.5 h-2.5 rounded-full bg-indigo-500" />
                      Held-Out Test Partition: Jan 16 – Apr 04, 2005 (1,799 hrs)
                    </span>
                  </div>
                </div>
              </Card>
            </motion.div>
          )}

          {/* =============================================================== */}
          {/* TAB 2: FEATURE-DISTRIBUTION DRIFT */}
          {/* =============================================================== */}
          {activeTab === 'features' && (
            <motion.div
              key="features_tab"
              variants={tabContentVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="space-y-6"
            >
              {/* Detailed Feature Shift Inspector */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Overlaid Density Histogram */}
                <Card
                  title={`Distribution Density Shift: ${driftData.selected_feature_detail.feature_name}`}
                  subtitle={`Overlaid density comparison: Reference (${driftData.reference_label}) vs Monitoring (${driftData.monitoring_label})`}
                >
                  <div className="space-y-4">
                    <div className="h-64 w-full pt-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart
                          data={driftData.selected_feature_detail.distribution_bins}
                          margin={{ top: 10, right: 20, left: 0, bottom: 20 }}
                        >
                          <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                          <XAxis dataKey="bin_center" tick={{ fontSize: 10 }} />
                          <YAxis tick={{ fontSize: 10 }} />
                          <Tooltip />
                          <Legend wrapperStyle={{ fontSize: 11, paddingTop: 10 }} />
                          <Bar
                            dataKey="ref_density"
                            name="Reference Density"
                            fill="#0d9488"
                            opacity={0.65}
                          />
                          <Bar
                            dataKey="mon_density"
                            name="Monitoring Density"
                            fill="#f59e0b"
                            opacity={0.65}
                          />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="text-[11px] text-slate-500 font-mono text-center">
                      Bin Centers across common feature range [min, max].
                    </div>
                  </div>
                </Card>

                {/* Statistical Shift Metrics Card */}
                <Card
                  title={`Statistical Drift Diagnostics: ${driftData.selected_feature_detail.feature_name}`}
                  subtitle="Quantifying magnitude of distribution shift with non-parametric and parametric tests."
                >
                  <div className="space-y-4 font-mono text-xs">
                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
                        <span className="text-[10px] text-slate-500 font-sans block">Reference Baseline Mean</span>
                        <span className="text-base font-bold text-slate-900 mt-0.5 block flex items-center gap-1">
                          {driftData.selected_feature_detail.ref_stats.mean} <MathInline math="\pm" /> {driftData.selected_feature_detail.ref_stats.std}
                        </span>
                        <span className="text-[10px] text-slate-400 font-sans">N = {driftData.selected_feature_detail.ref_stats.count}</span>
                      </div>

                      <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
                        <span className="text-[10px] text-slate-500 font-sans block">Monitoring Period Mean</span>
                        <span className="text-base font-bold text-slate-900 mt-0.5 block flex items-center gap-1">
                          {driftData.selected_feature_detail.mon_stats.mean} <MathInline math="\pm" /> {driftData.selected_feature_detail.mon_stats.std}
                        </span>
                        <span className="text-[10px] text-slate-400 font-sans">N = {driftData.selected_feature_detail.mon_stats.count}</span>
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
                        <span className="text-[10px] text-slate-500 font-sans block">Standardized Difference (Cohen's <MathInline math="d" />)</span>
                        <span className="text-base font-bold text-teal-700 mt-0.5 block">
                          <MathInline math={`d = ${driftData.selected_feature_detail.cohens_d}`} />
                        </span>
                        <span className="text-[10px] text-slate-500 font-sans">
                          {Math.abs(driftData.selected_feature_detail.cohens_d) > 0.8
                            ? 'Large effect size (> 0.8)'
                            : Math.abs(driftData.selected_feature_detail.cohens_d) > 0.2
                            ? 'Moderate effect size'
                            : 'Negligible effect size'}
                        </span>
                      </div>

                      <div className="p-3 rounded-xl bg-slate-50 border border-slate-200/80">
                        <span className="text-[10px] text-slate-500 font-sans block">Kolmogorov-Smirnov Statistic</span>
                        <span className="text-base font-bold text-indigo-700 mt-0.5 block">
                          <MathInline math={`D = ${driftData.selected_feature_detail.ks_statistic}`} />
                        </span>
                        <span className="text-[10px] text-slate-500 font-sans">
                          p = {driftData.selected_feature_detail.ks_pvalue < 1e-4 ? '< 0.0001' : driftData.selected_feature_detail.ks_pvalue.toFixed(4)}
                        </span>
                      </div>
                    </div>

                    <div className="p-3 bg-teal-50 border border-teal-200 rounded-xl text-xs text-teal-950 font-sans leading-relaxed">
                      <strong>Interpreting Feature Drift:</strong> Notice that for metal-oxide sensors like{' '}
                      <code>PT08.S4(NO2)</code>, the massive distribution shift (d = -1.87) is mirrored by the seasonal drop in
                      outdoor temperature (d = -1.87). This co-movement reflects temperature sensitivity rather than physical sensor breakdown.
                    </div>
                  </div>
                </Card>
              </div>

              {/* Full Feature Drift Comparison Table */}
              <Card
                title="Cross-Channel Sensor Drift Summary (All 8 Predictor Features)"
                subtitle="Evaluating distributional divergence between baseline reference and monitoring window."
              >
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 text-left text-[11px] text-slate-500 font-sans">
                        <th className="p-3">Sensor Feature</th>
                        <th className="p-3">Ref Mean</th>
                        <th className="p-3">Mon Mean</th>
                        <th className="p-3">Mean Shift (<MathInline math="\Delta \mu" />)</th>
                        <th className="p-3">Cohen's <MathInline math="d" /></th>
                        <th className="p-3">KS Distance (<MathInline math="D" />)</th>
                        <th className="p-3">KS <MathInline math="p" />-value</th>
                        <th className="p-3 text-right">Drift Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {driftData.feature_drift_summary.map((row) => (
                        <tr
                          key={row.feature}
                          onClick={() => setSelectedFeature(row.feature)}
                          className={`border-b border-slate-100 cursor-pointer transition-colors ${
                            selectedFeature === row.feature ? 'bg-teal-50/60 font-semibold' : 'hover:bg-slate-50'
                          }`}
                        >
                          <td className="p-3 font-sans font-bold text-slate-900">{row.feature}</td>
                          <td className="p-3 text-slate-700">{row.ref_mean}</td>
                          <td className="p-3 text-slate-700">{row.mon_mean}</td>
                          <td className={`p-3 font-bold ${row.shift < 0 ? 'text-rose-700' : 'text-teal-700'}`}>
                            {row.shift > 0 ? `+${row.shift}` : row.shift}
                          </td>
                          <td className="p-3 text-slate-900">{row.cohens_d}</td>
                          <td className="p-3 text-indigo-700">{row.ks_statistic}</td>
                          <td className="p-3 text-slate-500">
                            {row.ks_pvalue < 1e-4 ? '< 0.0001' : row.ks_pvalue.toFixed(4)}
                          </td>
                          <td className="p-3 text-right font-sans">
                            <span
                              className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                row.drift_status === 'Severe Drift'
                                  ? 'bg-rose-100 text-rose-800'
                                  : row.drift_status === 'Moderate Drift'
                                  ? 'bg-amber-100 text-amber-800'
                                  : 'bg-emerald-100 text-emerald-800'
                              }`}
                            >
                              {row.drift_status}
                            </span>
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
          {/* TAB 3: RESIDUALS MONITORING & CUSUM */}
          {/* =============================================================== */}
          {activeTab === 'residuals' && (
            <motion.div
              key="residuals_tab"
              variants={tabContentVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="space-y-6"
            >
              {/* CUSUM Drift Curve */}
              <Card
                title="CUSUM Drift Curve: Cumulative Sum of Centered Residuals"
                subtitle={<span><MathInline math="S_t = \sum_{i=1}^t (e_i - \bar{e}_{\mathrm{train}})" />. A steep downward slope indicates continuous, systematic negative prediction bias.</span>}
              >
                <div className="space-y-4">
                  <div className="h-64 w-full pt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <ComposedChart
                        data={driftData.timeline_series}
                        margin={{ top: 10, right: 30, left: 10, bottom: 20 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                        <XAxis dataKey="timestamp" tick={{ fontSize: 10 }} />
                        <YAxis tick={{ fontSize: 11 }} />
                        <Tooltip />
                        <ReferenceLine y={0} stroke="#94a3b8" />
                        <Line
                          type="monotone"
                          dataKey="cusum"
                          name="CUSUM (Cumulative Error)"
                          stroke="#e11d48"
                          strokeWidth={2.5}
                          dot={false}
                        />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>

                  <p className="text-[11px] text-slate-600 font-sans leading-relaxed">
                    The CUSUM trajectory remains flat throughout the training period (March–October 2004), then bends sharply
                    downward in November 2004 and plummets across winter 2005. This demonstrates that the calibration model systematically
                    over-predicts Benzene (producing negative residuals) as ambient temperatures fall.
                  </p>
                </div>
              </Card>

              {/* Monthly Error Progression Table */}
              <Card
                title="Monthly Error &amp; Mean Bias Progression (Across All 13 Contiguous Months)"
                subtitle="Chronological breakdown demonstrating the systematic drift in model performance."
              >
                <div className="overflow-x-auto">
                  <table className="w-full text-xs font-mono border-collapse">
                    <thead>
                      <tr className="border-b border-slate-200 text-left text-[11px] text-slate-500 font-sans">
                        <th className="p-3">Period</th>
                        <th className="p-3">Count</th>
                        <th className="p-3">RMSE (<MathInline math="\mu\mathrm{g/m}^3" />)</th>
                        <th className="p-3">MAE (<MathInline math="\mu\mathrm{g/m}^3" />)</th>
                        <th className="p-3"><MathInline math="R^2" /> Score</th>
                        <th className="p-3">Mean Bias (<MathInline math="\mu\mathrm{g/m}^3" />)</th>
                        <th className="p-3">Durbin-Watson</th>
                        <th className="p-3 text-right">Drift Phase</th>
                      </tr>
                    </thead>
                    <tbody>
                      {driftData.monthly_progression.map((m) => (
                        <tr key={m.period} className="border-b border-slate-100 hover:bg-slate-50">
                          <td className="p-3 font-bold text-slate-900">{m.period}</td>
                          <td className="p-3 text-slate-500">{m.count}</td>
                          <td className="p-3 text-slate-700 font-semibold">{m.rmse.toFixed(3)}</td>
                          <td className="p-3 text-slate-700">{m.mae.toFixed(3)}</td>
                          <td className="p-3 text-slate-700">{m.r2.toFixed(3)}</td>
                          <td
                            className={`p-3 font-bold ${
                              m.mean_bias < -1.0 ? 'text-rose-700' : m.mean_bias > 0.5 ? 'text-amber-700' : 'text-teal-700'
                            }`}
                          >
                            {m.mean_bias >= 0 ? `+${m.mean_bias.toFixed(3)}` : m.mean_bias.toFixed(3)}
                          </td>
                          <td className="p-3 text-amber-800">{m.durbin_watson.toFixed(4)}</td>
                          <td className="p-3 text-right font-sans">
                            {m.period <= '2004-10' ? (
                              <span className="px-2 py-0.5 rounded-full bg-teal-50 text-teal-800 text-[10px] font-bold">
                                Training Period
                              </span>
                            ) : m.period <= '2005-01' ? (
                              <span className="px-2 py-0.5 rounded-full bg-indigo-50 text-indigo-800 text-[10px] font-bold">
                                Validation Window
                              </span>
                            ) : (
                              <span className="px-2 py-0.5 rounded-full bg-rose-50 text-rose-800 text-[10px] font-bold">
                                Held-Out Test Window
                              </span>
                            )}
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
          {/* TAB 4: STATISTICAL EVIDENCE & DIAGNOSTICS */}
          {/* =============================================================== */}
          {activeTab === 'statistics' && (
            <motion.div
              key="statistics_tab"
              variants={tabContentVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="space-y-6"
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Kolmogorov-Smirnov Test */}
                <Card title="1. Two-Sample Kolmogorov-Smirnov Test">
                  <div className="space-y-3 text-xs leading-relaxed">
                    <div className="p-3 rounded-xl bg-slate-900 text-teal-300 font-mono text-center text-sm font-bold">
                      <MathBlock math="D = \sup_x |F_{\mathrm{ref}}(x) - F_{\mathrm{mon}}(x)|" />
                    </div>
                    <p className="text-slate-600">
                      <strong>Hypothesis:</strong> <MathInline math="H_0: F_{\mathrm{ref}} = F_{\mathrm{mon}}" /> (Identical reference and monitoring cumulative distributions).
                    </p>
                    <p className="text-slate-600">
                      <strong>Result for {driftData.selected_feature_detail.feature_name}:</strong> <MathInline math={`D = ${driftData.selected_feature_detail.ks_statistic}`} /> (p ={' '}
                      {driftData.selected_feature_detail.ks_pvalue < 1e-4 ? '< 0.0001' : driftData.selected_feature_detail.ks_pvalue.toFixed(4)}).
                    </p>
                    <Alert variant="warning" title="Large Sample Size Effect">
                      With sample sizes N &gt; 1,000, microscopic differences easily yield <MathInline math="p < 10^{-15}" />. For this reason, the
                      Kolmogorov-Smirnov distance <strong>D</strong> must be interpreted as an effect size rather than relying solely
                      on the nominal p-value.
                    </Alert>
                  </div>
                </Card>

                {/* Welch's t-test */}
                <Card title="2. Two-Sample Welch's t-Test &amp; Cohen's d">
                  <div className="space-y-3 text-xs leading-relaxed">
                    <div className="p-3 rounded-xl bg-slate-900 text-teal-300 font-mono text-center text-sm font-bold">
                      <MathBlock math="t = \frac{\bar{x}_{\mathrm{mon}} - \bar{x}_{\mathrm{ref}}}{\sqrt{\frac{s_{\mathrm{mon}}^2}{N_{\mathrm{mon}}} + \frac{s_{\mathrm{ref}}^2}{N_{\mathrm{ref}}}}}" />
                    </div>
                    <p className="text-slate-600">
                      <strong>Hypothesis:</strong> <MathInline math="H_0: \mu_{\mathrm{mon}} = \mu_{\mathrm{ref}}" /> (Equal population means).
                    </p>
                    <p className="text-slate-600">
                      <strong>Result for {driftData.selected_feature_detail.feature_name}:</strong> <MathInline math={`t = ${driftData.selected_feature_detail.welch_t_statistic}`} />, Cohen's <MathInline math={`d = ${driftData.selected_feature_detail.cohens_d}`} />.
                    </p>
                    <Alert variant="info" title="Welch-Satterthwaite Adjustment">
                      Does not assume equal population variances (<MathInline math="\sigma_1^2 \ne \sigma_2^2" />), making it robust to unequal seasonal dispersions
                      inherent to atmospheric sensor deployments.
                    </Alert>
                  </div>
                </Card>

                {/* Durbin-Watson Autocorrelation */}
                <Card title="3. Durbin-Watson Serial Autocorrelation Diagnostic">
                  <div className="space-y-3 text-xs leading-relaxed">
                    <div className="p-3 rounded-xl bg-slate-900 text-teal-300 font-mono text-center text-sm font-bold">
                      <MathBlock math={`DW = \\frac{\\sum_{t=2}^n (e_t - e_{t-1})^2}{\\sum_{t=1}^n e_t^2} = ${driftData.durbin_watson_monitoring}`} />
                    </div>
                    <p className="text-slate-600">
                      <strong>Standard Benchmark:</strong> <MathInline math="DW \approx 2.0" /> indicates independent errors. Values approaching 0
                      indicate severe positive first-order autocorrelation:
                      <br />
                      <div className="font-mono text-slate-800 bg-slate-50 border border-slate-200 p-2 rounded-lg mt-1">
                        <MathInline math={`\\rho \\approx 1 - \\frac{DW}{2} = 1 - \\frac{${driftData.durbin_watson_monitoring}}{2} \\approx ${(1.0 - driftData.durbin_watson_monitoring / 2.0).toFixed(3)}`} />
                      </div>
                    </p>
                    <Alert variant="warning" title="Violation of i.i.d. Assumption">
                      Hourly residuals violate independence. Standard error formulas underestimate true parameter variance by a factor
                      of <MathInline math="\sqrt{\frac{1 + \rho}{1 - \rho}} \approx 4.3 \times" />. Naive p-values must not be treated as definitive proof.
                    </Alert>
                  </div>
                </Card>

                {/* Multiple Testing & Bonferroni Correction */}
                <Card title="4. Multiple Hypothesis Testing Caveats">
                  <div className="space-y-3 text-xs leading-relaxed">
                    <div className="p-3 rounded-xl bg-slate-900 text-teal-300 font-mono text-center text-sm font-bold">
                      <MathBlock math="\alpha_{\mathrm{adjusted}} = \frac{\alpha}{m} = \frac{0.05}{8} = 0.00625" />
                    </div>
                    <p className="text-slate-600">
                      When monitoring across all 8 sensor channels and multiple rolling windows, family-wise error rate escalates.
                      Applying Bonferroni correction controls the false discovery rate.
                    </p>
                    <p className="text-slate-600">
                      Even under strict Bonferroni thresholds, sensor shifts in <code>PT08.S4(NO2)</code>, <code>T</code>, and <code>AH</code>{' '}
                      remain statistically significant with <MathInline math="p < 10^{-50}" />.
                    </p>
                  </div>
                </Card>
              </div>
            </motion.div>
          )}

          {/* =============================================================== */}
          {/* TAB 5: EXPLAINABLE FINDINGS & ATTRIBUTION DILEMMA */}
          {/* =============================================================== */}
          {activeTab === 'findings' && (
            <motion.div
              key="findings_tab"
              variants={tabContentVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              className="space-y-6"
            >
              {/* Findings Stream */}
              <div className="space-y-3">
                <div className="text-xs font-bold text-slate-800">
                  Dynamically Generated Analytical Findings ({driftData.explainable_findings.length} Items)
                </div>

                <div className="space-y-3">
                  {driftData.explainable_findings.map((item, idx) => (
                    <div
                      key={idx}
                      className={`p-4 rounded-xl border space-y-2 transition-all ${
                        item.severity === 'breach'
                          ? 'bg-rose-50/80 border-rose-200'
                          : item.severity === 'warning'
                          ? 'bg-amber-50/80 border-amber-200'
                          : 'bg-slate-50 border-slate-200'
                      }`}
                    >
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          {item.severity === 'breach' ? (
                            <span className="p-1 rounded bg-rose-100 text-rose-700">
                              <AlertTriangle className="w-4 h-4" />
                            </span>
                          ) : item.severity === 'warning' ? (
                            <span className="p-1 rounded bg-amber-100 text-amber-800">
                              <AlertTriangle className="w-4 h-4" />
                            </span>
                          ) : (
                            <span className="p-1 rounded bg-teal-100 text-teal-800">
                              <CheckCircle2 className="w-4 h-4" />
                            </span>
                          )}
                          <span className="text-xs font-bold text-slate-900">{item.title}</span>
                        </div>
                        <span className="font-mono text-[10px] font-bold px-2 py-0.5 rounded bg-white border text-slate-700">
                          {item.supporting_statistic}
                        </span>
                      </div>

                      <p className="text-xs text-slate-700 leading-relaxed font-sans">{item.description}</p>

                      <div className="text-[11px] text-slate-500 font-sans pt-1 border-t border-slate-200/60 leading-tight">
                        <strong>Scientific Caveat:</strong> {item.caveat}
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* The Attribution Dilemma Card */}
              <Card
                title="The Sensor Aging vs. Environmental Confounding Dilemma"
                subtitle="Why distribution changes do not automatically prove sensor hardware degradation."
              >
                <div className="space-y-4 text-xs leading-relaxed text-slate-700">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl space-y-2">
                      <div className="flex items-center gap-2 font-bold text-slate-900">
                        <Calendar className="w-4 h-4 text-teal-600" />
                        1. Climate Seasonal Variation
                      </div>
                      <p className="text-[11px] text-slate-600 leading-relaxed">
                        Outdoor temperatures in Rome swing from 40°C in July to 0°C in January. Absolute humidity
                        drops by 50%. Metal oxide gas sensors rely on heated semiconductor elements (~350°C to 450°C).
                        When ambient air is cold, heat dissipation changes surface reaction kinetics, causing co-varying
                        shifts across all sensor responses without hardware failure.
                      </p>
                    </div>

                    <div className="p-3.5 bg-slate-50 border border-slate-200 rounded-xl space-y-2">
                      <div className="flex items-center gap-2 font-bold text-slate-900">
                        <Clock className="w-4 h-4 text-amber-600" />
                        2. Physical Sensor Aging
                      </div>
                      <p className="text-[11px] text-slate-600 leading-relaxed">
                        Continuous operation causes irreversible crystal sintering, heater coil resistance changes, and surface
                        contamination from sulfur and silicone vapors. These effects alter baseline resistance over months.
                        However, because no periodic zero/span chamber calibrations were recorded, physical aging cannot be mathematically
                        decoupled from annual weather cycles.
                      </p>
                    </div>
                  </div>

                  <Alert variant="info" title="Academic Methodology Conclusion">
                    In university demonstrations, always state that drift observed in the UCI dataset represents
                    <strong> combined environmental-sensor co-drift</strong>. Claiming definitive physical hardware failure
                    without periodic reference-gas chamber verification is scientifically invalid.
                  </Alert>
                </div>
              </Card>
            </motion.div>
          )}
        </AnimatePresence>
      ) : null}
    </div>
  );
};
