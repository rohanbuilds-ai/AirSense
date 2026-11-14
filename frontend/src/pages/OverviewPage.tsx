import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
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
} from 'recharts';
import {
  Activity,
  Layers,
  Calendar,
  ShieldCheck,
  Sparkles,
  Info,
  ArrowRight,
  BookOpen,
} from 'lucide-react';
import { apiClient } from '../services/apiClient';
import { OverviewMetricsResponse, OverviewChartsResponse } from '../types/api';
import { Card } from '../components/ui/Card';
import { MetricCard } from '../components/ui/MetricCard';
import { Badge } from '../components/ui/Badge';
import { Alert } from '../components/ui/Alert';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { ErrorState } from '../components/ui/ErrorState';
import { MathInline } from '../components/ui/Math';

export const OverviewPage: React.FC = () => {
  const [metrics, setMetrics] = useState<OverviewMetricsResponse | null>(null);
  const [charts, setCharts] = useState<OverviewChartsResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [metricsData, chartsData] = await Promise.all([
        apiClient.getOverviewMetrics(),
        apiClient.getOverviewCharts(),
      ]);
      setMetrics(metricsData);
      setCharts(chartsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to retrieve overview telemetry from backend');
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (isLoading) {
    return <LoadingSpinner label="Loading scientific telemetry and benchmark matrices from backend..." />;
  }

  if (error || !metrics || !charts) {
    return (
      <ErrorState
        title="Telemetry Disconnected"
        message={error || 'Unable to establish connection with AirSense backend gateway.'}
        onRetry={loadData}
      />
    );
  }

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* 1. Visually Distinctive Dark Hero Region */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-[#090D16] via-[#0F172A] to-[#111C33] border border-slate-800/90 p-8 shadow-xl text-white">
        {/* Subtle technical background grid */}
        <div className="absolute inset-0 bg-tech-grid opacity-25 pointer-events-none" />
        <div className="absolute top-0 right-0 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl pointer-events-none -mr-20 -mt-20" />
        <div className="absolute bottom-0 left-1/3 w-80 h-80 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col lg:flex-row lg:items-center justify-between gap-8">
          <div className="space-y-3 max-w-3xl">
            <div className="flex flex-wrap items-center gap-2.5">
              <span className="text-[11px] font-bold text-teal-300 tracking-wider uppercase font-mono bg-teal-500/15 px-2.5 py-0.5 rounded-full border border-teal-500/30 flex items-center gap-1.5 shadow-2xs">
                <Sparkles className="w-3 h-3 text-teal-400" />
                University Mathematics &amp; Calibration Project
              </span>
              <span className="text-slate-500">·</span>
              <span className="text-xs text-slate-300 font-medium">UCI Machine Learning Repository #360</span>
            </div>

            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white leading-tight font-sans">
              AirSense: Multivariate Sensor Calibration &amp; Matrix Analysis
            </h1>

            <p className="text-sm text-slate-300 leading-relaxed font-sans font-normal">
              Estimating atmospheric benzene concentrations (<MathInline math="\mathrm{C}_6\mathrm{H}_6" />) from a multi-sensor array of solid-state
              metal-oxide sensors. This project investigates how real sensor measurements are represented as a matrix,
              how linear algebra solves calibration problems, and how dimensionality reduction and sensor drift affect prediction accuracy.
            </p>

            <div className="pt-2 flex flex-wrap items-center gap-3">
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-800/80 border border-slate-700/80 text-xs text-slate-300">
                <Calendar className="w-3.5 h-3.5 text-slate-400" />
                <span className="font-medium text-slate-200">13 Months: Mar 2004 – Apr 2005</span>
              </div>
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-800/80 border border-slate-700/80 text-xs text-slate-300">
                <ShieldCheck className="w-3.5 h-3.5 text-teal-400" />
                <span>Zero-Leakage Split (Train 60% / Val 20% / Held-Out Test 20%)</span>
              </div>
            </div>
          </div>

          {/* Hero Callout Card: Validation Selection & Walkthrough Route */}
          <div className="lg:w-84 shrink-0 bg-slate-900/90 backdrop-blur-md border border-slate-700/80 rounded-2xl p-5 space-y-4 shadow-lg ring-1 ring-white/5">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
              <span className="text-[10px] font-mono uppercase tracking-wider text-teal-400 font-bold">
                Validation-Selected Model
              </span>
              <Badge variant="teal" size="sm">
                <span><MathInline math="k^* = 8" /> (Full OLS)</span>
              </Badge>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Validation Performance:</span>
                <span className="font-mono font-bold text-teal-300">
                  RMSE 2.024 <MathInline math="\mu\mathrm{g/m}^3" /> (<MathInline math="R^2 = 0.9423" />)
                </span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-slate-400">Held-Out Test Performance:</span>
                <span className="font-mono font-bold text-slate-200">
                  RMSE 2.855 <MathInline math="\mu\mathrm{g/m}^3" /> (<MathInline math="R^2 = 0.8079" />)
                </span>
              </div>
            </div>

            <p className="text-[11px] text-slate-400 leading-snug">
              Selected strictly on the validation partition prior to test set evaluation, following scientific zero-leakage standards.
            </p>

            <Link
              to="/walkthrough"
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold rounded-xl text-xs transition-colors shadow-md shadow-teal-500/20"
            >
              <BookOpen className="w-3.5 h-3.5" />
              <span>Open Guided Project Walkthrough</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </Link>
          </div>
        </div>
      </div>

      {/* 2. Key Metrics Cards Grid (Focused 5-Card Layout, Post-Hoc KPI Removed) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <MetricCard
          label="Total Observations"
          value={metrics.observations_available.toLocaleString()}
          unit="hours"
          subtitle="389.8 continuous days"
          icon={<Calendar className="w-4 h-4 text-slate-500" />}
        />
        <MetricCard
          label="Supervised Usable"
          value={metrics.usable_target_observations.toLocaleString()}
          unit="hours"
          badge={{ text: '96.09% valid', variant: 'success' }}
          subtitle="366 blackout rows pruned"
          icon={<Activity className="w-4 h-4 text-slate-500" />}
        />
        <MetricCard
          label="Feature Channels"
          value={metrics.sensor_features_count}
          unit="inputs"
          subtitle="5 MOX + 3 Meteorology"
          icon={<Layers className="w-4 h-4 text-slate-500" />}
        />
        <MetricCard
          label="Validation Selected"
          value={<MathInline math="k^* = 8" />}
          unit="Full OLS"
          badge={{ text: 'Val RMSE: 2.024', variant: 'teal' }}
          subtitle={<>Val <MathInline math="R^2 = 0.9423" /> (Tuning Rule)</>}
          icon={<ShieldCheck className="w-4 h-4 text-teal-600" />}
        />
        <MetricCard
          label="Held-Out Test RMSE"
          value={metrics.validation_selected_test_rmse.toFixed(3)}
          unit={<MathInline math="\mu\mathrm{g/m}^3" />}
          subtitle={<>Test <MathInline math="R^2 =" /> {metrics.validation_selected_test_r2.toFixed(4)}</>}
          badge={{ text: 'Selected Model', variant: 'default' }}
        />
      </div>

      {/* Main Stages of Mathematical Workflow */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-900 font-sans">
              Main Stages of the Mathematical &amp; Calibration Workflow
            </h3>
            <p className="text-xs text-slate-500">
              How real-world sensor measurements progress from raw matrix formulation to evaluated predictions
            </p>
          </div>
          <Link
            to="/dataset"
            className="text-xs font-semibold text-teal-700 hover:text-teal-800 flex items-center gap-1"
          >
            Inspect Matrix Details <ArrowRight className="w-3 h-3" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 pt-1">
          <div className="p-3.5 bg-slate-50 border border-slate-200/80 rounded-xl space-y-1">
            <div className="flex items-center gap-1.5 text-xs font-bold text-slate-900">
              <span className="w-5 h-5 rounded-full bg-teal-100 text-teal-800 text-[11px] flex items-center justify-center font-mono">1</span>
              <span>Matrix Representation</span>
            </div>
            <p className="text-[11px] text-slate-600 font-mono"><MathInline math="X \in \mathbb{R}^{8991 \times 8}, \mathbf{y} \in \mathbb{R}^{8991 \times 1}" /></p>
            <p className="text-[11px] text-slate-500">8 eligible sensor features mapped against reference Benzene target.</p>
          </div>

          <div className="p-3.5 bg-slate-50 border border-slate-200/80 rounded-xl space-y-1">
            <div className="flex items-center gap-1.5 text-xs font-bold text-slate-900">
              <span className="w-5 h-5 rounded-full bg-teal-100 text-teal-800 text-[11px] flex items-center justify-center font-mono">2</span>
              <span>Chronological Splits</span>
            </div>
            <p className="text-[11px] text-slate-600 font-mono">60% Train / 20% Val / 20% Test</p>
            <p className="text-[11px] text-slate-500">Contiguous non-overlapping temporal partitions preventing data snooping.</p>
          </div>

          <div className="p-3.5 bg-slate-50 border border-slate-200/80 rounded-xl space-y-1">
            <div className="flex items-center gap-1.5 text-xs font-bold text-slate-900">
              <span className="w-5 h-5 rounded-full bg-teal-100 text-teal-800 text-[11px] flex items-center justify-center font-mono">3</span>
              <span>Linear Algebra Solvers</span>
            </div>
            <p className="text-[11px] text-slate-600 font-mono">MGS QR &amp; Spectral SVD</p>
            <p className="text-[11px] text-slate-500">Full-rank QR avoids condition-number squaring <MathInline math="\kappa(X^\mathsf{T} X) = \kappa(X)^2" /> in normal equations.</p>
          </div>

          <div className="p-3.5 bg-slate-50 border border-slate-200/80 rounded-xl space-y-1">
            <div className="flex items-center gap-1.5 text-xs font-bold text-slate-900">
              <span className="w-5 h-5 rounded-full bg-teal-100 text-teal-800 text-[11px] flex items-center justify-center font-mono">4</span>
              <span>Model Tuning &amp; Selection</span>
            </div>
            <p className="text-[11px] text-slate-600 font-mono"><MathInline math="k^* = 8" /> selected on Val</p>
            <p className="text-[11px] text-slate-500">Validation tuning selected full OLS, achieving 2.024 <MathInline math="\mu\mathrm{g/m}^3" /> validation RMSE.</p>
          </div>

          <div className="p-3.5 bg-slate-50 border border-slate-200/80 rounded-xl space-y-1">
            <div className="flex items-center gap-1.5 text-xs font-bold text-slate-900">
              <span className="w-5 h-5 rounded-full bg-teal-100 text-teal-800 text-[11px] flex items-center justify-center font-mono">5</span>
              <span>Drift &amp; Diagnostics</span>
            </div>
            <p className="text-[11px] text-slate-600 font-mono"><MathInline math="DW = 0.2062" /> &amp; CUSUM</p>
            <p className="text-[11px] text-slate-500">Autocorrelation &amp; seasonal confounding tracked across 13 continuous months.</p>
          </div>
        </div>
      </div>

      {/* Methodological Protocol Banner */}
      <div className="bg-slate-900 text-slate-100 rounded-2xl p-5 shadow-sm border border-slate-800 flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-1 max-w-4xl">
          <div className="flex items-center gap-2">
            <span className="text-[10px] font-bold tracking-widest text-teal-400 uppercase font-mono">
              Academic Protocol Note
            </span>
            <span className="text-slate-600">·</span>
            <span className="text-xs text-slate-300 font-medium">Model Selection vs. Post-Hoc Generalization</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed font-sans">
            {metrics.model_selection_note}
          </p>
        </div>
        <div className="shrink-0 px-3.5 py-2 rounded-xl bg-slate-800 border border-slate-700/80 text-xs font-mono text-teal-300 shadow-2xs">
          Autocorrelation DW: {metrics.durbin_watson} (<MathInline math="\rho \approx 0.90" />)
        </div>
      </div>

      {/* 3. Primary Visualizations Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Viz 1: Measured vs. Predicted Benzene */}
        <Card
          title="Measured vs. Predicted Benzene Concentration"
          subtitle="Held-out test set evaluation (Jan – Apr 2005) comparing Full OLS and Compact PCR against certified analyzer"
        >
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ScatterChart margin={{ top: 15, right: 25, bottom: 25, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                <XAxis
                  type="number"
                  dataKey="actual"
                  name="Reference C6H6(GT)"
                  unit=" &mu;g/m&sup3;"
                  stroke="#94A3B8"
                  fontSize={11}
                  domain={[0, 45]}
                  label={{
                    value: 'Certified Reference Analyzer: C6H6(GT) [µg/m³]',
                    position: 'insideBottom',
                    offset: -14,
                    fill: '#64748B',
                    fontSize: 11,
                  }}
                />
                <YAxis
                  type="number"
                  dataKey="predicted_ols"
                  name="Predicted C6H6"
                  unit=" &mu;g/m&sup3;"
                  stroke="#94A3B8"
                  fontSize={11}
                  domain={[0, 45]}
                  label={{
                    value: 'Calibrated Prediction [µg/m³]',
                    angle: -90,
                    position: 'insideLeft',
                    offset: 12,
                    fill: '#64748B',
                    fontSize: 11,
                  }}
                />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ payload }) => {
                    if (!payload || payload.length === 0) return null;
                    const d = payload[0].payload;
                    return (
                      <div className="bg-slate-900 text-white p-3.5 rounded-xl text-xs shadow-xl space-y-1.5 font-mono border border-slate-800">
                        <div className="text-[10px] text-slate-400 font-sans pb-1 border-b border-slate-800">{d.timestamp}</div>
                        <div>Actual <MathInline math="\mathrm{C}_6\mathrm{H}_6" />: <span className="font-bold text-teal-300">{d.actual} <MathInline math="\mu\mathrm{g/m}^3" /></span></div>
                        <div>Full OLS: <span className="text-slate-300">{d.predicted_ols} <MathInline math="\mu\mathrm{g/m}^3" /></span> (res: {d.residual_ols})</div>
                        <div>PCR (<MathInline math="k=3" />): <span className="text-teal-400 font-semibold">{d.predicted_pcr3} <MathInline math="\mu\mathrm{g/m}^3" /></span> (res: {d.residual_pcr3})</div>
                      </div>
                    );
                  }}
                />
                <ReferenceLine
                  segment={[{ x: 0, y: 0 }, { x: 45, y: 45 }]}
                  stroke="#94A3B8"
                  strokeDasharray="4 4"
                  strokeWidth={1.5}
                  label={{ value: '1:1 Agreement', position: 'insideTopLeft', fill: '#94a3b8', fontSize: 10 }}
                />
                <Scatter
                  name="Full OLS (k=8)"
                  data={charts.scatter_sample}
                  fill="#0D9488"
                  fillOpacity={0.5}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Points tightly clustered around the 1:1 diagonal verify high linear explanatory power.</span>
            <span className="font-mono text-teal-700 font-semibold">Test Sample: n = {charts.scatter_sample.length}</span>
          </div>
        </Card>

        {/* Viz 2: Actual vs. Predicted Time Series */}
        <Card
          title="Actual vs. Predicted Trajectory Over Time"
          subtitle="Chronological sequence across the untouched test horizon (January to April 2005)"
        >
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={charts.timeline_sample} margin={{ top: 15, right: 25, bottom: 25, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                <XAxis
                  dataKey="timestamp"
                  stroke="#94A3B8"
                  fontSize={10}
                  tickFormatter={(val) => val.split(' ')[0]}
                  interval={Math.floor(charts.timeline_sample.length / 5)}
                  label={{
                    value: 'Operational Timeline (Out-of-Time Test Set)',
                    position: 'insideBottom',
                    offset: -14,
                    fill: '#64748B',
                    fontSize: 11,
                  }}
                />
                <YAxis stroke="#94A3B8" fontSize={11} unit=" &mu;g/m&sup3;" />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0F172A', color: '#FFF', borderRadius: '0.75rem', fontSize: '11px', border: '1px solid #1E293B' }}
                  labelStyle={{ color: '#94A3B8', marginBottom: '4px' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Line
                  type="monotone"
                  dataKey="actual"
                  name="Certified Benzene (Reference)"
                  stroke="#0F172A"
                  strokeWidth={1.8}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="predicted_ols"
                  name="Full OLS (k*=8)"
                  stroke="#94A3B8"
                  strokeWidth={1.2}
                  strokeDasharray="3 3"
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="predicted_pcr3"
                  name="Compact PCR (k=3)"
                  stroke="#0D9488"
                  strokeWidth={1.6}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Notice Full OLS overprediction drift during late-winter weather peaks; PCR (<MathInline math="k = 3" />) stabilizes.</span>
            <span className="font-mono text-slate-600 font-semibold">Sample: {charts.timeline_sample.length} hours</span>
          </div>
        </Card>

        {/* Viz 3: Residual Distribution */}
        <Card
          title="Calibration Residual Distribution (Full OLS)"
          subtitle={<>Empirical error distribution <MathInline math="e_t = y_t - \hat{y}_t" /> vs. theoretical Gaussian PDF</>}
        >
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={charts.residuals_distribution} margin={{ top: 15, right: 25, bottom: 25, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                <XAxis
                  dataKey="bin_center"
                  stroke="#94A3B8"
                  fontSize={11}
                  unit=" &mu;g/m&sup3;"
                  label={{
                    value: 'Residual Error (e = y - y_hat) [µg/m³]',
                    position: 'insideBottom',
                    offset: -14,
                    fill: '#64748B',
                    fontSize: 11,
                  }}
                />
                <YAxis stroke="#94A3B8" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0F172A', color: '#FFF', borderRadius: '0.75rem', fontSize: '11px', border: '1px solid #1E293B' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <ReferenceLine x={0} stroke="#f43f5e" strokeWidth={1.5} label={{ value: 'Zero Error', fill: '#f43f5e', fontSize: 10 }} />
                <Bar dataKey="count" name="Observed Residual Frequency" fill="#0D9488" radius={[4, 4, 0, 0]} opacity={0.8} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>Mean Bias: -2.652 <MathInline math="\mu\mathrm{g/m}^3" /> · Standard Deviation: 1.059 <MathInline math="\mu\mathrm{g/m}^3" /></span>
            <span className="font-mono text-amber-600 font-medium">Asymmetric negative skew</span>
          </div>
        </Card>

        {/* Viz 4: PCA Scree & Explained Variance */}
        <Card
          title="PCA Explained Variance Spectrum (Scree Curve)"
          subtitle={<>Eigenvalue decay and cumulative feature variance captured across <MathInline math="k = 1, \dots, 8" /></>}
        >
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={charts.pca_variance} margin={{ top: 15, right: 25, bottom: 25, left: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                <XAxis
                  dataKey="k"
                  stroke="#94A3B8"
                  fontSize={11}
                  tickFormatter={(v) => `PC${v}`}
                  label={{
                    value: 'Principal Component (k)',
                    position: 'insideBottom',
                    offset: -14,
                    fill: '#64748B',
                    fontSize: 11,
                  }}
                />
                <YAxis stroke="#94A3B8" fontSize={11} unit=" %" domain={[0, 100]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0F172A', color: '#FFF', borderRadius: '0.75rem', fontSize: '11px', border: '1px solid #1E293B' }}
                />
                <Legend wrapperStyle={{ fontSize: '11px', paddingTop: '8px' }} />
                <Bar dataKey="variance_pct" name="Component Variance (%)" fill="#38BDF8" radius={[4, 4, 0, 0]} />
                <Line
                  type="monotone"
                  dataKey="cum_variance_pct"
                  name="Cumulative Variance (%)"
                  stroke="#0D9488"
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs text-slate-500">
            <span>First 3 components capture 94.04% of total sensor variance.</span>
            <span className="font-mono text-teal-700 font-medium"><MathInline math="k = 3" />: 94.04% · <MathInline math="k = 4" />: 96.50%</span>
          </div>
        </Card>
      </div>

      {/* 4. Full Model Comparison Table (Validation vs. Test Distinguished) */}
      <Card
        title={<>Comprehensive Benchmark Matrix across Component Count (<MathInline math="k = 1, \dots, 8" />)</>}
        subtitle="Rigorous comparison of Training, Validation (Tuning), and Untouched Test partitions matching reports/experiment_results.csv"
      >
        <div className="p-3.5 bg-amber-50/80 border border-amber-200 rounded-xl text-xs text-amber-950 mb-4 leading-relaxed">
          <strong>Academic Protocol Note:</strong> Model selection was conducted strictly on the validation partition, where <strong>Full OLS (<MathInline math="k^* = 8" />)</strong> achieved the lowest validation RMSE (2.024 <MathInline math="\mu\mathrm{g/m}^3" />). The lower test error observed in PCR (<MathInline math="k = 3" />) is an <em>exploratory post-hoc finding</em> demonstrating that truncating noisy principal components regularizes against seasonal collinear drift. It was not selected by the formal validation rule and must not be presented as a legitimate pre-test selection.
        </div>
        <div className="overflow-x-auto border border-slate-200/80 rounded-xl">
          <table className="w-full text-left text-xs">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold uppercase tracking-wider">
              <tr>
                <th className="py-3 px-4">Model Config</th>
                <th className="py-3 px-4">Cum. Var (%)</th>
                <th className="py-3 px-4">Train RMSE</th>
                <th className="py-3 px-4">Train <MathInline math="R^2" /></th>
                <th className="py-3 px-4 bg-teal-50/60 text-teal-900 border-x border-teal-100">Val RMSE (Tuning)</th>
                <th className="py-3 px-4 bg-teal-50/60 text-teal-900 border-r border-teal-100">Val <MathInline math="R^2" /></th>
                <th className="py-3 px-4 bg-slate-100/70 text-slate-900">Test RMSE (Frozen)</th>
                <th className="py-3 px-4 bg-slate-100/70 text-slate-900">Test <MathInline math="R^2" /></th>
                <th className="py-3 px-4">Protocol Role</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {charts.model_comparison.map((row) => {
                const isSelected = row.k === 8;
                const isPostHoc = row.k === 3 || row.k === 4;
                const isTrap = row.k === 7;

                return (
                  <tr
                    key={row.k}
                    className={`hover:bg-slate-50/80 transition-colors ${
                      isSelected ? 'bg-teal-50/40 font-semibold' : isPostHoc ? 'bg-sky-50/30' : ''
                    }`}
                  >
                    <td className="py-2.5 px-4 font-sans font-medium text-slate-900 flex items-center gap-2">
                      <span>PCR (<MathInline math={`k=${row.k}`} />)</span>
                      {isSelected && <Badge variant="teal" size="sm">Selected</Badge>}
                    </td>
                    <td className="py-2.5 px-4 text-slate-600">{row.cum_variance_pct.toFixed(2)}%</td>
                    <td className="py-2.5 px-4 text-slate-500">{row.train_rmse.toFixed(3)}</td>
                    <td className="py-2.5 px-4 text-slate-500">{row.train_r2.toFixed(4)}</td>

                    {/* Validation Partition Columns */}
                    <td className="py-2.5 px-4 bg-teal-50/50 text-teal-900 border-x border-teal-100 font-bold">
                      {row.val_rmse.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-4 bg-teal-50/50 text-teal-900 border-r border-teal-100">
                      {row.val_r2.toFixed(4)}
                    </td>

                    {/* Test Partition Columns */}
                    <td className="py-2.5 px-4 bg-slate-100/60 text-slate-900 font-bold">
                      {row.test_rmse.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-4 bg-slate-100/60 text-slate-900">
                      {row.test_r2.toFixed(4)}
                    </td>

                    <td className="py-2.5 px-4 font-sans">
                      {isSelected ? (
                        <Badge variant="teal" size="sm">
                          <span>Validation-Selected (<MathInline math="k^*" />)</span>
                        </Badge>
                      ) : isPostHoc ? (
                        <Badge variant="info" size="sm">Post-Hoc Regularized</Badge>
                      ) : isTrap ? (
                        <Badge variant="danger" size="sm">Collinear Overfit Trap</Badge>
                      ) : (
                        <Badge variant="outline" size="sm">Candidate</Badge>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </Card>

      {/* 5. Central Research Question & Scientific Findings Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <Card title="Central Research Question & Technical Synthesis">
            <div className="space-y-4 text-xs leading-relaxed text-slate-700">
              <blockquote className="p-4 bg-slate-50 rounded-xl border-l-4 border-teal-600 text-slate-800 font-medium italic">
                &ldquo;How does PCA-based dimensionality reduction affect the predictive accuracy and temporal stability of
                least-squares calibration for benzene concentration using real-world gas-sensor measurements?&rdquo;
              </blockquote>

              <div className="space-y-2">
                <h4 className="font-semibold text-slate-900 uppercase text-[11px] tracking-wider">
                  Core Empirical Findings
                </h4>
                <ul className="list-disc pl-4 space-y-2 text-slate-600">
                  <li>
                    <strong className="text-slate-800">Short-Term In-Sample Accuracy</strong>: Full-rank OLS (<MathInline math="k^* = 8" />) achieves
                    the highest training (<MathInline math="R^2 = 0.9826" />) and validation fit (<MathInline math="R^2 = 0.9423" />, RMSE = 2.024 <MathInline math="\mu\mathrm{g/m}^3" />) because
                    subtle chemical cross-sensitivity information is distributed across all five metal-oxide channels.
                  </li>
                  <li>
                    <strong className="text-slate-800">Out-of-Time Generalization Paradox</strong>: Over a 1-year test horizon,
                    Full OLS degrades to Test <MathInline math="R^2 = 0.8079" /> (RMSE = 2.855 <MathInline math="\mu\mathrm{g/m}^3" />). In contrast, truncating to
                    <MathInline math="k = 3" /> components (94.04% variance) acts as an implicit shrinkage regularizer, achieving Test <MathInline math="R^2 = 0.8626" />
                    and RMSE = 2.415 <MathInline math="\mu\mathrm{g/m}^3" /> (15.4% lower error).
                  </li>
                  <li>
                    <strong className="text-slate-800">The Collinear Overfit Trap (<MathInline math="k = 7" />)</strong>: Retaining 7 components
                    yields high validation accuracy (<MathInline math="R^2 = 0.9123" />), but collapses on the test set (<MathInline math="R^2 = 0.6225" />), proving that
                    trailing spectral directions fit transient seasonal temperature interactions.
                  </li>
                </ul>
              </div>
            </div>
          </Card>
        </div>

        {/* Scientific Limitations Card */}
        <div className="space-y-4">
          <Card title="Scientific Limitations & Caution">
            <div className="space-y-3 text-xs leading-relaxed text-slate-600">
              <Alert variant="warning" title="Drift vs. Weather Confounding">
                {metrics.drift_caveat}
              </Alert>

              <Alert variant="info" title="Serial Autocorrelation">
                The Durbin-Watson statistic is <strong>{metrics.durbin_watson} &lt;&lt; 2.0</strong>, confirming extreme positive
                serial correlation (<MathInline math="\rho \approx 0.90" />). Standard OLS error bars underestimate parameter variance by &approx; <MathInline math="\sqrt{\frac{1+\rho}{1-\rho}} \approx 4.29\times" />.
              </Alert>

              <div className="p-3 bg-slate-50 border border-slate-200/80 rounded-xl text-[11px] text-slate-500 flex items-start gap-2">
                <Info className="w-4 h-4 text-slate-400 shrink-0 mt-0.5" />
                <span>Zero ground-truth certified reference analyzer columns (GT) are used as inputs, preventing target leakage.</span>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
};
