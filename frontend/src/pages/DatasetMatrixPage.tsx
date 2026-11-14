import React, { useEffect, useState, useCallback } from 'react';
import {
  Database,
  Calendar,
  Layers,
  ShieldCheck,
  ChevronLeft,
  ChevronRight,
  Binary,
  Sparkles,
  Table,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { apiClient } from '../services/apiClient';
import {
  MatrixPreviewResponse,
  DatasetAuditResponse,
  MatrixRowSample,
  MatrixFeatureParam,
} from '../types/api';
import { Card } from '../components/ui/Card';
import { MetricCard } from '../components/ui/MetricCard';
import { Badge } from '../components/ui/Badge';
import { Alert } from '../components/ui/Alert';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { ErrorState } from '../components/ui/ErrorState';
import { MathInline, MathBlock } from '../components/ui/Math';
import { tabContentVariants } from '../lib/motion';

export const DatasetMatrixPage: React.FC = () => {
  // Navigation Tabs
  const [activeTab, setActiveTab] = useState<'matrix' | 'partitions' | 'quality'>('matrix');

  // Matrix View State
  const [viewMode, setViewMode] = useState<'raw' | 'standardized'>('raw');
  const [partitionFilter, setPartitionFilter] = useState<'all' | 'train' | 'val' | 'test'>('all');
  const [pageOffset, setPageOffset] = useState<number>(0);
  const [pageSize] = useState<number>(8);
  const [selectedCell, setSelectedCell] = useState<{
    rowIdx: number;
    featIdx: number;
  } | null>(null);

  // API State
  const [matrixData, setMatrixData] = useState<MatrixPreviewResponse | null>(null);
  const [auditData, setAuditData] = useState<DatasetAuditResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isMatrixLoading, setIsMatrixLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  // Fetch initial data
  const loadInitialData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [matrixRes, auditRes] = await Promise.all([
        apiClient.getMatrixPreview(0, pageSize, partitionFilter),
        apiClient.getDatasetAudit(),
      ]);
      setMatrixData(matrixRes);
      setAuditData(auditRes);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to retrieve dataset matrices from backend');
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch matrix slice on filter or page change
  const fetchMatrixSlice = useCallback(async (newOffset: number, newPartition: string) => {
    setIsMatrixLoading(true);
    try {
      const res = await apiClient.getMatrixPreview(newOffset, pageSize, newPartition);
      setMatrixData(res);
      setPageOffset(newOffset);
    } catch (err) {
      console.error('Error fetching matrix slice:', err);
    } finally {
      setIsMatrixLoading(false);
    }
  }, [pageSize]);

  useEffect(() => {
    loadInitialData();
  }, []);

  const handlePartitionChange = (newPartition: 'all' | 'train' | 'val' | 'test') => {
    setPartitionFilter(newPartition);
    setPageOffset(0);
    fetchMatrixSlice(0, newPartition);
  };

  const handlePageChange = (newOffset: number) => {
    fetchMatrixSlice(newOffset, partitionFilter);
  };

  if (isLoading) {
    return <LoadingSpinner label="Formulating empirical observation matrix and feature tensors from backend..." />;
  }

  if (error || !matrixData) {
    return (
      <ErrorState
        title="Dataset Connection Error"
        message={error || 'Unable to establish connection with the AirSense backend matrix service.'}
        onRetry={loadInitialData}
      />
    );
  }

  const selectedFeatureParam: MatrixFeatureParam | null =
    selectedCell && matrixData ? matrixData.features[selectedCell.featIdx] : null;
  const selectedRowSample: MatrixRowSample | null =
    selectedCell && matrixData ? matrixData.rows[selectedCell.rowIdx] : null;

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* 1. Header & Problem Statement */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            <span className="text-[11px] font-bold text-teal-700 tracking-wider uppercase font-mono bg-teal-50 px-2.5 py-0.5 rounded-full border border-teal-200">
              Data Representation
            </span>
            <span className="text-slate-300">&middot;</span>
            <span className="text-xs text-slate-500 font-medium">UCI Air Quality Dataset #360</span>
            <span className="text-slate-300">&middot;</span>
            <span className="text-[11px] font-mono text-slate-500">13 Months (March 2004 – April 2005)</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 font-sans">
            Dataset &amp; Mathematical Matrix Representation
          </h1>
          <p className="text-sm text-slate-600 mt-1 max-w-3xl leading-relaxed">
            In field environmental monitoring, a multi-sensor array of solid-state metal-oxide chemical sensors
            records electrical responses to ambient air. The AirSense project transforms these continuous physical
            measurements into a well-defined matrix <MathInline math="X" /> and target vector <MathInline math="y" /> to calibrate
            benzene concentration estimates against a certified reference analyzer.
          </p>
        </div>

        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-3 shrink-0">
          <div className="p-3 bg-slate-900 text-white rounded-xl shadow-md border border-slate-800 text-xs font-mono space-y-1.5">
            <div className="text-[10px] text-teal-400 uppercase tracking-wider font-bold">Matrix Dimensions</div>
            <div className="text-sm font-bold text-white tracking-wide">
              <MathInline math={`X \\in \\mathbb{R}^{${matrixData.total_usable_rows} \\times 8}`} />
            </div>
            <div className="text-[11px] text-slate-300">
              <MathInline math={`y \\in \\mathbb{R}^{${matrixData.total_usable_rows} \\times 1}`} />
            </div>
          </div>
        </div>
      </div>

      {/* 2. Top-level KPI Summary */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Total Hourly Slots"
          value={matrixData.total_raw_rows.toLocaleString()}
          unit="hours"
          subtitle="Continuous recording period"
          icon={<Calendar className="w-4 h-4 text-slate-500" />}
        />
        <MetricCard
          label="Usable Supervised Rows (n)"
          value={matrixData.total_usable_rows.toLocaleString()}
          unit="rows"
          badge={{ text: '96.09% complete', variant: 'success' }}
          subtitle="366 blackout rows pruned"
          icon={<Database className="w-4 h-4 text-slate-500" />}
        />
        <MetricCard
          label="Predictor Features (d)"
          value={matrixData.feature_count}
          unit="columns"
          subtitle="5 MOX sensors + 3 Meteorology"
          icon={<Layers className="w-4 h-4 text-slate-500" />}
        />
        <MetricCard
          label="Target Column"
          value={<MathInline math="\mathrm{C}_6\mathrm{H}_6\text{(GT)}" />}
          unit={<MathInline math="\mu\mathrm{g/m}^3" />}
          badge={{ text: 'Certified Reference', variant: 'teal' }}
          subtitle="Ground-truth gas analyzer"
          icon={<ShieldCheck className="w-4 h-4 text-teal-600" />}
        />
      </div>

      {/* 3. Section Navigation Tabs */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-2 shadow-xs">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-2">
          <button
            onClick={() => setActiveTab('matrix')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'matrix'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <Binary className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">1. Matrix Explorer (X &amp; y)</span>
              <span className={`text-[10px] ${activeTab === 'matrix' ? 'text-slate-300' : 'text-slate-500'}`}>
                Real observations preview &amp; standardisation
              </span>
            </div>
          </button>

          <button
            onClick={() => setActiveTab('partitions')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'partitions'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <Calendar className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">2. Chronological Partitions</span>
              <span className={`text-[10px] ${activeTab === 'partitions' ? 'text-slate-300' : 'text-slate-500'}`}>
                Train (60%), Val (20%), Held-Out Test (20%)
              </span>
            </div>
          </button>

          <button
            onClick={() => setActiveTab('quality')}
            className={`p-3 rounded-xl text-left flex items-center gap-2.5 transition-all ${
              activeTab === 'quality'
                ? 'bg-slate-900 text-white shadow-md'
                : 'bg-slate-50 hover:bg-slate-100 text-slate-700'
            }`}
          >
            <Table className="w-4 h-4 text-teal-400" />
            <div>
              <span className="text-xs font-bold block">3. Ingestion &amp; Data Audit</span>
              <span className={`text-[10px] ${activeTab === 'quality' ? 'text-slate-300' : 'text-slate-500'}`}>
                Sentinel -200 handling &amp; baseline stats
              </span>
            </div>
          </button>
        </div>
      </div>

      {/* 4. Tab Content */}
      <AnimatePresence mode="wait">
        {/* =============================================================== */}
        {/* TAB 1: INTERACTIVE MATRIX EXPLORER (X & y) */}
        {/* =============================================================== */}
        {activeTab === 'matrix' && (
          <motion.div
            key="matrix_tab"
            variants={tabContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-6"
          >
            {/* Mathematical Matrix Representation Card */}
            <Card
              title="Mathematical Representation of Sensor Measurements"
              subtitle="Showing how real physical observations map directly to design matrix X and target vector y"
            >
              <div className="space-y-4">
                {/* Mathematical formulation banner */}
                <div className="p-4 bg-slate-900 text-slate-100 rounded-xl font-mono text-xs border border-slate-800 space-y-2">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-800 pb-2">
                    <span className="text-teal-400 font-bold uppercase tracking-wider text-[11px]">
                      Canonical Linear Formulation
                    </span>
                    <span className="text-slate-400 text-[11px]">
                      Total usable observations: <MathInline math={`n = ${matrixData.total_usable_rows}`} />, features: <MathInline math="d = 8" />
                    </span>
                  </div>
                  <div className="text-sm font-semibold text-white pt-1 flex items-center gap-4 flex-wrap">
                    <MathInline math={`X \\in \\mathbb{R}^{${matrixData.total_usable_rows} \\times 8}`} />
                    <span className="text-slate-500">,</span>
                    <MathInline math={`y \\in \\mathbb{R}^{${matrixData.total_usable_rows} \\times 1}`} />
                  </div>
                  <p className="text-[11px] text-slate-300 leading-relaxed font-sans">
                    Each row <MathInline math="\mathbf{x}_i^\mathsf{T} \in \mathbb{R}^8" /> represents one valid hourly
                    observation across 5 metal-oxide sensors and 3 meteorological variables. The target scalar{' '}
                    <MathInline math="y_i" /> contains the simultaneous ground-truth benzene reading from a certified co-located reference analyzer.
                  </p>
                </div>

                {/* Interactive Controls Bar: Raw vs Standardized & Partition Filter */}
                <div className="flex flex-wrap items-center justify-between gap-4 p-3 bg-slate-50 border border-slate-200/80 rounded-xl text-xs">
                  {/* View Mode Toggle */}
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-slate-700">Display Values:</span>
                    <div className="inline-flex rounded-lg border border-slate-200 bg-white p-0.5">
                      <button
                        onClick={() => setViewMode('raw')}
                        className={`px-3 py-1 rounded-md font-medium text-xs transition-all ${
                          viewMode === 'raw'
                            ? 'bg-slate-900 text-white shadow-xs'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        Raw Features (Physical Units)
                      </button>
                      <button
                        onClick={() => setViewMode('standardized')}
                        className={`px-3 py-1 rounded-md font-medium text-xs transition-all ${
                          viewMode === 'standardized'
                            ? 'bg-slate-900 text-white shadow-xs'
                            : 'text-slate-600 hover:text-slate-900'
                        }`}
                      >
                        Standardized (z-scores)
                      </button>
                    </div>
                  </div>

                  {/* Partition Filter */}
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-slate-700">Filter Partition:</span>
                    <div className="inline-flex rounded-lg border border-slate-200 bg-white p-0.5">
                      {(['all', 'train', 'val', 'test'] as const).map((p) => (
                        <button
                          key={p}
                          onClick={() => handlePartitionChange(p)}
                          className={`px-2.5 py-1 rounded-md font-medium text-xs uppercase tracking-wider transition-all ${
                            partitionFilter === p
                              ? 'bg-teal-600 text-white shadow-xs'
                              : 'text-slate-600 hover:text-slate-900'
                          }`}
                        >
                          {p}
                        </button>
                      ))}
                    </div>
                  </div>

                  {/* Row Stepper Controls */}
                  <div className="flex items-center gap-2">
                    <button
                      onClick={() => handlePageChange(Math.max(0, pageOffset - pageSize))}
                      disabled={pageOffset === 0 || isMatrixLoading}
                      className="p-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-white disabled:opacity-40 transition-colors"
                      title="Previous rows"
                    >
                      <ChevronLeft className="w-4 h-4" />
                    </button>
                    <span className="font-mono text-slate-600 text-xs">
                      Rows {pageOffset + 1}–{Math.min(pageOffset + pageSize, matrixData.total_rows)} of {matrixData.total_rows.toLocaleString()}
                    </span>
                    <button
                      onClick={() => handlePageChange(pageOffset + pageSize)}
                      disabled={pageOffset + pageSize >= matrixData.total_rows || isMatrixLoading}
                      className="p-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-white disabled:opacity-40 transition-colors"
                      title="Next rows"
                    >
                      <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                </div>

                {/* Matrix Preview Table with Explicit Matrix Bracket Aesthetics */}
                <div className="relative border border-slate-200/90 rounded-xl overflow-hidden bg-white shadow-xs">
                  {isMatrixLoading && (
                    <div className="absolute inset-0 bg-white/70 backdrop-blur-2xs flex items-center justify-center z-10">
                      <div className="flex items-center gap-2 text-xs font-mono text-teal-800 bg-white px-3 py-1.5 rounded-xl border border-teal-200 shadow-sm">
                        <div className="w-3.5 h-3.5 border-2 border-teal-500 border-t-transparent rounded-full animate-spin" />
                        <span>Fetching matrix slice...</span>
                      </div>
                    </div>
                  )}

                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs font-mono border-collapse">
                      <thead>
                        {/* Group Header */}
                        <tr className="border-b border-slate-200 bg-slate-100/70 text-[10px] uppercase tracking-wider text-slate-500 font-sans">
                          <th className="py-2 px-3 border-r border-slate-200 text-center" colSpan={3}>
                            Observation Identification
                          </th>
                          <th className="py-2 px-3 border-r border-slate-200 text-center bg-teal-50/50 text-teal-900 font-bold" colSpan={8}>
                            <span>Predictor Matrix </span>
                            <MathInline math={`X \\in \\mathbb{R}^{${matrixData.total_rows} \\times 8}`} />
                            <span className="text-[9px] font-normal block lowercase text-teal-700">
                              {viewMode === 'raw' ? '(physical units: mV, °C, %, g/m³)' : <span>(dimensionless standard scores <MathInline math="z_{ij}" />)</span>}
                            </span>
                          </th>
                          <th className="py-2 px-3 text-center bg-indigo-50/60 text-indigo-900 font-bold">
                            <span>Target Vector </span>
                            <MathInline math="y" />
                            <span className="text-[9px] font-normal block lowercase text-indigo-700">
                              (reference <MathInline math="\mathrm{C}_6\mathrm{H}_6" /> in <MathInline math="\mu\mathrm{g/m}^3" />)
                            </span>
                          </th>
                        </tr>

                        {/* Column Names & Symbols */}
                        <tr className="border-b border-slate-200 bg-slate-50 text-[11px] text-slate-600 font-sans">
                          <th className="py-2 px-3">
                            Row <MathInline math="i" />
                          </th>
                          <th className="py-2 px-3">Timestamp</th>
                          <th className="py-2 px-3 border-r border-slate-200">Partition</th>

                          {matrixData.features.map((feat, fIdx) => (
                            <th
                              key={feat.name}
                              className={`py-2 px-3 text-right cursor-pointer transition-colors ${
                                selectedCell?.featIdx === fIdx ? 'bg-teal-100 text-teal-950 font-bold' : ''
                              }`}
                              onClick={() => setSelectedCell({ rowIdx: 0, featIdx: fIdx })}
                              title={`${feat.name}: ${feat.description}`}
                            >
                              <div className="font-mono text-teal-800 font-bold">
                                <MathInline math={`x_{${fIdx + 1}}`} />
                              </div>
                              <div className="text-[10px] text-slate-700 truncate max-w-[80px]" title={feat.name}>
                                {feat.name.split('(')[0]}
                              </div>
                              <div className="text-[9px] text-slate-400 font-mono">
                                {viewMode === 'raw' ? feat.unit : 'z-score'}
                              </div>
                            </th>
                          ))}

                          <th className="py-2 px-3 text-right bg-indigo-50/40 border-l border-indigo-100 font-bold text-indigo-950">
                            <div className="font-mono text-indigo-700">
                              <MathInline math="y_i" />
                            </div>
                            <div className="text-[10px]"><MathInline math="\mathrm{C}_6\mathrm{H}_6\text{(GT)}" /></div>
                            <div className="text-[9px] text-indigo-500 font-mono"><MathInline math="\mu\mathrm{g/m}^3" /></div>
                          </th>
                        </tr>
                      </thead>

                      <tbody className="divide-y divide-slate-100">
                        {matrixData.rows.map((row, rIdx) => {
                          const isSelectedRow = selectedCell?.rowIdx === rIdx;

                          return (
                            <tr
                              key={row.row_index}
                              className={`transition-colors ${
                                isSelectedRow ? 'bg-teal-50/30' : 'hover:bg-slate-50/70'
                              }`}
                            >
                              <td className="py-2 px-3 text-slate-500 font-semibold">{row.row_index}</td>
                              <td className="py-2 px-3 text-slate-700 whitespace-nowrap">{row.timestamp}</td>
                              <td className="py-2 px-3 border-r border-slate-200 font-sans">
                                <span
                                  className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
                                    row.partition === 'train'
                                      ? 'bg-teal-100 text-teal-800'
                                      : row.partition === 'val'
                                      ? 'bg-amber-100 text-amber-800'
                                      : 'bg-indigo-100 text-indigo-800'
                                  }`}
                                >
                                  {row.partition}
                                </span>
                              </td>

                              {/* 8 Feature Values */}
                              {(viewMode === 'raw' ? row.raw_x : row.standardized_x).map((val, fIdx) => {
                                const isSelected =
                                  selectedCell?.rowIdx === rIdx && selectedCell?.featIdx === fIdx;

                                return (
                                  <td
                                    key={fIdx}
                                    onClick={() => setSelectedCell({ rowIdx: rIdx, featIdx: fIdx })}
                                    className={`py-2 px-3 text-right cursor-pointer transition-colors ${
                                      isSelected
                                        ? 'bg-teal-600 text-white font-bold ring-2 ring-teal-600 rounded-sm'
                                        : selectedCell?.featIdx === fIdx
                                        ? 'bg-teal-50/60 text-teal-950'
                                        : 'text-slate-800'
                                    }`}
                                  >
                                    {val.toFixed(viewMode === 'raw' ? 1 : 3)}
                                  </td>
                                );
                              })}

                              {/* Target Vector Element */}
                              <td className="py-2 px-3 text-right font-bold text-indigo-900 bg-indigo-50/30 border-l border-indigo-100">
                                {row.raw_y.toFixed(2)}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  {/* Cell Inspector Drawer */}
                  {selectedFeatureParam && selectedRowSample && (
                    <div className="p-4 bg-slate-900 text-white border-t border-slate-800 text-xs flex flex-col md:flex-row items-start md:items-center justify-between gap-4 font-mono">
                      <div className="space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="text-teal-400 font-bold uppercase tracking-wider text-[11px]">
                            Selected Matrix Cell: <MathInline math={`X[${selectedRowSample.row_index}, ${selectedCell?.featIdx}]`} />
                          </span>
                          <span className="text-slate-500">&middot;</span>
                          <span className="text-slate-300 font-sans">{selectedFeatureParam.name}</span>
                        </div>
                        <p className="text-[11px] text-slate-400 font-sans">
                          {selectedFeatureParam.description}
                        </p>
                      </div>

                      <div className="flex flex-wrap items-center gap-4 shrink-0 text-slate-300">
                        <div>
                          <span className="text-slate-500 block text-[10px]">Raw Value:</span>
                          <span className="text-white font-bold">
                            {selectedRowSample.raw_x[selectedCell?.featIdx ?? 0].toFixed(2)} {selectedFeatureParam.unit}
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-500 block text-[10px]">Training Mean (<MathInline math="\mu_j" />):</span>
                          <span>{selectedFeatureParam.train_mean.toFixed(2)}</span>
                        </div>
                        <div>
                          <span className="text-slate-500 block text-[10px]">Training Std (<MathInline math="\sigma_j" />):</span>
                          <span>{selectedFeatureParam.train_std.toFixed(2)}</span>
                        </div>
                        <div className="pl-2 border-l border-slate-700">
                          <span className="text-teal-400 block text-[10px]">Standardized Score (<MathInline math="z_{ij}" />):</span>
                          <span className="text-teal-300 font-bold">
                            {selectedRowSample.standardized_x[selectedCell?.featIdx ?? 0].toFixed(4)}
                          </span>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Standardisation Formula & Physical Units Context */}
                <div className="p-4 bg-slate-50 border border-slate-200/90 rounded-xl space-y-3 text-xs leading-relaxed text-slate-700">
                  <div className="flex items-center gap-2 text-slate-900 font-bold">
                    <Sparkles className="w-4 h-4 text-teal-600" />
                    <span>Mathematical Standardization &amp; Unit Invariance</span>
                  </div>
                  <div className="font-mono text-xs bg-white p-3 rounded-lg border border-slate-200 space-y-2">
                    <div className="font-semibold text-teal-800">
                      <MathBlock math="z_{ij} = \frac{x_{ij} - \mu_{j,\mathrm{train}}}{\sigma_{j,\mathrm{train}}}" />
                    </div>
                    <p className="text-[11px] text-slate-500 font-sans text-center">
                      Where <MathInline math="\mu_{j,\mathrm{train}}" /> and <MathInline math="\sigma_{j,\mathrm{train}}" /> are calculated strictly from the 5,394 training partition rows.
                    </p>
                  </div>
                  <p>
                    <strong>Why standardisation matters:</strong> Raw sensor resistances range up to 2,000+ mV (e.g., <code>PT08.S4</code> has a mean of 1,615.7),
                    whereas temperature ranges from 0°C to 40°C (mean 23.1). Without standardization, unscaled Principal Component Analysis
                    would attribute 88.5% of variance purely to sensor resistance scale differences. Standardizing transforms each column into a
                    dimensionless <MathInline math="z" />-score with zero mean and unit variance.
                  </p>
                  <p className="text-[11px] text-slate-500">
                    <em>Important Distinction:</em> Standardization is crucial for PCA and SVD dimensionality reduction, but is not mathematically required for
                    ordinary least-squares regression, which is scale-equivariant.
                  </p>
                </div>
              </div>
            </Card>

            {/* Feature Parameter Summary Table */}
            <Card
              title="Predictor Feature Space Parameters (Fitted on Training Partition)"
              subtitle="All 8 eligible features with physical descriptions, units, and baseline standardization statistics"
            >
              <div className="overflow-x-auto border border-slate-200/80 rounded-xl">
                <table className="w-full text-left text-xs font-mono">
                  <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold uppercase text-[11px]">
                    <tr>
                      <th className="py-2.5 px-3">Symbol</th>
                      <th className="py-2.5 px-3 font-sans">Feature Column</th>
                      <th className="py-2.5 px-3 font-sans">Sensor Technology</th>
                      <th className="py-2.5 px-3">Unit</th>
                      <th className="py-2.5 px-3 text-right font-sans">Training Mean (<MathInline math="\mu_j" />)</th>
                      <th className="py-2.5 px-3 text-right font-sans">Training Std (<MathInline math="\sigma_j" />)</th>
                      <th className="py-2.5 px-3 font-sans">Role in Pipeline</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-100">
                    {matrixData.features.map((feat) => (
                      <tr key={feat.name} className="hover:bg-slate-50/80 transition-colors">
                        <td className="py-2 px-3 font-bold text-teal-800">
                          <MathInline math={`x_{${feat.symbol.replace(/[^0-9]/g, '') || 'j'}}`} />
                        </td>
                        <td className="py-2 px-3 font-sans font-semibold text-slate-900">{feat.name}</td>
                        <td className="py-2 px-3 font-sans text-slate-600 text-[11px]">{feat.description}</td>
                        <td className="py-2 px-3 text-slate-500">{feat.unit}</td>
                        <td className="py-2 px-3 text-right text-slate-800">{feat.train_mean.toFixed(2)}</td>
                        <td className="py-2 px-3 text-right text-slate-800">{feat.train_std.toFixed(2)}</td>
                        <td className="py-2 px-3 font-sans">
                          <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-teal-50 text-teal-700 border border-teal-200">
                            Predictor Feature
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
        {/* TAB 2: CHRONOLOGICAL PARTITIONS & ANTI-LEAKAGE */}
        {/* =============================================================== */}
        {activeTab === 'partitions' && (
          <motion.div
            key="partitions_tab"
            variants={tabContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-6"
          >
            <Card
              title="Chronological Partitioning Protocol (Zero-Data-Leakage)"
              subtitle="Contiguous out-of-time splits enforcing strict temporal guardrails"
            >
              <div className="space-y-6">
                {/* Visual Timeline Partition Ribbon */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-slate-500 font-mono">
                    <span>March 10, 2004 18:00</span>
                    <span>13 Continuous Months (389.8 Days)</span>
                    <span>April 4, 2005 14:00</span>
                  </div>

                  <div className="h-10 w-full rounded-xl overflow-hidden flex shadow-inner font-mono text-xs font-bold text-white text-center">
                    <div className="bg-teal-600 flex items-center justify-center w-[60%]" title="Training: 60%">
                      Training (5,394 hrs &middot; 60%)
                    </div>
                    <div className="bg-amber-600 flex items-center justify-center w-[20%]" title="Validation: 20%">
                      Val (1,798 hrs &middot; 20%)
                    </div>
                    <div className="bg-indigo-600 flex items-center justify-center w-[20%]" title="Held-Out Test: 20%">
                      Held-Out Test (1,799 hrs &middot; 20%)
                    </div>
                  </div>
                </div>

                {/* 3 Partition Detail Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {matrixData.partitions.map((part) => (
                    <div
                      key={part.name}
                      className="p-4 bg-white border border-slate-200 rounded-xl space-y-3 shadow-xs"
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-bold text-xs uppercase tracking-wider text-slate-900 font-sans">
                          {part.label}
                        </span>
                        <Badge
                          variant={part.name === 'train' ? 'teal' : part.name === 'val' ? 'warning' : 'default'}
                          size="sm"
                        >
                          {part.pct}%
                        </Badge>
                      </div>

                      <div className="space-y-1 font-mono text-xs text-slate-600">
                        <div className="flex justify-between py-1 border-b border-slate-100">
                          <span className="font-sans text-slate-500">Hourly Records:</span>
                          <span className="font-bold text-slate-900">{part.row_count.toLocaleString()}</span>
                        </div>
                        <div className="flex justify-between py-1 border-b border-slate-100">
                          <span className="font-sans text-slate-500">Start Timestamp:</span>
                          <span>{part.start_date}</span>
                        </div>
                        <div className="flex justify-between py-1">
                          <span className="font-sans text-slate-500">End Timestamp:</span>
                          <span>{part.end_date}</span>
                        </div>
                      </div>

                      <div className="pt-2 text-[11px] text-slate-500 leading-snug font-sans">
                        {part.name === 'train' &&
                          'Used exclusively to fit OLS regression weights, standardizer means/stds, and PCA singular vectors.'}
                        {part.name === 'val' &&
                          'Used strictly for model tuning and selection (k* component selection) prior to final evaluation.'}
                        {part.name === 'test' &&
                          'Held-out out-of-time evaluation set scored strictly with frozen training parameters to test generalization.'}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Anti-Leakage Guardrails Alert */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <Alert variant="info" title="Why Random Cross-Validation is Prohibited">
                    In time-series sensor calibration, standard random k-fold cross-validation randomly samples hours
                    adjacent to test points. High serial autocorrelation (<MathInline math="\rho \approx 0.90" />) causes adjacent hours to share
                    near-identical noise, leading to artificial, hyper-optimistic error estimates. AirSense enforces strictly
                    contiguous, chronological partitions: <MathInline math="\max(t_{\mathrm{train}}) < \min(t_{\mathrm{val}}) < \max(t_{\mathrm{val}}) < \min(t_{\mathrm{test}})" />.
                  </Alert>

                  <Alert variant="warning" title="Exclusion of Reference Analyzer Channels">
                    The dataset contains certified reference measurements for other gases: CO(GT), NMHC(GT), NOx(GT), and NO2(GT).
                    To prevent target contamination and data leakage, these reference columns are strictly excluded from design matrix <MathInline math="X" />.
                    Only raw metal-oxide sensor responses (PT08.S1 through PT08.S5) and meteorological variables are eligible predictors.
                  </Alert>
                </div>
              </div>
            </Card>
          </motion.div>
        )}

        {/* =============================================================== */}
        {/* TAB 3: DATA QUALITY & INGESTION AUDIT */}
        {/* =============================================================== */}
        {activeTab === 'quality' && auditData && (
          <motion.div
            key="quality_tab"
            variants={tabContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="space-y-6"
          >
            {/* Column Missingness Profile */}
            <Card
              title="Ingestion Cleaning & Missingness Profile"
              subtitle="Handling of European semicolon delimiters, comma decimal separators, and numeric -200 sentinels"
            >
              <div className="space-y-4">
                <div className="p-3 bg-teal-50 border border-teal-200 rounded-xl text-xs text-teal-950 font-sans leading-relaxed">
                  <strong>The 366 Blackout Hours Finding:</strong> In the raw dataset of 9,357 hourly slots, exactly 366 rows
                  have missing values in the target benzene column <MathInline math="\mathrm{C}_6\mathrm{H}_6\text{(GT)}" />. These 366 hours represent simultaneous instrument blackouts
                  when the reference gas chromatograph was offline. Pruning these 366 rows leaves <strong>8,991 usable supervised hours</strong> with
                  <strong> ZERO missing values</strong> across any of the 8 predictor features.
                </div>

                <div className="overflow-x-auto border border-slate-200/80 rounded-xl">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold uppercase text-[11px]">
                      <tr>
                        <th className="py-2.5 px-3.5 font-sans">Column Name</th>
                        <th className="py-2.5 px-3.5 font-sans">Role / Category</th>
                        <th className="py-2.5 px-3.5">Dtype</th>
                        <th className="py-2.5 px-3.5 text-right">Missing Count</th>
                        <th className="py-2.5 px-3.5 text-right">Missing %</th>
                        <th className="py-2.5 px-3.5 text-right">Valid Count</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {auditData.missingness_profile.map((col) => {
                        const isTarget = col.name === 'C6H6(GT)';
                        const isExcluded = col.category.includes('Excluded');

                        return (
                          <tr key={col.name} className="hover:bg-slate-50/80 transition-colors">
                            <td className="py-2.5 px-3.5 font-sans font-medium text-slate-900 flex items-center gap-1.5">
                              <span>{col.name}</span>
                              {isTarget && <Badge variant="teal" size="sm">Target</Badge>}
                            </td>
                            <td className="py-2.5 px-3.5 font-sans">
                              {isExcluded ? (
                                <Badge variant="warning" size="sm">Excluded (GT)</Badge>
                              ) : (
                                <span className="text-slate-600 text-[11px]">{col.category}</span>
                              )}
                            </td>
                            <td className="py-2.5 px-3.5 text-slate-500 text-[11px]">{col.dtype}</td>
                            <td className="py-2.5 px-3.5 text-right text-slate-700">{col.missing_count}</td>
                            <td className="py-2.5 px-3.5 text-right">
                              <span className={col.missing_pct > 0 ? 'text-amber-600 font-bold' : 'text-slate-600'}>
                                {col.missing_pct.toFixed(2)}%
                              </span>
                            </td>
                            <td className="py-2.5 px-3.5 text-right text-slate-600">{col.valid_count}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </Card>

            {/* Target Benzene Distribution & Feature Correlations */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card
                title={<span>Target Benzene <MathInline math="\mathrm{C}_6\mathrm{H}_6\text{(GT)}" /> Parametric Distribution</span>}
                subtitle={<span>Summary statistics of certified reference analyzer (<MathInline math="\mu\mathrm{g/m}^3" />)</span>}
              >
                <div className="space-y-3 font-mono text-xs">
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <span className="text-slate-500 font-sans">Usable Observation Count</span>
                    <span className="font-semibold text-slate-900">{auditData.target_distribution.count.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <span className="text-slate-500 font-sans">Mean Concentration</span>
                    <span className="font-semibold text-teal-700">{auditData.target_distribution.mean.toFixed(3)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <span className="text-slate-500 font-sans">Standard Deviation</span>
                    <span className="font-semibold text-slate-700">{auditData.target_distribution.std.toFixed(3)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <span className="text-slate-500 font-sans">25th Percentile (Q1)</span>
                    <span className="text-slate-700">{auditData.target_distribution.p25.toFixed(2)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <span className="text-slate-500 font-sans">Median (Q2)</span>
                    <span className="text-slate-700">{auditData.target_distribution.p50.toFixed(2)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
                  </div>
                  <div className="flex justify-between py-1.5 border-b border-slate-100">
                    <span className="text-slate-500 font-sans">75th Percentile (Q3)</span>
                    <span className="text-slate-700">{auditData.target_distribution.p75.toFixed(2)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
                  </div>
                  <div className="flex justify-between py-1.5">
                    <span className="text-slate-500 font-sans">Maximum Recorded Concentration</span>
                    <span className="font-bold text-rose-600">{auditData.target_distribution.max.toFixed(2)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
                  </div>
                </div>
              </Card>

              <Card
                title="Predictor Feature Linear Correlation with Benzene"
                subtitle={<span>Pearson correlation coefficient <MathInline math="r" /> with target <MathInline math="\mathrm{C}_6\mathrm{H}_6\text{(GT)}" /></span>}
              >
                <div className="overflow-x-auto border border-slate-200/80 rounded-xl">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold uppercase text-[11px]">
                      <tr>
                        <th className="py-2 px-3 font-sans">Feature Column</th>
                        <th className="py-2 px-3 text-right">Pearson <MathInline math="r" /></th>
                        <th className="py-2 px-3 text-right">Linear Association</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {auditData.feature_correlations.map((corr) => (
                        <tr key={corr.feature} className="hover:bg-slate-50/80 transition-colors">
                          <td className="py-2 px-3 font-sans font-semibold text-slate-900">{corr.feature}</td>
                          <td className="py-2 px-3 text-right font-bold text-teal-800">
                            {corr.pearson_r.toFixed(4)}
                          </td>
                          <td className="py-2 px-3 text-right font-sans">
                            <span
                              className={`px-2 py-0.5 rounded-full text-[10px] font-bold ${
                                corr.abs_r > 0.8
                                  ? 'bg-emerald-100 text-emerald-800'
                                  : corr.abs_r > 0.5
                                  ? 'bg-teal-50 text-teal-700'
                                  : 'bg-slate-100 text-slate-600'
                              }`}
                            >
                              {corr.abs_r > 0.8 ? 'Very Strong' : corr.abs_r > 0.5 ? 'Moderate' : 'Weak / Inverse'}
                            </span>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
