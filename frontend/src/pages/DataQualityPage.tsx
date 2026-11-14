import React, { useEffect, useState } from 'react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Cell,
} from 'recharts';
import {
  Database,
  Calendar,
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Layers,
  CheckCircle2,
} from 'lucide-react';
import { apiClient } from '../services/apiClient';
import { DatasetAuditResponse, PaginatedRowsResponse } from '../types/api';
import { Card } from '../components/ui/Card';
import { MetricCard } from '../components/ui/MetricCard';
import { Badge } from '../components/ui/Badge';
import { Alert } from '../components/ui/Alert';
import { LoadingSpinner } from '../components/ui/LoadingSpinner';
import { ErrorState } from '../components/ui/ErrorState';
import { MathInline } from '../components/ui/Math';

export const DataQualityPage: React.FC = () => {
  const [audit, setAudit] = useState<DatasetAuditResponse | null>(null);
  const [rowsData, setRowsData] = useState<PaginatedRowsResponse | null>(null);
  const [pageOffset, setPageOffset] = useState<number>(0);
  const [pageSize] = useState<number>(25);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isRowsLoading, setIsRowsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const loadAudit = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const [auditRes, rowsRes] = await Promise.all([
        apiClient.getDatasetAudit(),
        apiClient.getDatasetRows(pageSize, 0),
      ]);
      setAudit(auditRes);
      setRowsData(rowsRes);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to retrieve dataset audit from backend');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchPage = async (newOffset: number) => {
    setIsRowsLoading(true);
    try {
      const res = await apiClient.getDatasetRows(pageSize, newOffset);
      setRowsData(res);
      setPageOffset(newOffset);
    } catch (err) {
      console.error('Error fetching page:', err);
    } finally {
      setIsRowsLoading(false);
    }
  };

  useEffect(() => {
    loadAudit();
  }, []);

  if (isLoading) {
    return <LoadingSpinner label="Auditing dataset dimensions, missingness profiles & correlations..." />;
  }

  if (error || !audit) {
    return (
      <ErrorState
        title="Dataset Audit Disconnected"
        message={error || 'Unable to retrieve data quality profile from backend.'}
        onRetry={loadAudit}
      />
    );
  }

  const { provenance, counts, missingness_profile, target_distribution, feature_correlations, quarterly_baselines, warnings } = audit;

  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* 1. Page Header */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-xs flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-[11px] font-bold text-teal-700 tracking-wider uppercase font-mono bg-teal-50 px-2.5 py-0.5 rounded-full border border-teal-200">
              Data Audit &amp; Ingestion
            </span>
            <span className="text-slate-300">·</span>
            <span className="text-xs text-slate-500 font-medium">Schema &amp; Missingness Verification</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 font-sans">Data Quality &amp; Ingestion Profile</h1>
          <p className="text-sm text-slate-600 mt-1 max-w-3xl leading-relaxed">
            Examine European semicolon/comma dialect parsing, numeric sentinel (-200) replacement with IEEE NaN,
            target benzene availability, and quarterly baseline sensor drift.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3 shrink-0">
          <div className="flex items-center gap-2 px-3 py-2 bg-slate-50 border border-slate-200 rounded-xl text-xs text-slate-700">
            <Database className="w-3.5 h-3.5 text-teal-600" />
            <span className="font-semibold">{counts.raw_rows.toLocaleString()} Parsed Slots</span>
          </div>
          <div className="flex items-center gap-2 px-3 py-2 bg-emerald-50 border border-emerald-200 rounded-xl text-xs text-emerald-800">
            <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
            <span className="font-semibold">Zero Missing in Predictors X</span>
          </div>
        </div>
      </div>

      {/* 2. Structural KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          label="Total Raw Slots"
          value={counts.raw_rows.toLocaleString()}
          unit="hours"
          subtitle={`${provenance.temporal_range.start.split('T')[0]} to ${provenance.temporal_range.end.split('T')[0]}`}
          icon={<Calendar className="w-4 h-4 text-slate-500" />}
        />
        <MetricCard
          label="Supervised Usable"
          value={counts.supervised_usable_rows.toLocaleString()}
          unit="hours"
          badge={{ text: '96.09% available', variant: 'success' }}
          subtitle="Complete predictor-target pairs"
          icon={<Database className="w-4 h-4 text-slate-500" />}
        />
        <MetricCard
          label="Coincident Blackouts"
          value={counts.coincident_blackout_rows}
          unit="dropouts"
          badge={{ text: '3.91% missing', variant: 'warning' }}
          subtitle="Simultaneous blackout on all sensors"
          icon={<AlertTriangle className="w-4 h-4 text-amber-500" />}
        />
        <MetricCard
          label="Benzene Mean Target"
          value={target_distribution.mean.toFixed(2)}
          unit={<MathInline math="\mu\mathrm{g/m}^3" />}
          subtitle={`Std: ${target_distribution.std.toFixed(2)} · Max: ${target_distribution.max.toFixed(1)}`}
          icon={<Layers className="w-4 h-4 text-slate-500" />}
        />
      </div>

      {/* 3. Missingness & Target Availability Profile */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2">
          <Card
            title="Column Missingness & Ingestion Roles"
            subtitle={<>Pruning rows with missing <MathInline math="\mathrm{C}_6\mathrm{H}_6\text{(GT)}" /> removes exactly 366 coincident blackout hours and leaves ZERO missing values in the 8 predictive features</>}
          >
            <div className="overflow-x-auto border border-slate-200/80 rounded-xl">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold uppercase tracking-wider">
                  <tr>
                    <th className="py-2.5 px-3.5">Column Name</th>
                    <th className="py-2.5 px-3.5">Category</th>
                    <th className="py-2.5 px-3.5">Dtype</th>
                    <th className="py-2.5 px-3.5 font-mono">Missing #</th>
                    <th className="py-2.5 px-3.5 font-mono">Missing %</th>
                    <th className="py-2.5 px-3.5 font-mono">Valid #</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 font-mono">
                  {missingness_profile.map((col) => {
                    const isTarget = col.name === 'C6H6(GT)';
                    const isExcluded = col.category.includes('Excluded');
                    const isSevere = col.missing_pct > 80;

                    return (
                      <tr key={col.name} className="hover:bg-slate-50/80 transition-colors">
                        <td className="py-2.5 px-3.5 font-medium text-slate-900 flex items-center gap-1.5 font-sans">
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
                        <td className="py-2.5 px-3.5 text-slate-700">{col.missing_count}</td>
                        <td className="py-2.5 px-3.5">
                          <span className={isSevere ? 'text-rose-600 font-bold' : col.missing_pct > 0 ? 'text-amber-600' : 'text-slate-600'}>
                            {col.missing_pct.toFixed(2)}%
                          </span>
                        </td>
                        <td className="py-2.5 px-3.5 text-slate-600">{col.valid_count}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </Card>
        </div>

        {/* Target Benzene Distribution Summary Card */}
        <div>
          <Card
            title={<>Target Benzene <MathInline math="\mathrm{C}_6\mathrm{H}_6\text{(GT)}" /> Distribution</>}
            subtitle={<>Reference analyzer parametric summary (<MathInline math="\mu\mathrm{g/m}^3" />)</>}
          >
            <div className="space-y-3 font-mono text-xs">
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-sans">Valid Observations</span>
                <span className="font-semibold text-slate-900">{target_distribution.count.toLocaleString()}</span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-sans">Mean Concentration</span>
                <span className="font-semibold text-teal-700">{target_distribution.mean.toFixed(3)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-sans">Standard Deviation</span>
                <span className="font-semibold text-slate-700">{target_distribution.std.toFixed(3)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-sans">Minimum</span>
                <span className="text-slate-700">{target_distribution.min.toFixed(2)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-sans">25th Percentile (Q1)</span>
                <span className="text-slate-700">{target_distribution.p25.toFixed(2)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-sans">Median (Q2)</span>
                <span className="text-slate-700">{target_distribution.p50.toFixed(2)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
              </div>
              <div className="flex justify-between py-1.5 border-b border-slate-100">
                <span className="text-slate-500 font-sans">75th Percentile (Q3)</span>
                <span className="text-slate-700">{target_distribution.p75.toFixed(2)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
              </div>
              <div className="flex justify-between py-1.5">
                <span className="text-slate-500 font-sans">Maximum</span>
                <span className="font-bold text-rose-600">{target_distribution.max.toFixed(2)} <MathInline math="\mu\mathrm{g/m}^3" /></span>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-slate-100">
              <Alert variant="info" title="Target Availability">
                The 366 missing values occur in exactly the same rows where all sensor channels are missing, confirming a station blackout rather than sensor failure.
              </Alert>
            </div>
          </Card>
        </div>
      </div>

      {/* 4. Predictor Correlations & Quarterly Baseline Exploration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Correlations Table & Bar */}
        <Card
          title={<>Predictor Correlations with Target <MathInline math="\mathrm{C}_6\mathrm{H}_6\text{(GT)}" /></>}
          subtitle={<>Pearson correlation coefficients <MathInline math="r" /> indicating strong collinearity between MOX sensors</>}
        >
          <div className="h-68 w-full mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={feature_correlations}
                layout="vertical"
                margin={{ top: 5, right: 30, left: 90, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
                <XAxis type="number" domain={[-1, 1]} stroke="#94A3B8" fontSize={11} />
                <YAxis dataKey="feature" type="category" stroke="#94A3B8" fontSize={10} width={90} />
                <Tooltip
                  content={({ payload }) => {
                    if (!payload || payload.length === 0) return null;
                    const d = payload[0].payload;
                    return (
                      <div className="bg-slate-900 text-white p-3 rounded-xl text-xs shadow-xl space-y-1 font-mono">
                        <div className="font-bold text-teal-300">{d.feature}</div>
                        <div>Pearson <MathInline math="r" />: {d.pearson_r.toFixed(4)}</div>
                        <div className="text-[10px] text-slate-400">Absolute <MathInline math="|r|" />: {d.abs_r.toFixed(4)}</div>
                      </div>
                    );
                  }}
                />
                <Bar dataKey="pearson_r" name="Pearson r" radius={[0, 4, 4, 0]}>
                  {feature_correlations.map((entry) => (
                    <Cell
                      key={`corr-${entry.feature}`}
                      fill={entry.pearson_r >= 0.8 ? '#0D9488' : entry.pearson_r >= 0 ? '#38BDF8' : '#F43F5E'}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="pt-3 border-t border-slate-100 text-xs text-slate-500 flex justify-between">
            <span>PT08.S2(NMHC) has the strongest correlation with benzene: <MathInline math="r = +0.9820" />.</span>
            <span className="font-mono text-teal-700 font-semibold"><MathInline math="|r|_{\max} = 0.9820" /></span>
          </div>
        </Card>

        {/* Quarterly Baseline Drift Table */}
        <Card
          title="Quarterly Sensor Baseline Trends (Drift Exploration)"
          subtitle="Tracking mean sensor response across 5 quarters (March 2004 – April 2005)"
        >
          <div className="overflow-x-auto border border-slate-200/80 rounded-xl">
            <table className="w-full text-left text-xs font-mono">
              <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold uppercase">
                <tr>
                  <th className="py-2.5 px-3 font-sans">Quarter</th>
                  <th className="py-2.5 px-3">PT08.S1</th>
                  <th className="py-2.5 px-3">PT08.S2</th>
                  <th className="py-2.5 px-3">PT08.S3</th>
                  <th className="py-2.5 px-3">Temp (&deg;C)</th>
                  <th className="py-2.5 px-3">AH</th>
                  <th className="py-2.5 px-3"><MathInline math="\mathrm{C}_6\mathrm{H}_6" /></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {quarterly_baselines.map((q) => (
                  <tr key={q.quarter} className="hover:bg-slate-50/80 transition-colors">
                    <td className="py-2.5 px-3 font-sans font-medium text-slate-900">{q.quarter}</td>
                    <td className="py-2.5 px-3 text-slate-700">{q.pt08_s1.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-slate-700">{q.pt08_s2.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-teal-800">{q.pt08_s3.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-slate-600">{q.t.toFixed(1)}</td>
                    <td className="py-2.5 px-3 text-slate-600">{q.ah.toFixed(2)}</td>
                    <td className="py-2.5 px-3 font-bold text-slate-900">{q.c6h6_gt.toFixed(1)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-3 pt-3 border-t border-slate-100 text-xs text-slate-500">
            Notice PT08.S3(NOx) drops monotonically from 1029.2 in 2004Q1 to 761.8 in 2005Q1.
          </div>
        </Card>
      </div>

      {/* 5. Interactive Paginated Data Explorer Table */}
      <Card
        title="Interactive Observation Explorer"
        subtitle={`Viewing cleaned rows ${pageOffset + 1} to ${Math.min(pageOffset + pageSize, rowsData?.total || 0)} of ${rowsData?.total.toLocaleString() || 0}`}
        action={
          <div className="flex items-center gap-2">
            <button
              onClick={() => fetchPage(Math.max(0, pageOffset - pageSize))}
              disabled={pageOffset === 0 || isRowsLoading}
              className="p-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-40 transition-colors cursor-pointer"
              title="Previous page"
            >
              <ChevronLeft className="w-4 h-4" />
            </button>
            <span className="text-xs font-mono text-slate-500 px-1">
              Page {Math.floor(pageOffset / pageSize) + 1} of {Math.ceil((rowsData?.total || 1) / pageSize)}
            </span>
            <button
              onClick={() => fetchPage(pageOffset + pageSize)}
              disabled={pageOffset + pageSize >= (rowsData?.total || 0) || isRowsLoading}
              className="p-1.5 rounded-lg border border-slate-200 text-slate-600 hover:bg-slate-50 disabled:opacity-40 transition-colors cursor-pointer"
              title="Next page"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        }
      >
        <div className="overflow-x-auto border border-slate-200/80 rounded-xl relative">
          {isRowsLoading && (
            <div className="absolute inset-0 bg-white/70 backdrop-blur-2xs flex items-center justify-center z-10">
              <div className="flex items-center gap-2 text-xs font-mono text-teal-800 bg-white px-3 py-1.5 rounded-xl border border-teal-200 shadow-sm">
                <div className="w-3.5 h-3.5 border-2 border-teal-500 border-t-transparent rounded-full animate-spin" />
                <span>Loading records...</span>
              </div>
            </div>
          )}

          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold uppercase text-[11px]">
              <tr>
                <th className="py-2.5 px-3 font-sans">Timestamp</th>
                <th className="py-2.5 px-3 text-teal-800"><MathInline math="\mathrm{C}_6\mathrm{H}_6\text{(GT)}" /></th>
                <th className="py-2.5 px-3">PT08.S1</th>
                <th className="py-2.5 px-3">PT08.S2</th>
                <th className="py-2.5 px-3">PT08.S3</th>
                <th className="py-2.5 px-3">PT08.S4</th>
                <th className="py-2.5 px-3">PT08.S5</th>
                <th className="py-2.5 px-3">T (&deg;C)</th>
                <th className="py-2.5 px-3">RH (%)</th>
                <th className="py-2.5 px-3">AH</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {rowsData?.rows.map((row, idx) => (
                <tr key={idx} className="hover:bg-teal-50/20 transition-colors">
                  <td className="py-2 px-3 font-sans text-slate-800 whitespace-nowrap">
                    {String(row['Timestamp'] || '')}
                  </td>
                  <td className="py-2 px-3 font-bold text-teal-700">
                    {row['C6H6(GT)'] !== null ? Number(row['C6H6(GT)']).toFixed(2) : (
                      <span className="text-[10px] text-rose-500 bg-rose-50 px-1 py-0.5 rounded font-mono">NULL</span>
                    )}
                  </td>
                  <td className="py-2 px-3 text-slate-700">{row['PT08.S1(CO)'] ?? 'NULL'}</td>
                  <td className="py-2 px-3 text-slate-700">{row['PT08.S2(NMHC)'] ?? 'NULL'}</td>
                  <td className="py-2 px-3 text-slate-700">{row['PT08.S3(NOx)'] ?? 'NULL'}</td>
                  <td className="py-2 px-3 text-slate-700">{row['PT08.S4(NO2)'] ?? 'NULL'}</td>
                  <td className="py-2 px-3 text-slate-700">{row['PT08.S5(O3)'] ?? 'NULL'}</td>
                  <td className="py-2 px-3 text-slate-600">{row['T'] !== null ? Number(row['T']).toFixed(1) : 'NULL'}</td>
                  <td className="py-2 px-3 text-slate-600">{row['RH'] !== null ? Number(row['RH']).toFixed(1) : 'NULL'}</td>
                  <td className="py-2 px-3 text-slate-600">{row['AH'] !== null ? Number(row['AH']).toFixed(3) : 'NULL'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* 6. Ingestion Warnings Alert Box */}
      <div className="space-y-3">
        <h3 className="font-bold text-xs uppercase tracking-wider text-slate-500 font-mono">
          Data Engineering Ingestion Notes
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {warnings.map((warn, i) => (
            <Alert key={i} variant={i === 0 ? 'info' : 'warning'}>
              {warn}
            </Alert>
          ))}
        </div>
      </div>
    </div>
  );
};
