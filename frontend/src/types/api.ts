/**
 * TypeScript interface definitions matching the AirSense API schemas.
 * Conforms to: docs/frontend_integration_contract.md
 */

export interface HealthResponse {
  status: string;
  version: string;
  dataset_loaded: boolean;
  supervised_rows: number;
  cleaned_rows: number;
}

export interface ColumnMetadata {
  name: string;
  category: string;
  unit: string;
  dtype: string;
  missing_count: number;
  missing_pct: number;
  valid_count: number;
}

export interface TemporalRange {
  start: string;
  end: string;
  total_hours: number;
}

export interface DatasetProvenance {
  source: string;
  citation: string;
  sampling_location: string;
  temporal_range: TemporalRange;
}

export interface DatasetCounts {
  raw_rows: number;
  coincident_blackout_rows: number;
  supervised_usable_rows: number;
  sensor_features_count: number;
}

export interface DatasetSummaryResponse {
  provenance: DatasetProvenance;
  counts: DatasetCounts;
  columns: ColumnMetadata[];
}

export interface TargetDistribution {
  count: number;
  mean: number;
  std: number;
  min: number;
  p25: number;
  p50: number;
  p75: number;
  max: number;
}

export interface FeatureCorrelation {
  feature: string;
  pearson_r: number;
  abs_r: number;
}

export interface QuarterlyBaseline {
  quarter: string;
  pt08_s1: number;
  pt08_s2: number;
  pt08_s3: number;
  pt08_s4: number;
  pt08_s5: number;
  t: number;
  rh: number;
  ah: number;
  c6h6_gt: number;
}

export interface DatasetAuditResponse {
  provenance: DatasetProvenance;
  counts: DatasetCounts;
  missingness_profile: ColumnMetadata[];
  target_availability: {
    target_column: string;
    unit: string;
    total_slots: number;
    missing_slots: number;
    missing_pct: number;
    usable_supervised_rows: number;
    usable_supervised_pct: number;
  };
  target_distribution: TargetDistribution;
  feature_correlations: FeatureCorrelation[];
  quarterly_baselines: QuarterlyBaseline[];
  warnings: string[];
}

export interface PaginatedRowsResponse {
  total: number;
  offset: number;
  limit: number;
  rows: Record<string, string | number | null>[];
}

export interface MatrixFeatureParam {
  name: string;
  symbol: string;
  unit: string;
  description: string;
  train_mean: number;
  train_std: number;
}

export interface MatrixPartitionInfo {
  name: string;
  label: string;
  row_count: number;
  pct: number;
  start_date: string;
  end_date: string;
}

export interface MatrixRowSample {
  row_index: number;
  timestamp: string;
  partition: string;
  raw_x: number[];
  standardized_x: number[];
  raw_y: number;
}

export interface MatrixPreviewResponse {
  total_usable_rows: number;
  total_raw_rows: number;
  total_blackout_rows: number;
  feature_count: number;
  target_count: number;
  matrix_notation: string;
  target_notation: string;
  features: MatrixFeatureParam[];
  target_feature: string;
  target_unit: string;
  target_description: string;
  partitions: MatrixPartitionInfo[];
  offset: number;
  limit: number;
  total_rows: number;
  rows: MatrixRowSample[];
}

export interface OverviewMetricsResponse {
  observations_available: number;
  usable_target_observations: number;
  sensor_features_count: number;
  validation_selected_model: string;
  validation_selected_val_rmse: number;
  validation_selected_val_r2: number;
  validation_selected_test_rmse: number;
  validation_selected_test_r2: number;
  validation_selected_test_mae: number;
  validation_selected_test_mbe: number;
  post_hoc_best_test_model: string;
  post_hoc_test_rmse: number;
  post_hoc_test_r2: number;
  post_hoc_test_mae: number;
  post_hoc_test_mbe: number;
  durbin_watson: number;
  model_selection_note: string;
  drift_caveat: string;
}

export interface ModelBenchmarkRow {
  k: number;
  cum_variance_pct: number;
  train_rmse: number;
  train_r2: number;
  val_rmse: number;
  val_r2: number;
  test_rmse: number;
  test_r2: number;
  role: string;
}

export interface PCAVariancePoint {
  k: number;
  variance_pct: number;
  cum_variance_pct: number;
}

export interface ScatterDataPoint {
  timestamp: string;
  actual: number;
  predicted_ols: number;
  predicted_pcr3: number;
  residual_ols: number;
  residual_pcr3: number;
}

export interface TimelinePoint {
  timestamp: string;
  actual: number;
  predicted_ols: number;
  predicted_pcr3: number;
}

export interface ResidualsHistogramBin {
  bin_start: number;
  bin_end: number;
  bin_center: number;
  count: number;
  density: number;
  normal_density: number;
}

export interface OverviewChartsResponse {
  model_comparison: ModelBenchmarkRow[];
  pca_variance: PCAVariancePoint[];
  scatter_sample: ScatterDataPoint[];
  timeline_sample: TimelinePoint[];
  residuals_distribution: ResidualsHistogramBin[];
}

export interface BenchmarkMatrixResponse {
  models: ModelBenchmarkRow[];
}

export interface CalibrationRunRequest {
  model_type: 'OLS' | 'PCR';
  selected_features: string[];
  k_components?: number;
}

export interface PartitionMetrics {
  rmse: number;
  mae: number;
  r2: number;
  mbe: number;
}

export interface FeatureSensitivity {
  feature: string;
  weight: number;
  unit: string;
}

export interface CalibrationScatterPoint {
  timestamp: string;
  actual: number;
  predicted: number;
  residual: number;
}

export interface CalibrationTimelinePoint {
  timestamp: string;
  actual: number;
  predicted: number;
}

export interface PCAComponentVariance {
  k: number;
  variance_pct: number;
  cum_variance_pct: number;
}

export interface CalibrationRunResponse {
  experiment_id: string;
  model_type: string;
  selected_features: string[];
  k_components?: number;
  feature_count: number;
  train_metrics: PartitionMetrics;
  val_metrics: PartitionMetrics;
  feature_sensitivities: FeatureSensitivity[];
  intercept: number;
  pca_variance?: PCAComponentVariance[];
  scatter_sample: CalibrationScatterPoint[];
  timeline_sample: CalibrationTimelinePoint[];
  residuals_distribution: ResidualsHistogramBin[];
  split_info: Record<string, string | number>;
  warnings: string[];
  scientific_notice: Record<string, string>;
}

// ==============================================================================
// Linear Algebra Lab Interfaces
// ==============================================================================

export interface MatrixExtractRequest {
  features?: string[];
  n_rows?: number;
  standardize?: boolean;
  include_intercept?: boolean;
}

export interface MatrixExtractResponse {
  matrix: number[][];
  row_labels: string[];
  col_labels: string[];
  target_vector: number[];
  dimensions: number[];
  means: number[];
  stds: number[];
  is_standardized: boolean;
  source_description: string;
}

export interface GaussianStep {
  step_index: number;
  operation: string;
  description: string;
  matrix: number[][];
  pivot_row?: number | null;
  pivot_col?: number | null;
  target_row?: number | null;
}

export interface LUFactorization {
  P: number[][];
  L: number[][];
  U: number[][];
  residual_error: number;
  is_valid: boolean;
}

export interface GaussianEliminationRequest {
  matrix: number[][];
  compute_lu?: boolean;
}

export interface GaussianEliminationResponse {
  steps: GaussianStep[];
  final_rref: number[][];
  pivot_columns: number[];
  rank: number;
  lu?: LUFactorization | null;
  determinant?: number | null;
}

export interface SubspaceAnalysisRequest {
  matrix: number[][];
  tolerance?: number;
}

export interface SubspaceAnalysisResponse {
  m_rows: number;
  n_cols: number;
  rank: number;
  nullity: number;
  rank_nullity_satisfied: boolean;
  column_space_basis: number[][];
  null_space_basis: number[][];
  pivot_columns: number[];
  free_columns: number[];
  max_null_residual: number;
}

export interface GramSchmidtRequest {
  matrix: number[][];
  tolerance?: number;
}

export interface GramSchmidtResponse {
  q_mgs: number[][];
  r_mgs: number[][];
  q_cgs: number[][];
  r_cgs: number[][];
  loss_mgs: number;
  loss_cgs: number;
  stability_gain: number;
  dot_product_mgs: number[][];
  dot_product_cgs: number[][];
  rank: number;
}

export interface LeastSquaresRequest {
  matrix: number[][];
  target_vector: number[];
  tolerance?: number;
}

export interface LeastSquaresResponse {
  weights: number[];
  projected_target: number[];
  residual_vector: number[];
  rss: number;
  orthogonality_residual: number;
  cond_X: number;
  cond_XtX: number;
  cond_ratio: number;
  projection_matrix_sample: number[][];
}

export interface SpectralAnalysisRequest {
  features?: string[];
  use_training_set?: boolean;
  custom_matrix?: number[][] | null;
}

export interface SpectralAnalysisResponse {
  matrix: number[][];
  feature_names: string[];
  is_psd: boolean;
  is_symmetric: boolean;
  symmetry_residual: number;
  eigenvalues: number[];
  eigenvectors: number[][];
  diagonal_matrix: number[][];
  max_eigenvector_residual: number;
  variance_explained_ratio: number[];
  cumulative_variance_ratio: number[];
  condition_number: number;
}

// ==============================================================================
// PCA Explorer Interfaces
// ==============================================================================

export interface PCAScorePoint {
  timestamp: string;
  score_x: number;
  score_y: number;
  target: number;
  hour_idx: number;
}

export interface PCAFeatureContribution {
  feature: string;
  loading: number;
  abs_loading: number;
  contribution_pct: number;
}

export interface PCAFeatureReconstruction {
  feature: string;
  rmse: number;
  mae: number;
  relative_error_pct: number;
}

export interface PCAReconstructionTimeSeriesPoint {
  timestamp: string;
  original: number;
  reconstructed: number;
  residual: number;
}

export interface PCAVarianceTableItem {
  component: string;
  eigenvalue: number;
  singular_value: number;
  variance_ratio_pct: number;
  cumulative_variance_pct: number;
  is_retained: boolean;
}

export interface PCARequest {
  features?: string[];
  standardize?: boolean;
  n_components?: number;
  partition?: 'train' | 'val' | 'all';
  score_pc_x?: number;
  score_pc_y?: number;
  sample_size?: number;
  reconstruction_feature?: string | null;
}

export interface PCAResponse {
  selected_features: string[];
  standardize: boolean;
  partition: string;
  total_training_rows: number;
  evaluated_partition_rows: number;
  n_components: number;
  total_components: number;
  variance_table: PCAVarianceTableItem[];
  retained_variance_pct: number;
  discarded_variance_pct: number;
  loadings_matrix: number[][];
  loadings_columns: string[];
  loadings_features: string[];
  feature_contributions_pc_x: PCAFeatureContribution[];
  feature_contributions_pc_y: PCAFeatureContribution[];
  scores_sample: PCAScorePoint[];
  score_pc_x_label: string;
  score_pc_y_label: string;
  score_pc_x_var_pct: number;
  score_pc_y_var_pct: number;
  reconstruction_error_frobenius: number;
  theoretical_error_eckart_young: number;
  relative_reconstruction_error: number;
  feature_reconstruction: PCAFeatureReconstruction[];
  time_series_reconstruction: PCAReconstructionTimeSeriesPoint[];
  reconstruction_feature_name: string;
  feature_means: number[];
  feature_stds: number[];
  scientific_notes: Record<string, string>;
}

// ==============================================================================
// Drift Analysis & Sensor Monitoring Interfaces
// ==============================================================================

export interface DriftTimelinePoint {
  timestamp: string;
  actual: number;
  predicted: number;
  residual: number;
  rolling_bias: number;
  rolling_rmse: number;
  cusum: number;
  upper_bound_3sigma: number;
  lower_bound_3sigma: number;
  partition: string;
}

export interface FeatureDriftItem {
  feature: string;
  ref_mean: number;
  mon_mean: number;
  shift: number;
  cohens_d: number;
  ks_statistic: number;
  ks_pvalue: number;
  drift_status: 'Stable' | 'Moderate Drift' | 'Severe Drift';
}

export interface FeatureStats {
  mean: number;
  std: number;
  min: number;
  p25: number;
  median: number;
  p75: number;
  max: number;
  count: number;
}

export interface DriftHistogramBin {
  bin_center: number;
  ref_density: number;
  mon_density: number;
}

export interface FeatureDistributionDetail {
  feature_name: string;
  ref_stats: FeatureStats;
  mon_stats: FeatureStats;
  cohens_d: number;
  ks_statistic: number;
  ks_pvalue: number;
  welch_t_statistic: number;
  welch_pvalue: number;
  distribution_bins: DriftHistogramBin[];
}

export interface MonthlyErrorItem {
  period: string;
  count: number;
  rmse: number;
  mae: number;
  r2: number;
  mean_bias: number;
  durbin_watson: number;
}

export interface DriftFinding {
  category: string;
  severity: 'info' | 'warning' | 'breach';
  title: string;
  description: string;
  supporting_statistic: string;
  caveat: string;
}

export interface DriftMetricsSummary {
  ref_rmse: number;
  ref_mae: number;
  ref_r2: number;
  ref_mbe: number;
  mon_rmse: number;
  mon_mae: number;
  mon_r2: number;
  mon_mbe: number;
  rmse_inflation_pct: number;
  bias_shift: number;
}

export interface DriftAnalysisRequest {
  model_type?: 'OLS' | 'PCR';
  k_components?: number;
  reference_partition?: 'train' | 'val' | 'test';
  monitoring_partition?: 'val' | 'test' | 'monitoring';
  selected_feature?: string;
  rolling_window_hours?: number;
  timeline_subsample?: number;
}

export interface DriftAnalysisResponse {
  model_type: string;
  k_components?: number | null;
  reference_label: string;
  monitoring_label: string;
  reference_rows: number;
  monitoring_rows: number;
  total_rows: number;
  overall_metrics: DriftMetricsSummary;
  durbin_watson_reference: number;
  durbin_watson_monitoring: number;
  durbin_watson_full: number;
  timeline_series: DriftTimelinePoint[];
  feature_drift_summary: FeatureDriftItem[];
  selected_feature_detail: FeatureDistributionDetail;
  monthly_progression: MonthlyErrorItem[];
  explainable_findings: DriftFinding[];
  scientific_notes: Record<string, string>;
}




