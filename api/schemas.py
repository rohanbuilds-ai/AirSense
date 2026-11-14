"""Pydantic schemas for the AirSense REST API.

Defines typed request and response contracts conforming to:
docs/frontend_integration_contract.md
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ==============================================================================
# Health Schemas
# ==============================================================================

class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="0.1.0")
    dataset_loaded: bool = Field(..., example=True)
    supervised_rows: int = Field(..., example=8991)
    cleaned_rows: int = Field(..., example=9357)


# ==============================================================================
# Dataset Ingestion & Audit Schemas
# ==============================================================================

class ColumnMetadata(BaseModel):
    name: str
    category: str
    unit: str
    dtype: str
    missing_count: int
    missing_pct: float
    valid_count: int


class TemporalRange(BaseModel):
    start: str
    end: str
    total_hours: int


class DatasetProvenance(BaseModel):
    source: str
    citation: str
    sampling_location: str
    temporal_range: TemporalRange


class DatasetCounts(BaseModel):
    raw_rows: int
    coincident_blackout_rows: int
    supervised_usable_rows: int
    sensor_features_count: int


class DatasetSummaryResponse(BaseModel):
    provenance: DatasetProvenance
    counts: DatasetCounts
    columns: List[ColumnMetadata]


class TargetDistribution(BaseModel):
    count: int
    mean: float
    std: float
    min: float
    p25: float
    p50: float
    p75: float
    max: float


class FeatureCorrelation(BaseModel):
    feature: str
    pearson_r: float
    abs_r: float


class QuarterlyBaseline(BaseModel):
    quarter: str
    pt08_s1: float
    pt08_s2: float
    pt08_s3: float
    pt08_s4: float
    pt08_s5: float
    t: float
    rh: float
    ah: float
    c6h6_gt: float


class DatasetAuditResponse(BaseModel):
    provenance: DatasetProvenance
    counts: DatasetCounts
    missingness_profile: List[ColumnMetadata]
    target_availability: Dict[str, Any]
    target_distribution: TargetDistribution
    feature_correlations: List[FeatureCorrelation]
    quarterly_baselines: List[QuarterlyBaseline]
    warnings: List[str]


class PaginatedRowsResponse(BaseModel):
    total: int
    offset: int
    limit: int
    rows: List[Dict[str, Any]]


class MatrixFeatureParam(BaseModel):
    name: str
    symbol: str
    unit: str
    description: str
    train_mean: float
    train_std: float


class MatrixPartitionInfo(BaseModel):
    name: str
    label: str
    row_count: int
    pct: float
    start_date: str
    end_date: str


class MatrixRowSample(BaseModel):
    row_index: int
    timestamp: str
    partition: str
    raw_x: List[float]
    standardized_x: List[float]
    raw_y: float


class MatrixPreviewResponse(BaseModel):
    total_usable_rows: int
    total_raw_rows: int
    total_blackout_rows: int
    feature_count: int
    target_count: int
    matrix_notation: str
    target_notation: str
    features: List[MatrixFeatureParam]
    target_feature: str
    target_unit: str
    target_description: str
    partitions: List[MatrixPartitionInfo]
    offset: int
    limit: int
    total_rows: int
    rows: List[MatrixRowSample]


# ==============================================================================
# Overview Page Schemas
# ==============================================================================

class OverviewMetricsResponse(BaseModel):
    observations_available: int
    usable_target_observations: int
    sensor_features_count: int
    validation_selected_model: str
    validation_selected_val_rmse: float
    validation_selected_val_r2: float
    validation_selected_test_rmse: float
    validation_selected_test_r2: float
    validation_selected_test_mae: float
    validation_selected_test_mbe: float
    post_hoc_best_test_model: str
    post_hoc_test_rmse: float
    post_hoc_test_r2: float
    post_hoc_test_mae: float
    post_hoc_test_mbe: float
    durbin_watson: float
    model_selection_note: str
    drift_caveat: str


class ModelBenchmarkRow(BaseModel):
    k: int
    cum_variance_pct: float
    train_rmse: float
    train_r2: float
    val_rmse: float
    val_r2: float
    test_rmse: float
    test_r2: float
    role: str


class PCAVariancePoint(BaseModel):
    k: int
    variance_pct: float
    cum_variance_pct: float


class ScatterDataPoint(BaseModel):
    timestamp: str
    actual: float
    predicted_ols: float
    predicted_pcr3: float
    residual_ols: float
    residual_pcr3: float


class TimelinePoint(BaseModel):
    timestamp: str
    actual: float
    predicted_ols: float
    predicted_pcr3: float


class ResidualsHistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    bin_center: float
    count: int
    density: float
    normal_density: float


class OverviewChartsResponse(BaseModel):
    model_comparison: List[ModelBenchmarkRow]
    pca_variance: List[PCAVariancePoint]
    scatter_sample: List[ScatterDataPoint]
    timeline_sample: List[TimelinePoint]
    residuals_distribution: List[ResidualsHistogramBin]


# ==============================================================================
# Calibration Schemas
# ==============================================================================

class BenchmarkMatrixResponse(BaseModel):
    models: List[ModelBenchmarkRow]


class CalibrationRunRequest(BaseModel):
    model_type: str = Field(..., description="Model architecture: 'OLS' or 'PCR'")
    selected_features: List[str] = Field(..., description="Predictor feature subset")
    k_components: Optional[int] = Field(None, description="Number of PCA components if model_type is PCR (1 <= k <= len(selected_features))")


class PartitionMetrics(BaseModel):
    rmse: float
    mae: float
    r2: float
    mbe: float


class FeatureSensitivity(BaseModel):
    feature: str
    weight: float
    unit: str


class CalibrationScatterPoint(BaseModel):
    timestamp: str
    actual: float
    predicted: float
    residual: float


class CalibrationTimelinePoint(BaseModel):
    timestamp: str
    actual: float
    predicted: float


class PCAComponentVariance(BaseModel):
    k: int
    variance_pct: float
    cum_variance_pct: float


class CalibrationRunResponse(BaseModel):
    experiment_id: str
    model_type: str
    selected_features: List[str]
    k_components: Optional[int]
    feature_count: int
    train_metrics: PartitionMetrics
    val_metrics: PartitionMetrics
    feature_sensitivities: List[FeatureSensitivity]
    intercept: float
    pca_variance: Optional[List[PCAComponentVariance]] = None
    scatter_sample: List[CalibrationScatterPoint]
    timeline_sample: List[CalibrationTimelinePoint]
    residuals_distribution: List[ResidualsHistogramBin]
    split_info: Dict[str, Any]
    warnings: List[str]
    scientific_notice: Dict[str, str]


# ==============================================================================
# Linear Algebra Lab Schemas
# ==============================================================================

class MatrixExtractRequest(BaseModel):
    features: List[str] = Field(default=["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)"])
    n_rows: int = Field(default=8, ge=3, le=30)
    standardize: bool = Field(default=False)
    include_intercept: bool = Field(default=False)


class MatrixExtractResponse(BaseModel):
    matrix: List[List[float]]
    row_labels: List[str]
    col_labels: List[str]
    target_vector: List[float]
    dimensions: List[int]
    means: List[float]
    stds: List[float]
    is_standardized: bool
    source_description: str


class GaussianStep(BaseModel):
    step_index: int
    operation: str
    description: str
    matrix: List[List[float]]
    pivot_row: Optional[int] = None
    pivot_col: Optional[int] = None
    target_row: Optional[int] = None


class LUFactorization(BaseModel):
    P: List[List[float]]
    L: List[List[float]]
    U: List[List[float]]
    residual_error: float
    is_valid: bool


class GaussianEliminationRequest(BaseModel):
    matrix: List[List[float]]
    compute_lu: bool = Field(default=True)


class GaussianEliminationResponse(BaseModel):
    steps: List[GaussianStep]
    final_rref: List[List[float]]
    pivot_columns: List[int]
    rank: int
    lu: Optional[LUFactorization] = None
    determinant: Optional[float] = None


class SubspaceAnalysisRequest(BaseModel):
    matrix: List[List[float]]
    tolerance: float = Field(default=1e-10)


class SubspaceAnalysisResponse(BaseModel):
    m_rows: int
    n_cols: int
    rank: int
    nullity: int
    rank_nullity_satisfied: bool
    column_space_basis: List[List[float]]
    null_space_basis: List[List[float]]
    pivot_columns: List[int]
    free_columns: List[int]
    max_null_residual: float


class GramSchmidtRequest(BaseModel):
    matrix: List[List[float]]
    tolerance: float = Field(default=1e-10)


class GramSchmidtResponse(BaseModel):
    q_mgs: List[List[float]]
    r_mgs: List[List[float]]
    q_cgs: List[List[float]]
    r_cgs: List[List[float]]
    loss_mgs: float
    loss_cgs: float
    stability_gain: float
    dot_product_mgs: List[List[float]]
    dot_product_cgs: List[List[float]]
    rank: int


class LeastSquaresRequest(BaseModel):
    matrix: List[List[float]]
    target_vector: List[float]
    tolerance: float = Field(default=1e-10)


class LeastSquaresResponse(BaseModel):
    weights: List[float]
    projected_target: List[float]
    residual_vector: List[float]
    rss: float
    orthogonality_residual: float
    cond_X: float
    cond_XtX: float
    cond_ratio: float
    projection_matrix_sample: List[List[float]]


class SpectralAnalysisRequest(BaseModel):
    features: List[str] = Field(default=["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "T"])
    use_training_set: bool = Field(default=True)
    custom_matrix: Optional[List[List[float]]] = None


class SpectralAnalysisResponse(BaseModel):
    matrix: List[List[float]]
    feature_names: List[str]
    is_psd: bool
    is_symmetric: bool
    symmetry_residual: float
    eigenvalues: List[float]
    eigenvectors: List[List[float]]
    diagonal_matrix: List[List[float]]
    max_eigenvector_residual: float
    variance_explained_ratio: List[float]
    cumulative_variance_ratio: List[float]
    condition_number: float


# ==============================================================================
# PCA Explorer Schemas
# ==============================================================================

class PCAScorePoint(BaseModel):
    timestamp: str
    score_x: float
    score_y: float
    target: float  # Reference analyzer C6H6(GT), purely visual annotation
    hour_idx: int


class PCAFeatureContribution(BaseModel):
    feature: str
    loading: float
    abs_loading: float
    contribution_pct: float


class PCAFeatureReconstruction(BaseModel):
    feature: str
    rmse: float
    mae: float
    relative_error_pct: float


class PCAReconstructionTimeSeriesPoint(BaseModel):
    timestamp: str
    original: float
    reconstructed: float
    residual: float


class PCAVarianceTableItem(BaseModel):
    component: str  # "PC1", "PC2", etc.
    eigenvalue: float
    singular_value: float
    variance_ratio_pct: float
    cumulative_variance_pct: float
    is_retained: bool


class PCARequest(BaseModel):
    features: List[str] = Field(
        default=["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"]
    )
    standardize: bool = Field(default=True)
    n_components: int = Field(default=3, ge=1)
    partition: str = Field(default="train")  # "train", "val", "all"
    score_pc_x: int = Field(default=1, ge=1)  # 1-indexed (e.g. 1 for PC1)
    score_pc_y: int = Field(default=2, ge=1)  # 1-indexed (e.g. 2 for PC2)
    sample_size: int = Field(default=250, ge=50, le=1000)
    reconstruction_feature: Optional[str] = None


class PCAResponse(BaseModel):
    selected_features: List[str]
    standardize: bool
    partition: str
    total_training_rows: int
    evaluated_partition_rows: int
    n_components: int
    total_components: int
    variance_table: List[PCAVarianceTableItem]
    retained_variance_pct: float
    discarded_variance_pct: float
    loadings_matrix: List[List[float]]  # shape (d, d), columns are PC1..PCd
    loadings_columns: List[str]
    loadings_features: List[str]
    feature_contributions_pc_x: List[PCAFeatureContribution]
    feature_contributions_pc_y: List[PCAFeatureContribution]
    scores_sample: List[PCAScorePoint]
    score_pc_x_label: str
    score_pc_y_label: str
    score_pc_x_var_pct: float
    score_pc_y_var_pct: float
    reconstruction_error_frobenius: float
    theoretical_error_eckart_young: float
    relative_reconstruction_error: float
    feature_reconstruction: List[PCAFeatureReconstruction]
    time_series_reconstruction: List[PCAReconstructionTimeSeriesPoint]
    reconstruction_feature_name: str
    feature_means: List[float]
    feature_stds: List[float]
    scientific_notes: Dict[str, str]


# ==============================================================================
# Drift Analysis & Sensor Monitoring Schemas
# ==============================================================================

class DriftTimelinePoint(BaseModel):
    timestamp: str
    actual: float
    predicted: float
    residual: float
    rolling_bias: float
    rolling_rmse: float
    cusum: float
    upper_bound_3sigma: float
    lower_bound_3sigma: float
    partition: str  # "train", "val", "test"


class FeatureDriftItem(BaseModel):
    feature: str
    ref_mean: float
    mon_mean: float
    shift: float
    cohens_d: float
    ks_statistic: float
    ks_pvalue: float
    drift_status: str  # "Stable", "Moderate Drift", "Severe Drift"


class FeatureStats(BaseModel):
    mean: float
    std: float
    min: float
    p25: float
    median: float
    p75: float
    max: float
    count: int


class DriftHistogramBin(BaseModel):
    bin_center: float
    ref_density: float
    mon_density: float


class FeatureDistributionDetail(BaseModel):
    feature_name: str
    ref_stats: FeatureStats
    mon_stats: FeatureStats
    cohens_d: float
    ks_statistic: float
    ks_pvalue: float
    welch_t_statistic: float
    welch_pvalue: float
    distribution_bins: List[DriftHistogramBin]


class MonthlyErrorItem(BaseModel):
    period: str
    count: int
    rmse: float
    mae: float
    r2: float
    mean_bias: float
    durbin_watson: float


class DriftFinding(BaseModel):
    category: str
    severity: str  # "info", "warning", "breach"
    title: str
    description: str
    supporting_statistic: str
    caveat: str


class DriftMetricsSummary(BaseModel):
    ref_rmse: float
    ref_mae: float
    ref_r2: float
    ref_mbe: float
    mon_rmse: float
    mon_mae: float
    mon_r2: float
    mon_mbe: float
    rmse_inflation_pct: float
    bias_shift: float


class DriftAnalysisRequest(BaseModel):
    model_type: str = Field(default="OLS")  # "OLS" or "PCR"
    k_components: Optional[int] = Field(default=3, ge=1, le=8)
    reference_partition: str = Field(default="train")  # "train", "q1", "q2"
    monitoring_partition: str = Field(default="test")  # "val", "test", "monitoring"
    selected_feature: str = Field(default="PT08.S1(CO)")
    rolling_window_hours: int = Field(default=72, ge=12, le=720)
    timeline_subsample: int = Field(default=250, ge=50, le=1000)


class DriftAnalysisResponse(BaseModel):
    model_type: str
    k_components: Optional[int]
    reference_label: str
    monitoring_label: str
    reference_rows: int
    monitoring_rows: int
    total_rows: int
    overall_metrics: DriftMetricsSummary
    durbin_watson_reference: float
    durbin_watson_monitoring: float
    durbin_watson_full: float
    timeline_series: List[DriftTimelinePoint]
    feature_drift_summary: List[FeatureDriftItem]
    selected_feature_detail: FeatureDistributionDetail
    monthly_progression: List[MonthlyErrorItem]
    explainable_findings: List[DriftFinding]
    scientific_notes: Dict[str, str]




