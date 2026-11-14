/**
 * Typed API Client for AirSense REST Gateway.
 */

import {
  HealthResponse,
  DatasetSummaryResponse,
  DatasetAuditResponse,
  PaginatedRowsResponse,
  MatrixPreviewResponse,
  OverviewMetricsResponse,
  OverviewChartsResponse,
  BenchmarkMatrixResponse,
  CalibrationRunRequest,
  CalibrationRunResponse,
  MatrixExtractRequest,
  MatrixExtractResponse,
  GaussianEliminationRequest,
  GaussianEliminationResponse,
  SubspaceAnalysisRequest,
  SubspaceAnalysisResponse,
  GramSchmidtRequest,
  GramSchmidtResponse,
  LeastSquaresRequest,
  LeastSquaresResponse,
  SpectralAnalysisRequest,
  SpectralAnalysisResponse,
  PCARequest,
  PCAResponse,
  DriftAnalysisRequest,
  DriftAnalysisResponse,
} from '../types/api';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl.replace(/\/$/, '');
  }

  private async fetchJson<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint.startsWith('/') ? endpoint : `/${endpoint}`}`;
    try {
      const response = await fetch(url, {
        headers: {
          'Content-Type': 'application/json',
          Accept: 'application/json',
          ...options.headers,
        },
        ...options,
      });

      if (!response.ok) {
        const errorBody = await response.text();
        throw new Error(`API HTTP ${response.status}: ${errorBody || response.statusText}`);
      }

      return (await response.json()) as T;
    } catch (err) {
      if (err instanceof Error) {
        throw err;
      }
      throw new Error(`Unknown network error connecting to ${url}`);
    }
  }

  public async checkHealth(): Promise<HealthResponse> {
    return this.fetchJson<HealthResponse>('/health');
  }

  public async getDatasetSummary(): Promise<DatasetSummaryResponse> {
    return this.fetchJson<DatasetSummaryResponse>('/dataset/summary');
  }

  public async getDatasetAudit(): Promise<DatasetAuditResponse> {
    return this.fetchJson<DatasetAuditResponse>('/dataset/audit');
  }

  public async getDatasetRows(limit: number = 50, offset: number = 0): Promise<PaginatedRowsResponse> {
    return this.fetchJson<PaginatedRowsResponse>(`/dataset/rows?limit=${limit}&offset=${offset}`);
  }

  public async getMatrixPreview(
    offset: number = 0,
    limit: number = 10,
    partition: string = 'all'
  ): Promise<MatrixPreviewResponse> {
    return this.fetchJson<MatrixPreviewResponse>(
      `/dataset/matrix-preview?offset=${offset}&limit=${limit}&partition=${partition}`
    );
  }

  public async getOverviewMetrics(): Promise<OverviewMetricsResponse> {
    return this.fetchJson<OverviewMetricsResponse>('/overview/metrics');
  }

  public async getOverviewCharts(): Promise<OverviewChartsResponse> {
    return this.fetchJson<OverviewChartsResponse>('/overview/charts');
  }

  public async runCalibration(config: CalibrationRunRequest): Promise<CalibrationRunResponse> {
    return this.fetchJson<CalibrationRunResponse>('/calibration/run', {
      method: 'POST',
      body: JSON.stringify(config),
    });
  }

  public async getBenchmarkMatrix(): Promise<BenchmarkMatrixResponse> {
    return this.fetchJson<BenchmarkMatrixResponse>('/calibration/benchmark-matrix');
  }

  // Linear Algebra Lab Methods
  public async extractMatrix(request: MatrixExtractRequest = {}): Promise<MatrixExtractResponse> {
    return this.fetchJson<MatrixExtractResponse>('/math/extract-matrix', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  public async computeGaussianElimination(request: GaussianEliminationRequest): Promise<GaussianEliminationResponse> {
    return this.fetchJson<GaussianEliminationResponse>('/math/gaussian-elimination', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  public async analyzeSubspaces(request: SubspaceAnalysisRequest): Promise<SubspaceAnalysisResponse> {
    return this.fetchJson<SubspaceAnalysisResponse>('/math/subspaces', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  public async compareGramSchmidt(request: GramSchmidtRequest): Promise<GramSchmidtResponse> {
    return this.fetchJson<GramSchmidtResponse>('/math/gram-schmidt', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  public async solveLeastSquares(request: LeastSquaresRequest): Promise<LeastSquaresResponse> {
    return this.fetchJson<LeastSquaresResponse>('/math/least-squares', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  public async analyzeSpectral(request: SpectralAnalysisRequest = {}): Promise<SpectralAnalysisResponse> {
    return this.fetchJson<SpectralAnalysisResponse>('/math/spectral', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // PCA Explorer Methods
  public async analyzePCA(request: PCARequest = {}): Promise<PCAResponse> {
    return this.fetchJson<PCAResponse>('/pca/analyze', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  // Drift Analysis & Sensor Monitoring Methods
  public async analyzeDrift(request: DriftAnalysisRequest = {}): Promise<DriftAnalysisResponse> {
    return this.fetchJson<DriftAnalysisResponse>('/drift/analyze', {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }
}

export const apiClient = new ApiClient();
