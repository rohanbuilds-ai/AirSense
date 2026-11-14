"""Linear Algebra Laboratory API endpoints.

Exposes pure mathematical algorithms from src/airsense/math_core.py
for interactive experimentation with real UCI Air Quality data and demonstration matrices.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import (
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
)
from api.services import service

router = APIRouter(prefix="/math", tags=["Linear Algebra Lab"])


@router.post("/extract-matrix", response_model=MatrixExtractResponse)
def extract_matrix(request: MatrixExtractRequest) -> MatrixExtractResponse:
    """Extract real data matrix from the training partition with optional standardization and intercept."""
    try:
        return service.extract_matrix(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gaussian-elimination", response_model=GaussianEliminationResponse)
def compute_gaussian_elimination(request: GaussianEliminationRequest) -> GaussianEliminationResponse:
    """Perform step-by-step Gaussian Elimination / RREF and optional PA=LU decomposition."""
    try:
        if not request.matrix or len(request.matrix) == 0:
            raise HTTPException(status_code=400, detail="Matrix cannot be empty.")
        return service.compute_gaussian_elimination(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/subspaces", response_model=SubspaceAnalysisResponse)
def analyze_subspaces(request: SubspaceAnalysisRequest) -> SubspaceAnalysisResponse:
    """Compute Column Space Basis, Null Space Basis, Rank, and Nullity verifying the Rank-Nullity Theorem."""
    try:
        if not request.matrix or len(request.matrix) == 0:
            raise HTTPException(status_code=400, detail="Matrix cannot be empty.")
        return service.analyze_subspaces(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/gram-schmidt", response_model=GramSchmidtResponse)
def compare_gram_schmidt(request: GramSchmidtRequest) -> GramSchmidtResponse:
    """Compare Classical vs Modified Gram-Schmidt numerical stability and orthogonality loss."""
    try:
        if not request.matrix or len(request.matrix) == 0:
            raise HTTPException(status_code=400, detail="Matrix cannot be empty.")
        return service.compare_gram_schmidt(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/least-squares", response_model=LeastSquaresResponse)
def solve_projection_least_squares(request: LeastSquaresRequest) -> LeastSquaresResponse:
    """Compute orthogonal projection, QR least squares solution, and condition number comparison."""
    try:
        if not request.matrix or len(request.matrix) == 0:
            raise HTTPException(status_code=400, detail="Matrix cannot be empty.")
        if not request.target_vector or len(request.target_vector) != len(request.matrix):
            raise HTTPException(
                status_code=400,
                detail=f"Target vector length ({len(request.target_vector) if request.target_vector else 0}) must match matrix row count ({len(request.matrix)})."
            )
        return service.solve_projection_least_squares(request)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/spectral", response_model=SpectralAnalysisResponse)
def analyze_spectral(request: SpectralAnalysisRequest) -> SpectralAnalysisResponse:
    """Compute covariance matrix, PSD check, and Jacobi symmetric eigenvalue decomposition."""
    try:
        return service.analyze_spectral(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
