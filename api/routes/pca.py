"""PCA Explorer API endpoints.

Exposes PCA SVD decomposition, component loadings, score projections,
and low-rank feature reconstruction for interactive scientific exploration.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import PCARequest, PCAResponse
from api.services import service

router = APIRouter(prefix="/pca", tags=["PCA Explorer"])


@router.post("/analyze", response_model=PCAResponse)
def analyze_pca(request: PCARequest) -> PCAResponse:
    """Run PCA analysis on the selected feature set and partition.
    
    All transformations (means, standard deviations, components V) are fitted
    exclusively on the training partition with zero data leakage.
    """
    try:
        return service.analyze_pca(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
