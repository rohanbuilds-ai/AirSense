"""Health check endpoint."""

from fastapi import APIRouter
from api.schemas import HealthResponse
from api.services import service

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return backend service status and dataset availability."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        dataset_loaded=(service.supervised_df is not None and not service.supervised_df.empty),
        supervised_rows=len(service.supervised_df) if service.supervised_df is not None else 0,
        cleaned_rows=len(service.cleaned_df) if service.cleaned_df is not None else 0,
    )
