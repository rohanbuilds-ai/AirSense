"""Overview page endpoints."""

from fastapi import APIRouter, HTTPException
from api.schemas import OverviewMetricsResponse, OverviewChartsResponse
from api.services import service

router = APIRouter(prefix="/overview", tags=["Overview"])


@router.get("/metrics", response_model=OverviewMetricsResponse)
def get_overview_metrics() -> OverviewMetricsResponse:
    """Return KPI summary cards and model comparison metrics for Overview page."""
    try:
        return service.get_overview_metrics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/charts", response_model=OverviewChartsResponse)
def get_overview_charts() -> OverviewChartsResponse:
    """Return chart series data for Overview page visualizations."""
    try:
        return service.get_overview_charts()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
