"""Calibration and benchmark endpoints."""

from fastapi import APIRouter, HTTPException
from api.schemas import BenchmarkMatrixResponse, CalibrationRunRequest, CalibrationRunResponse
from api.services import service

router = APIRouter(prefix="/calibration", tags=["Calibration"])


@router.get("/benchmark-matrix", response_model=BenchmarkMatrixResponse)
def get_benchmark_matrix() -> BenchmarkMatrixResponse:
    """Return the reproducible 8-configuration PCR/OLS benchmark table."""
    try:
        charts = service.get_overview_charts()
        return BenchmarkMatrixResponse(models=charts.model_comparison)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/run", response_model=CalibrationRunResponse)
def run_calibration_experiment(request: CalibrationRunRequest) -> CalibrationRunResponse:
    """Run interactive sensor calibration experiment with dynamic feature subset and model selection.
    
    Evaluates exclusively on training and validation partitions with zero data leakage.
    Final test partition remains sealed.
    """
    try:
        return service.run_calibration(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

