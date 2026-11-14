"""Drift Analysis & Sensor Monitoring endpoints.

Exposes calibration error progression, feature distribution drift,
rolling error dynamics, and retrospective statistical diagnostics.
"""

from fastapi import APIRouter, HTTPException

from api.schemas import DriftAnalysisRequest, DriftAnalysisResponse
from api.services import service

router = APIRouter(prefix="/drift", tags=["Drift Analysis"])


@router.post("/analyze", response_model=DriftAnalysisResponse)
def analyze_drift(request: DriftAnalysisRequest) -> DriftAnalysisResponse:
    """Execute retrospective sensor calibration drift and distribution analysis."""
    try:
        return service.analyze_drift(request)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
