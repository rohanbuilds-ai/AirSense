"""Dataset inspection and audit endpoints."""

from typing import Optional
from fastapi import APIRouter, Query, HTTPException
from api.schemas import (
    DatasetSummaryResponse,
    DatasetAuditResponse,
    PaginatedRowsResponse,
    MatrixPreviewResponse,
)
from api.services import service

router = APIRouter(prefix="/dataset", tags=["Dataset"])


@router.get("/summary", response_model=DatasetSummaryResponse)
def get_dataset_summary() -> DatasetSummaryResponse:
    """Return dataset provenance, schema, and column metadata."""
    try:
        return service.get_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/audit", response_model=DatasetAuditResponse)
def get_dataset_audit() -> DatasetAuditResponse:
    """Return complete dataset audit, missingness, target stats, and feature correlations."""
    try:
        return service.get_audit()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rows", response_model=PaginatedRowsResponse)
def get_dataset_rows(
    limit: int = Query(50, ge=1, le=500, description="Rows to return"),
    offset: int = Query(0, ge=0, description="Starting row index"),
) -> PaginatedRowsResponse:
    """Return paginated observations from the cleaned dataset."""
    try:
        data = service.get_paginated_rows(limit=limit, offset=offset)
        return PaginatedRowsResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/matrix-preview", response_model=MatrixPreviewResponse)
def get_matrix_preview(
    offset: int = Query(0, ge=0, description="Starting row index"),
    limit: int = Query(10, ge=1, le=100, description="Rows to return"),
    partition: str = Query("all", description="Partition filter: all, train, val, test"),
) -> MatrixPreviewResponse:
    """Return real mathematical matrix representation with raw and standardized values."""
    try:
        return service.get_matrix_preview(offset=offset, limit=limit, partition=partition)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
