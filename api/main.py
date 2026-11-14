"""AirSense FastAPI Gateway Application.

Connects the React/TypeScript frontend to the existing Python mathematical core.
Conforms to: docs/frontend_integration_contract.md
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.health import router as health_router
from api.routes.dataset import router as dataset_router
from api.routes.overview import router as overview_router
from api.routes.calibration import router as calibration_router
from api.routes.math_lab import router as math_lab_router
from api.routes.pca import router as pca_router
from api.routes.drift import router as drift_router

app = FastAPI(
    title="AirSense API Gateway",
    description="REST Adapter for Mathematical Sensor Calibration, PCA & Drift Monitoring",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    openapi_url="/api/openapi.json",
)

# Enable CORS for the local Vite development server
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers under /api/v1
app.include_router(health_router, prefix="/api/v1")
app.include_router(dataset_router, prefix="/api/v1")
app.include_router(overview_router, prefix="/api/v1")
app.include_router(calibration_router, prefix="/api/v1")
app.include_router(math_lab_router, prefix="/api/v1")
app.include_router(pca_router, prefix="/api/v1")
app.include_router(drift_router, prefix="/api/v1")


@app.get("/")
def root_redirect():
    """Root redirect indicating API status and interactive documentation."""
    return {
        "project": "AirSense",
        "description": "Mathematical Sensor Calibration & Drift Monitoring Engine",
        "version": "0.1.0",
        "docs": "/api/docs",
        "health": "/api/v1/health",
    }
