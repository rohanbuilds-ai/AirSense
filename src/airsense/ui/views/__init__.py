"""AirSense UI View Modules.

Provides modular views for each section of the Streamlit application:
1. Overview & Provenance (overview_view)
2. Data Explorer & Missingness (data_explorer_view)
3. Linear-Algebra Laboratory (math_lab_view)
4. PCA Explorer & Low-Rank Reconstruction (pca_explorer_view)
5. Sensor Calibration Experiment (calibration_view)
6. Sensor Drift & Error Dynamics (drift_view)
7. Probability & Statistics Laboratory (inference_view)
8. Methodology & Viva Defense Dossier (methodology_view)
"""

from airsense.ui.views.overview_view import render_overview_view
from airsense.ui.views.data_explorer_view import render_data_explorer_view
from airsense.ui.views.math_lab_view import render_math_lab_view
from airsense.ui.views.pca_explorer_view import render_pca_explorer_view
from airsense.ui.views.calibration_view import render_calibration_view
from airsense.ui.views.drift_view import render_drift_view
from airsense.ui.views.inference_view import render_inference_view
from airsense.ui.views.methodology_view import render_methodology_view

__all__ = [
    "render_overview_view",
    "render_data_explorer_view",
    "render_math_lab_view",
    "render_pca_explorer_view",
    "render_calibration_view",
    "render_drift_view",
    "render_inference_view",
    "render_methodology_view",
]
