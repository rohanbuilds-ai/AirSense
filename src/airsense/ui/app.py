"""AirSense: Mathematical Engine for Sensor Calibration, PCA, and Drift Monitoring.

Main Streamlit Application Engine.
"""

from pathlib import Path
import pandas as pd
import streamlit as st

from airsense.ui.views import (
    render_overview_view,
    render_data_explorer_view,
    render_math_lab_view,
    render_pca_explorer_view,
    render_calibration_view,
    render_drift_view,
    render_inference_view,
    render_methodology_view,
)

# Page Configuration
st.set_page_config(
    page_title="AirSense | Mathematical Engine & Calibration Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_datasets():
    """Load preprocessed datasets safely with caching."""
    # Look for data directory relative to root or module
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    supervised_path = base_dir / "data" / "processed" / "air_quality_supervised.csv"
    cleaned_path = base_dir / "data" / "processed" / "air_quality_cleaned.csv"
    results_path = base_dir / "reports" / "experiment_results.csv"

    df_supervised = pd.DataFrame()
    df_cleaned = pd.DataFrame()
    df_results = pd.DataFrame()

    if supervised_path.exists():
        df_supervised = pd.read_csv(supervised_path)
    if cleaned_path.exists():
        df_cleaned = pd.read_csv(cleaned_path)
    if results_path.exists():
        df_results = pd.read_csv(results_path)

    return df_supervised, df_cleaned, df_results


def run_app():
    """Main application navigation router."""
    df_supervised, df_cleaned, df_results = load_datasets()

    st.sidebar.title("AirSense Navigation")
    st.sidebar.markdown(
        """
        **Undergraduate AI & Scientific Computing Mini-Project**  
        *Mathematical Foundations for AI & Data Science*
        """
    )

    sections = [
        "🏛️ 1. Overview & Provenance",
        "🔍 2. Data Explorer & Missingness",
        "📐 3. Linear-Algebra Laboratory",
        "🌐 4. PCA Explorer & Low-Rank",
        "🔬 5. Sensor Calibration Experiment",
        "📉 6. Sensor Drift & Error Dynamics",
        "📊 7. Probability & Statistics Lab",
        "🎓 8. Methodology & Viva Dossier",
    ]

    selected_section = st.sidebar.radio("Select Application Section:", sections)

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Viva Tip**: Every algorithm is implemented from scratch in `airsense.math_core` "
        "and numerically validated against SciPy and NumPy within machine precision ($10^{-12}$)."
    )
    st.sidebar.caption("Dataset: UCI Air Quality (ID 360) | March 2004 – April 2005")

    # Route to selected section
    if "1. Overview" in selected_section:
        render_overview_view(df_supervised, df_cleaned, df_results)
    elif "2. Data Explorer" in selected_section:
        render_data_explorer_view(df_supervised, df_cleaned)
    elif "3. Linear-Algebra" in selected_section:
        render_math_lab_view()
    elif "4. PCA Explorer" in selected_section:
        render_pca_explorer_view(df_supervised)
    elif "5. Sensor Calibration" in selected_section:
        render_calibration_view(df_supervised)
    elif "6. Sensor Drift" in selected_section:
        render_drift_view(df_supervised)
    elif "7. Probability & Statistics" in selected_section:
        render_inference_view(df_supervised)
    elif "8. Methodology & Viva" in selected_section:
        render_methodology_view()


if __name__ == "__main__":
    run_app()
