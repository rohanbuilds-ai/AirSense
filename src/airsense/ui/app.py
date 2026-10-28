"""AirSense: Mathematical Engine for Sensor Calibration, PCA, and Drift Monitoring.

Main Streamlit Application Entrypoint.
"""

from pathlib import Path
import pandas as pd
import streamlit as st
from PIL import Image

from airsense.ui.views.math_lab import render_math_lab
from airsense.ui.views.calibration_view import render_calibration_view
from airsense.ui.views.drift_view import render_drift_view
from airsense.ui.views.inference_view import render_inference_view

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


def render_overview(df_supervised: pd.DataFrame, df_cleaned: pd.DataFrame, df_results: pd.DataFrame):
    """Render the project overview, dataset audit, and executive summary."""
    st.title("🔬 AirSense: Mathematical Engine & Calibration Lab")
    st.subheader("Sensor Calibration, Principal Component Analysis, and Temporal Drift Monitoring")

    st.markdown(
        """
        Welcome to **AirSense**, a research-grade scientific computing and linear algebra engine 
        built on the official **UCI Air Quality Dataset** (March 2004 – April 2005).
        
        The project addresses the university syllabus for *Mathematical Foundations for AI and Data Science*, 
        following the rigorous explanatory paradigm:
        $$\\text{Concept} \\longrightarrow \\text{Purpose} \\longrightarrow \\text{Outcome}$$
        """
    )

    st.markdown("---")

    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    col_stat1.metric("Total Raw Records", "9,357 hours", "389.8 contiguous days")
    col_stat2.metric("Supervised Observations", f"{len(df_supervised):,}", "Zero Missing Predictors")
    col_stat3.metric("Predictor Features", "8 Channels", "5 Chemical + 3 Weather")
    col_stat4.metric("Optimal PCR Test R²", "0.8626", "k = 3 vs Full OLS 0.8079", delta_color="normal")

    st.markdown("---")

    st.subheader("🏛️ Core Research Question & Key Findings")
    st.markdown(
        """
        > **Research Question**: *How does PCA-based dimensionality reduction affect the predictive 
        > accuracy and temporal stability of least-squares calibration for benzene concentration 
        > using real-world gas-sensor measurements?*
        """
    )

    c_box1, c_box2 = st.columns(2)
    with c_box1:
        st.markdown("### 🔍 1. In-Sample vs. Out-of-Time Tradeoff")
        st.write(
            "- **In-Sample / Short-Term**: Full-rank OLS achieves the highest fit ($R^2 = 0.9423$ on validation) "
            "because chemical selectivity information is distributed across all 5 metal-oxide sensors.\n"
            "- **Out-of-Time / Long-Term ($1$ Year Later)**: Full-rank OLS degrades significantly ($R^2 = 0.8079$, $\\text{RMSE} = 2.855\\ \\mu g/m^3$).\n"
            "- **PCR ($k=3, 4$) Regularization**: Retaining 3 or 4 principal components achieves superior test accuracy "
            "($R^2 = 0.8626$, $\\text{RMSE} = 2.415\\ \\mu g/m^3$) by discarding the collinear tail that overfits to seasonal noise."
        )

    with c_box2:
        st.markdown("### 📉 2. Physical Sensor Aging & Ambient Drift")
        st.write(
            "- **Systemic Drift**: Over 13 months, Mean Bias Error (MBE) drifts from $+0.44\\ \\mu g/m^3$ in March 2004 "
            "down to **$-3.69\\ \\mu g/m^3$** in April 2005.\n"
            "- **Physical Mechanism**: Metal-oxide (MOX) sensors suffer surface contamination and gradual loss of baseline "
            "resistance, causing static linear calibrations to systematically overpredict.\n"
            "- **Autocorrelation**: Durbin-Watson statistic is $0.2062 \\ll 2.0$, demonstrating strong positive serial correlation in hourly error streams."
        )

    st.markdown("---")

    st.subheader("📊 Experimental Benchmarking Summary Table")
    if not df_results.empty:
        st.dataframe(
            df_results.style.format({
                "Cum_Variance_Pct": "{:.2f}%",
                "Train_RMSE": "{:.3f}",
                "Train_R2": "{:.4f}",
                "Val_RMSE": "{:.3f}",
                "Val_MAE": "{:.3f}",
                "Val_R2": "{:.4f}",
                "Test_RMSE": "{:.3f}",
                "Test_MAE": "{:.3f}",
                "Test_R2": "{:.4f}",
            }),
            use_container_width=True,
        )
    else:
        st.info("Benchmark results file not found.")

    st.markdown("---")

    st.subheader("🖼️ Ingestion & Exploratory Audit Artifacts")
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    figures_dir = base_dir / "reports" / "figures"

    audit_tabs = st.tabs(["1. Missingness Profile", "2. Target Time Series", "3. Sensor Distributions", "4. Correlation Structure"])
    
    with audit_tabs[0]:
        img_path = figures_dir / "01_missingness_audit.png"
        if img_path.exists():
            st.image(str(img_path), caption="Missingness Profile across all 15 raw attributes (-200 sentinel).", use_container_width=True)
    with audit_tabs[1]:
        img_path = figures_dir / "02_target_time_series.png"
        if img_path.exists():
            st.image(str(img_path), caption="Target Benzene C6H6(GT) Concentration over 13-month timeline.", use_container_width=True)
    with audit_tabs[2]:
        img_path = figures_dir / "03_sensor_distributions.png"
        if img_path.exists():
            st.image(str(img_path), caption="Distribution and skewness of raw gas-sensor responses.", use_container_width=True)
    with audit_tabs[3]:
        img_path = figures_dir / "04_correlation_heatmap.png"
        if img_path.exists():
            st.image(str(img_path), caption="Pearson Correlation Heatmap showing severe collinearity among metal-oxide sensors (r > 0.85).", use_container_width=True)


def main():
    """Main application navigation."""
    df_supervised, df_cleaned, df_results = load_datasets()

    st.sidebar.title("AirSense Navigation")
    st.sidebar.markdown(
        """
        **Undergraduate AI & Scientific Computing Mini-Project**  
        *Mathematical Foundations for AI & Data Science*
        """
    )

    page = st.sidebar.radio(
        "Select Workspace Module:",
        [
            "🏛️ Overview & Audit",
            "📐 Mathematical Foundations Lab",
            "🔬 Calibration & PCR Engine",
            "📉 Sensor Drift & Residuals",
            "📊 Probability & Inference Lab",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "💡 **Viva Tip**: Every algorithm is implemented from scratch in `airsense.math_core` "
        "and numerically validated against SciPy and NumPy within machine tolerance."
    )

    if page == "🏛️ Overview & Audit":
        render_overview(df_supervised, df_cleaned, df_results)
    elif page == "📐 Mathematical Foundations Lab":
        render_math_lab()
    elif page == "🔬 Calibration & PCR Engine":
        render_calibration_view(df_supervised)
    elif page == "📉 Sensor Drift & Residuals":
        render_drift_view(df_supervised)
    elif page == "📊 Probability & Inference Lab":
        render_inference_view(df_supervised)


if __name__ == "__main__":
    main()
