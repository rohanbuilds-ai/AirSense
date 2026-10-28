"""Overview View for AirSense Streamlit Interface.

Covers:
- Problem Statement: Chemical sensor cross-sensitivity and drift calibration
- Dataset Provenance: Official UCI Air Quality Dataset (De Vito et al., 2008)
- Scientific & Engineering Limitations: Temperature/humidity confounding, missingness, non-linearity
- End-to-End Computational Pipeline Architecture
- Data Licensing & Academic Attribution
"""

from pathlib import Path
import pandas as pd
import streamlit as st


def render_overview_view(df_supervised: pd.DataFrame, df_cleaned: pd.DataFrame, df_results: pd.DataFrame) -> None:
    """Render Section 1: Overview, Provenance, Limitations, and Pipeline."""
    st.header("🏛️ 1. Project Overview & Scientific Provenance")
    st.markdown(
        """
        **AirSense** is an academic scientific computing engine and portfolio project designed for 
        the university syllabus *Mathematical Foundations for AI and Data Science*. 
        It investigates linear algebra, spectral decompositions, least-squares calibration, 
        and statistical hypothesis testing on real-world environmental sensor data.
        """
    )

    st.markdown("---")

    # High-level Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Dataset Duration", "389.8 Contiguous Days", "March 2004 – April 2005")
    c1.caption("Hourly resolution: 9,357 records")

    c2.metric("Supervised Samples", f"{len(df_supervised):,} Hours", "Zero Missing Predictors")
    c2.caption("Pruned 366 power/logging outages")

    c3.metric("Predictor Features", "8 Channels", "5 Chemical + 3 Meteorological")
    c3.caption("All reference (GT) columns excluded")

    c4.metric("Optimal Regularized R²", "0.8626 (Test)", "PCR k=3 vs Full OLS 0.8079")
    c4.caption("15.4% out-of-time RMSE reduction")

    st.markdown("---")

    # Section 1.1: Problem Statement
    st.subheader("1.1 Problem Statement: Low-Cost Gas-Sensor Calibration")
    st.markdown(
        """
        Urban air-quality monitoring stations equipped with certified reference optical analyzers 
        (such as gas chromatographs and chemiluminescence detectors) provide gold-standard 
        accuracy but cost tens of thousands of dollars, limiting spatial monitoring resolution. 
        Solid-state metal-oxide semiconductor (MOX) sensors offer a miniature, low-power alternative, 
        but suffer from severe physical and mathematical obstacles:
        
        1. **Broad Chemical Cross-Sensitivity**: Metal-oxide sensors react non-specifically to broad classes of reducing 
           and oxidizing gases. For instance, carbon monoxide ($CO$) and non-methanic hydrocarbons ($NMHC$) 
           activate overlapping surface oxidation sites, producing extreme multi-collinearity ($r > 0.88$) in the sensor design matrix.
        2. **Physical Sensor Drift & Aging**: Prolonged operation causes grain boundary restructuring, chemical poisoning, 
           and baseline resistance loss in semiconductor thin films. Over time, static linear calibration parameters fail.
        3. **Environmental Confounding**: Changes in ambient temperature ($T$) and humidity ($RH, AH$) directly alter 
           carrier density in metal oxides, confounding chemical signal interpretation.
           
        **Core Research Objective**:  
        Evaluate how Principal Component Analysis (PCA) dimensionality reduction regularizes least-squares calibration, 
        mitigates the ill-conditioning caused by collinear sensors, and affects temporal stability over a 13-month field campaign.
        """
    )

    st.markdown("---")

    # Section 1.2: Dataset Provenance
    st.subheader("1.2 Dataset Provenance & Sensor Specifications")
    st.markdown(
        r"""
        - **Source**: UCI Machine Learning Repository — Air Quality Dataset ([Dataset ID 360](https://archive.ics.uci.edu/dataset/360/air+quality)).
        - **Deployment Site**: An Italian city with high traffic density.
        - **Deployment Period**: March 10, 2004, to April 4, 2005 ($389.8$ contiguous days, exactly $9,357$ hourly slots).
        - **Reference Analyzer**: Co-located certified reference analyzer measuring ground truth $C_6H_6$ (benzene), $CO$, $NO_x$, and $NO_2$.
        
        | Sensor Code | Target Gas Nominal Sensitivity | Operating Principle | Role in AirSense |
        |---|---|---|---|
        | `PT08.S1(CO)` | Carbon Monoxide ($CO$) | Tin Oxide ($SnO_2$) Resistance | Predictive Feature |
        | `PT08.S2(NMHC)` | Non-Metanic Hydrocarbons | Titania ($TiO_2$) Resistance | Predictive Feature |
        | `PT08.S3(NOx)` | Nitrogen Oxides ($NO_x$) | Tungsten Oxide ($WO_3$) Resistance | Predictive Feature |
        | `PT08.S4(NO2)` | Nitrogen Dioxide ($NO_2$) | Tungsten Oxide ($WO_3$) Resistance | Predictive Feature |
        | `PT08.S5(O3)` | Ozone ($O_3$) | Indium Oxide ($In_2O_3$) Resistance | Predictive Feature |
        | `T` | Temperature (°C) | Thermistor Sensor | Environmental Covariate |
        | `RH` | Relative Humidity (%) | Capacitive Sensor | Environmental Covariate |
        | `AH` | Absolute Humidity ($g/m^3$) | Derived Physical Variable | Environmental Covariate |
        | `C6H6(GT)` | Benzene Concentration ($\mu g/m^3$) | Certified Reference Gas Chromatograph | **Primary Prediction Target** |
        """
    )

    st.markdown("---")

    # Section 1.3: Pipeline Architecture
    st.subheader("1.3 End-to-End Computational Architecture")
    st.markdown(
        """
        The pipeline adheres to strict two-tier architecture: from-scratch pure NumPy/SciPy linear algebra algorithms 
        in `airsense.math_core`, paired with leakage-free scientific pipelines in `airsense.calibration` and `airsense.pca_model`.
        """
    )

    st.code(
        """
  [1. Raw UCI CSV Ingestion]
         │ (Delimiter: ';', Decimal: ',', Strip 114 blank trailing lines)
         ▼
  [2. Missing Sentinel Conversion]
         │ (-200 -> NaN; Prune 366 outage rows where target is missing)
         ▼
  [3. Strict Chronological Split]
         ├── 60% Train (Mar 10, 2004 - Nov 3, 2004; N = 5,394)
         ├── 20% Val   (Nov 3, 2004 - Jan 17, 2005; N = 1,798)
         └── 20% Test  (Jan 17, 2005 - Apr 4, 2005; N = 1,799)
         │
  [4. Zero-Leakage Preprocessing]
         │ (Standardize Z = (X - μ_train) / σ_train strictly using Train parameters)
         ▼
  [5. Mathematical Core Solvers]
         ├── From-scratch QR Decomposition via Modified Gram-Schmidt
         ├── Thin SVD / Jacobi Covariance Diagonalization (S = V Λ V^T)
         └── Least-Squares Solvers (avoiding condition squaring κ(X^T X) = κ(X)^2)
         │
  [6. Principal Component Regression (PCR)]
         │ (Evaluate k ∈ {1, ..., 8} components; Back-project physical weights)
         ▼
  [7. Quality-Control Drift & Residual Analysis]
         ├── Two-sided CUSUM Drift Monitoring (Slack allowance δ, Alarm limit H)
         └── Autocorrelation (Durbin-Watson DW = 0.2062 << 2.0)
         │
  [8. Probability & Statistical Inference Laboratory]
         └── CLT simulations, Normal MLE bias, CI duality, Seasonal ANOVA, Power
        """,
        language="text",
    )

    st.markdown("---")

    # Section 1.4: Limitations & Boundary Conditions
    st.subheader("1.4 Scientific & Modeling Limitations")
    st.warning(
        r"""
        ⚠️ **Documented Scientific Limitations**:
        1. **Ambient Confounding**: While metal-oxide sensors undergo physical aging, outdoor field measurements are 
           confounded by annual seasonal temperature swings (Summer high $40^\circ\text{C}$ to Winter low $0^\circ\text{C}$). 
           Changing error behavior cannot be attributed purely to sensor degradation without laboratory chamber verification.
        2. **Coincident Missingness**: 366 hourly observations simultaneously dropped all 5 sensors, 3 weather variables, 
           and the reference target due to station-wide electrical failures.
        3. **Defective Reference NMHC**: The certified $NMHC(GT)$ analyzer failed after May 2004 ($90.23\%$ missing), 
           confirming why only the 5 metal-oxide sensors and 3 weather variables are valid predictors.
        4. **Linear Model Assumption**: Real metal oxide response follows non-linear power-law chemisorption kinetics 
           ($R \propto C^{-\alpha}$). Linear calibration provides an interpretable first-order approximation but incurs 
           residual heteroscedasticity.
        """
    )

    st.markdown("---")

    # Section 1.5: Licensing & Attribution
    st.subheader("1.5 Data Licensing, Ethics & Attribution")
    st.info(
        """
        **Attribution & Academic Citation**:
        The UCI Air Quality Dataset is provided for academic, research, and educational use courtesy of:
        
        > **S. De Vito, E. Massera, M. Piga, L. Martinotto, G. Di Francia**,  
        > *"On field calibration of an electronic nose for benzene estimation in an urban environment,"*  
        > *Sensors and Actuators B: Chemical*, vol. 129, no. 2, pp. 750–757, 2008.  
        > DOI: [10.1016/j.snb.2007.09.060](https://doi.org/10.1016/j.snb.2007.09.060)
        
        *License Notice*: This software project is distributed under the MIT License for educational and research evaluation. 
        It is intended for scientific research and university coursework, not for commercial life-safety air quality compliance.
        """
    )
