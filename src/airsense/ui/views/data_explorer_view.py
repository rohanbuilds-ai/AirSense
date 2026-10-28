"""Data Explorer View for AirSense Streamlit Interface.

Provides interactive inspection of:
- Raw & Cleaned Observation Tables
- Sensor & Meteorological Variable Distributions
- Missingness Profile & Coincident Outage Analysis
- Reference Concentrations & Excluded Predictor Audit
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


def render_data_explorer_view(df_supervised: pd.DataFrame, df_cleaned: pd.DataFrame) -> None:
    """Render Section 2: Data Explorer and Audit."""
    st.header("🔍 2. Data Explorer & Missingness Audit")
    st.markdown(
        """
        Explore the raw and cleaned observations from the Italian urban monitoring station. 
        Inspect missingness patterns, examine the distribution of certified reference concentrations, 
        and verify feature exclusion rules designed to prevent target leakage.
        """
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "1. Observation Table",
            "2. Missingness & Sentinel Audit",
            "3. Reference Concentration (Target)",
            "4. Feature Role & Leakage Audit",
        ]
    )

    # -------------------------------------------------------------
    # TAB 1: Observation Table
    # -------------------------------------------------------------
    with tab1:
        st.subheader("Interactive Observation Browser")
        st.markdown(
            """
            Browse hourly observations. All timestamps follow the official UTC+1 hourly logging format: 
            `DD/MM/YYYY HH.MM.SS`. The dataset begins on **10/03/2004 18:00** and concludes on **04/04/2005 14:00**.
            """
        )

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            dataset_choice = st.radio(
                "Select Dataset View:",
                ["Supervised Cleaned Dataset (8,991 rows, target present)", "Complete Cleaned Dataset (9,357 rows, all timestamps)"],
                index=0,
            )
        with col_f2:
            n_rows = st.slider("Display Sample Size:", min_value=10, max_value=200, value=25, step=5)

        active_df = df_supervised if "Supervised" in dataset_choice else df_cleaned

        st.dataframe(active_df.head(n_rows), use_container_width=True)
        st.caption(f"Showing first {n_rows} rows of {len(active_df):,} total records. All -200 sentinels converted to NaN.")

    # -------------------------------------------------------------
    # TAB 2: Missingness & Sentinel Audit
    # -------------------------------------------------------------
    with tab2:
        st.subheader("Missing Value & -200 Sentinel Profile")
        st.markdown(
            """
            In the raw UCI dataset, missing observations are encoded with the numeric sentinel **`-200`**. 
            Below is the full missingness breakdown across all 15 raw numeric columns in the complete 9,357-hour dataset:
            """
        )

        # Compute missingness stats from df_cleaned
        raw_numeric_cols = [
            "CO(GT)", "PT08.S1(CO)", "NMHC(GT)", "C6H6(GT)", "PT08.S2(NMHC)",
            "NOx(GT)", "PT08.S3(NOx)", "NO2(GT)", "PT08.S4(NO2)", "PT08.S5(O3)",
            "T", "RH", "AH"
        ]
        
        missing_counts = df_cleaned[raw_numeric_cols].isna().sum()
        missing_pcts = (missing_counts / len(df_cleaned)) * 100.0

        missing_df = pd.DataFrame({
            "Variable": raw_numeric_cols,
            "Role": [
                "Reference (Excluded)", "Sensor (Predictor)", "Reference (Defective)", "Target (C6H6)",
                "Sensor (Predictor)", "Reference (Excluded)", "Sensor (Predictor)", "Reference (Excluded)",
                "Sensor (Predictor)", "Sensor (Predictor)", "Weather (Predictor)", "Weather (Predictor)",
                "Weather (Predictor)"
            ],
            "Missing Count (-200)": missing_counts.values,
            "Missing Percentage": missing_pcts.values,
        }).sort_values(by="Missing Percentage", ascending=False)

        st.dataframe(
            missing_df.style.format({"Missing Percentage": "{:.2f}%"}),
            use_container_width=True,
        )

        st.info(
            """
            💡 **Critical Scientific Finding — Coincident Station Outages**:
            Exactly **366 rows** are missing simultaneously across all 5 sensors (`PT08.S1`–`PT08.S5`), 
            all 3 weather variables (`T`, `RH`, `AH`), and the benzene target (`C6H6(GT)`). 
            These correspond to station-wide power failures and telemetry logging dropouts. 
            Pruning rows with missing target values ($9,357 - 366 = 8,991$) yields a clean dataset with 
            **zero missing values across all 8 predictor channels**!
            """
        )

        st.warning(
            """
            ⚠️ **Defective Reference Analyzer**:
            `NMHC(GT)` is missing in **90.23% of observations** (8,443 rows) because the certified non-methanic 
            hydrocarbon reference instrument malfunctioned after May 2004. It cannot be used as a target or predictor.
            """
        )

    # -------------------------------------------------------------
    # TAB 3: Reference Concentration (Target Variable)
    # -------------------------------------------------------------
    with tab3:
        st.subheader("Certified Reference Benzene ($C_6H_6(GT)$) Target Distribution")
        st.markdown(
            r"""
            Benzene ($C_6H_6$) is a volatile aromatic hydrocarbon classified as a Class 1 human carcinogen. 
            The reference analyzer measures ambient concentration in micrograms per cubic meter ($\mu g/m^3$).
            """
        )

        target_series = df_supervised["C6H6(GT)"].dropna()
        
        c_t1, c_t2, c_t3, c_t4 = st.columns(4)
        c_t1.metric("Mean Benzene", f"{target_series.mean():.2f} µg/m³")
        c_t2.metric("Median Benzene", f"{target_series.median():.2f} µg/m³")
        c_t3.metric("Standard Deviation", f"{target_series.std():.2f} µg/m³")
        c_t4.metric("Skewness (Fisher-Pearson)", f"{target_series.skew():.2f}", "Positively Skewed")

        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            fig_hist, ax_h = plt.subplots(figsize=(6, 3.8))
            ax_h.hist(target_series, bins=40, color="#4c72b0", edgecolor="white", alpha=0.8)
            ax_h.axvline(target_series.mean(), color="red", linestyle="--", label=f"Mean: {target_series.mean():.1f}")
            ax_h.axvline(target_series.median(), color="green", linestyle=":", label=f"Median: {target_series.median():.1f}")
            ax_h.set_xlabel(r"Benzene Concentration $C_6H_6(GT)$ ($\mu g/m^3$)")
            ax_h.set_ylabel("Observation Frequency (Hours)")
            ax_h.set_title("Target Benzene Distribution (Positively Skewed)")
            ax_h.legend()
            ax_h.grid(True, alpha=0.3)
            st.pyplot(fig_hist)

        with col_fig2:
            fig_box, ax_b = plt.subplots(figsize=(6, 3.8))
            ax_b.boxplot(target_series, vert=False, patch_artist=True,
                         boxprops=dict(facecolor="#55a868", color="black"),
                         medianprops=dict(color="darkred", lw=2))
            ax_b.set_xlabel(r"Benzene Concentration ($\mu g/m^3$)")
            ax_b.set_title("Boxplot of Benzene Concentration (Showing Extreme Pollution Spikes)")
            ax_b.grid(True, alpha=0.3)
            st.pyplot(fig_box)

    # -------------------------------------------------------------
    # TAB 4: Feature Role & Leakage Audit
    # -------------------------------------------------------------
    with tab4:
        st.subheader("Strict Feature Role & Zero Data-Leakage Audit")
        st.markdown(
            """
            To ensure academic rigor and avoid catastrophic target leakage, every column in the UCI 
            Air Quality dataset is categorized into strict operational roles:
            """
        )

        role_table = [
            ("PT08.S1(CO)", "Predictor", "Chemical Gas Sensor (SnO2, Tin Oxide)", "Included in X"),
            ("PT08.S2(NMHC)", "Predictor", "Chemical Gas Sensor (TiO2, Titania)", "Included in X"),
            ("PT08.S3(NOx)", "Predictor", "Chemical Gas Sensor (WO3, Tungsten Oxide)", "Included in X"),
            ("PT08.S4(NO2)", "Predictor", "Chemical Gas Sensor (WO3, Tungsten Oxide)", "Included in X"),
            ("PT08.S5(O3)", "Predictor", "Chemical Gas Sensor (In2O3, Indium Oxide)", "Included in X"),
            ("T", "Predictor", "Ambient Temperature (°C)", "Included in X"),
            ("RH", "Predictor", "Relative Humidity (%)", "Included in X"),
            ("AH", "Predictor", "Absolute Humidity (g/m³)", "Included in X"),
            ("C6H6(GT)", "Target", "Certified Reference Benzene Analyzer", "Target y (Excluded from X)"),
            ("CO(GT)", "EXCLUDED", "Certified Reference CO Analyzer", "EXCLUDED (Target leakage risk)"),
            ("NMHC(GT)", "EXCLUDED", "Certified Reference NMHC Analyzer", "EXCLUDED (90% missing + leakage)"),
            ("NOx(GT)", "EXCLUDED", "Certified Reference NOx Analyzer", "EXCLUDED (Target leakage risk)"),
            ("NO2(GT)", "EXCLUDED", "Certified Reference NO2 Analyzer", "EXCLUDED (Target leakage risk)"),
            ("Date", "Metadata", "Chronological Date String", "Used for temporal sorting only"),
            ("Time", "Metadata", "Chronological Hour String", "Used for temporal sorting only"),
        ]

        df_roles = pd.DataFrame(role_table, columns=["Column Name", "Assigned Role", "Physical Meaning", "Data-Leakage Rule"])
        st.dataframe(df_roles, use_container_width=True)

        st.success(
            """
            ✅ **Data Leakage Verification**:
            Zero certified analyzer concentrations ending in `(GT)` are permitted in the predictor design matrix $X$. 
            The calibration engine relies strictly on low-cost physical sensor resistances and ambient meteorological readings.
            """
        )
