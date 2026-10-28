"""Sensor Drift Monitoring & Residual Dynamics View.

Provides interactive diagnostics of long-term sensor degradation,
CUSUM shift detection, and ambient environmental confounding.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from airsense.calibration import chronological_split, LinearCalibrationModel
from airsense.evaluation import durbin_watson_statistic, temporal_error_breakdown


def compute_cusum(residuals: np.ndarray, allowance: float = 0.5, threshold: float = 5.0):
    """Compute two-sided Cumulative Sum (CUSUM) on residual stream.
    
    Parameters
    ----------
    residuals : np.ndarray
        Array of calibration errors (e_t = y_t - y_pred_t).
    allowance : float
        Slack allowance parameter (k_slack in standard CUSUM literature).
    threshold : float
        Decision limit / alarm threshold (h).
        
    Returns
    -------
    tuple of (s_pos, s_neg, alarms_pos, alarms_neg)
    """
    n = len(residuals)
    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    alarms_pos = []
    alarms_neg = []

    for t in range(1, n):
        s_pos[t] = max(0.0, s_pos[t - 1] + residuals[t] - allowance)
        s_neg[t] = max(0.0, s_neg[t - 1] - residuals[t] - allowance)
        if s_pos[t] > threshold:
            alarms_pos.append(t)
        if s_neg[t] > threshold:
            alarms_neg.append(t)

    return s_pos, s_neg, np.array(alarms_pos), np.array(alarms_neg)


def render_drift_view(df_supervised: pd.DataFrame) -> None:
    """Render the interactive Sensor Drift & Residual Dynamics view."""
    st.header("📉 Sensor Drift Monitoring & Residual Dynamics")
    st.markdown(
        """
        Low-cost metal-oxide sensors undergo physical aging, surface contamination, and chemical 
        depletion over their operational lifespan. This view monitors calibration residuals 
        $e_t = y_t - \\hat{y}_t$ over the contiguous 13-month timeline (March 2004 to April 2005) 
        to detect systemic drift and analyze environmental confounding.
        """
    )

    if df_supervised.empty:
        st.error("Supervised dataset is empty or not loaded.")
        return

    # Train model on Train partition to monitor out-of-sample drift
    train_df, val_df, test_df = chronological_split(df_supervised, train_ratio=0.6, val_ratio=0.2)
    feature_cols = [
        "PT08.S1(CO)",
        "PT08.S2(NMHC)",
        "PT08.S3(NOx)",
        "PT08.S4(NO2)",
        "PT08.S5(O3)",
        "T",
        "RH",
        "AH",
    ]
    target_col = "C6H6(GT)"

    model = LinearCalibrationModel()
    model.fit(train_df[feature_cols].to_numpy(), train_df[target_col].to_numpy())

    # Predict across entire contiguous supervised timeline
    X_all = df_supervised[feature_cols].to_numpy()
    y_all = df_supervised[target_col].to_numpy()
    preds_all = model.predict(X_all)
    residuals_all = y_all - preds_all

    df_eval = df_supervised.copy()
    df_eval["Residual"] = residuals_all
    df_eval["Predicted"] = preds_all

    dw_stat = durbin_watson_statistic(residuals_all)

    # Top metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Overall Mean Residual", f"{np.mean(residuals_all):+.3f} µg/m³")
    c2.metric("Residual Std Deviation", f"{np.std(residuals_all):.3f} µg/m³")
    c3.metric(
        "Durbin-Watson Statistic",
        f"{dw_stat:.4f}",
        delta="Severe positive autocorrelation" if dw_stat < 1.0 else "Normal",
        delta_color="inverse",
    )

    st.markdown("---")

    # Section 1: Monthly Bias Progression
    st.subheader("1. Monthly Bias Progression (March 2004 – April 2005)")
    st.markdown(
        """
        Notice the progressive negative shift in Mean Bias Error (MBE). The model systematically 
        overpredicts as the metal oxide sensors lose baseline resistance over time.
        """
    )

    monthly_summary = temporal_error_breakdown(df_eval, "Timestamp", y_all, preds_all)
    
    fig_monthly, ax_m = plt.subplots(figsize=(10, 3.5))
    months = monthly_summary["Month"]
    bias = monthly_summary["Mean_Bias_Error"]
    rmse = monthly_summary["RMSE"]

    x_idx = np.arange(len(months))
    ax_m.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax_m.bar(x_idx - 0.2, bias, width=0.4, label="Mean Bias Error (µg/m³)", color="#c44e52")
    ax_m.bar(x_idx + 0.2, rmse, width=0.4, label="RMSE (µg/m³)", color="#4c72b0")
    ax_m.set_xticks(x_idx)
    ax_m.set_xticklabels(months, rotation=45, ha="right")
    ax_m.set_ylabel("µg/m³")
    ax_m.set_title("Progression of Residual Mean Bias & RMSE Across 14 Months")
    ax_m.legend()
    ax_m.grid(True, alpha=0.3)
    st.pyplot(fig_monthly)

    st.markdown("---")

    # Section 2: CUSUM Drift Monitoring
    st.subheader("2. Cumulative Sum (CUSUM) Quality-Control Drift Monitor")
    st.markdown(
        r"""
        The two-sided CUSUM algorithm tracks cumulative excursions in calibration residuals:
        $$S_t^+ = \max(0, S_{t-1}^+ + e_t - \delta), \quad S_t^- = \max(0, S_{t-1}^- - e_t - \delta)$$
        An alarm is triggered whenever $S_t^+ > H$ or $S_t^- > H$.
        """
    )

    col_ctrl1, col_ctrl2 = st.columns(2)
    with col_ctrl1:
        allowance = st.slider(r"CUSUM Slack Allowance ($\delta$):", 0.1, 2.0, 0.5, 0.1)
    with col_ctrl2:
        threshold = st.slider("CUSUM Alarm Threshold ($H$):", 1.0, 30.0, 10.0, 1.0)

    s_pos, s_neg, alarms_pos, alarms_neg = compute_cusum(residuals_all, allowance=allowance, threshold=threshold)

    fig_cusum, (ax_res, ax_c) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    
    # Subsampled for rendering smoothness
    step_sub = 5
    sub_indices = np.arange(0, len(residuals_all), step_sub)

    ax_res.plot(sub_indices, residuals_all[sub_indices], color="#7f7f7f", alpha=0.5, lw=0.8, label="Residuals $e_t$")
    ax_res.axhline(0, color="black", linestyle="--", lw=1)
    ax_res.set_ylabel("Error (µg/m³)")
    ax_res.set_title("Residual Stream and CUSUM Drift Detection")
    ax_res.legend(loc="upper right", fontsize=8)

    ax_c.plot(sub_indices, s_pos[sub_indices], color="#2ca02c", lw=1.2, label=r"$S_t^+$ (Positive Excursion)")
    ax_c.plot(sub_indices, s_neg[sub_indices], color="#d62728", lw=1.2, label=r"$S_t^-$ (Negative Excursion / Drift)")
    ax_c.axhline(threshold, color="black", linestyle=":", lw=1.5, label=f"Alarm Limit H={threshold}")
    ax_c.set_ylabel("CUSUM Statistic")
    ax_c.set_xlabel("Observation Index (Contiguous Hours)")
    ax_c.legend(loc="upper left", fontsize=8)
    ax_c.grid(True, alpha=0.3)
    st.pyplot(fig_cusum)

    st.info(
        f"**CUSUM Alarms Triggered**: {len(alarms_neg)} negative drift alarms, {len(alarms_pos)} positive excursion alarms. "
        f"Notice persistent upward trips in $S_t^-$ during the final 3 months (Spring 2005) demonstrating sustained sensor drift."
    )

    st.markdown("---")

    # Section 3: Environmental Confounding
    st.subheader("3. Environmental Confounding Diagnostics")
    st.markdown(
        """
        Are residual errors purely caused by sensor degradation, or are they confounded by seasonal 
        changes in Temperature ($T$) and Absolute Humidity ($AH$)?
        """
    )

    col_env1, col_env2 = st.columns(2)
    with col_env1:
        fig_t, ax_t = plt.subplots(figsize=(5, 3.5))
        ax_t.scatter(df_eval["T"], df_eval["Residual"], alpha=0.15, s=8, color="#ff7f0e")
        corr_t = np.corrcoef(df_eval["T"], df_eval["Residual"])[0, 1]
        ax_t.set_xlabel("Temperature (°C)")
        ax_t.set_ylabel("Residual (µg/m³)")
        ax_t.set_title(f"Residuals vs. Temperature (r = {corr_t:+.3f})")
        ax_t.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax_t.grid(True, alpha=0.3)
        st.pyplot(fig_t)

    with col_env2:
        fig_ah, ax_ah = plt.subplots(figsize=(5, 3.5))
        ax_ah.scatter(df_eval["AH"], df_eval["Residual"], alpha=0.15, s=8, color="#17becf")
        corr_ah = np.corrcoef(df_eval["AH"], df_eval["Residual"])[0, 1]
        ax_ah.set_xlabel("Absolute Humidity")
        ax_ah.set_ylabel("Residual (µg/m³)")
        ax_ah.set_title(f"Residuals vs. Absolute Humidity (r = {corr_ah:+.3f})")
        ax_ah.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax_ah.grid(True, alpha=0.3)
        st.pyplot(fig_ah)
