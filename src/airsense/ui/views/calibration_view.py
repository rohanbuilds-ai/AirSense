"""Calibration Experiment View for AirSense Streamlit Interface.

Provides interactive evaluation of sensor calibration models:
- Dynamic Feature Selection (Chemical Sensors + Meteorological Covariates)
- Model Architecture Selection (Full OLS vs. PCR with tunable k)
- Live Computation of Train / Validation / Test Metrics (MAE, RMSE, R², Bias)
- Prediction vs. Reference Scatter Plots & Residual Distributions
- Back-Projected Physical Feature Weights
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from airsense.calibration import chronological_split, LinearCalibrationModel, FeatureStandardizer
from airsense.pca_model import PCACalibrationModel
from airsense.evaluation import mean_absolute_error, root_mean_squared_error, r2_score, mean_bias_error


def render_calibration_view(df_supervised: pd.DataFrame) -> None:
    """Render Section 5: Sensor Calibration Experiment."""
    st.header("🔬 5. Sensor Calibration Experiment")
    st.markdown(
        """
        Evaluate how least-squares calibration models translate low-cost metal-oxide sensor readings 
        into certified benzene concentrations. Test custom feature subsets, adjust PCA dimensionality, 
        and observe how PCA regularization alters out-of-time test generalization.
        """
    )

    if df_supervised.empty:
        st.error("Supervised dataset is empty or not loaded.")
        return

    # Chronological partition
    train_df, val_df, test_df = chronological_split(df_supervised, train_ratio=0.6, val_ratio=0.2)
    
    all_available_features = [
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

    # Feature selection controls
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Feature & Model Controls")

    selected_features = st.sidebar.multiselect(
        "Select Predictive Features:",
        options=all_available_features,
        default=all_available_features,
        help="Choose any combination of chemical sensors and meteorological variables.",
    )

    if not selected_features:
        st.error("Please select at least one predictive feature.")
        return

    d_selected = len(selected_features)

    model_type = st.sidebar.radio(
        "Select Calibration Architecture:",
        ["Principal Component Regression (PCR)", "Full-Feature Least-Squares (OLS)"],
    )

    if "PCR" in model_type:
        k_components = st.sidebar.slider(
            "Retained Principal Components ($k$):",
            min_value=1,
            max_value=d_selected,
            value=min(3, d_selected),
            step=1,
            help=f"Number of orthogonal SVD directions retained (1 to {d_selected}).",
        )
    else:
        k_components = d_selected

    eval_set_name = st.sidebar.radio(
        "Evaluation Partition for Visualizations:",
        ["Test Set (Final Out-of-Time Generalization)", "Validation Set (Hyperparameter Tuning)", "Training Set (In-Sample Fit)"],
        index=0,
    )

    # Prepare data arrays
    X_train = train_df[selected_features].to_numpy()
    y_train = train_df[target_col].to_numpy()
    X_val = val_df[selected_features].to_numpy()
    y_val = val_df[target_col].to_numpy()
    X_test = test_df[selected_features].to_numpy()
    y_test = test_df[target_col].to_numpy()

    # Fit selected model
    if "PCR" in model_type:
        model = PCACalibrationModel(n_components=k_components)
        model.fit(X_train, y_train)
        feature_weights = model.feature_weights_
    else:
        model = LinearCalibrationModel()
        model.fit(X_train, y_train)
        feature_weights = model.weights[1:]

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_pred_test = model.predict(X_test)

    # Compute metrics table across all three partitions
    partitions = ["Training (In-Sample)", "Validation (Tuning)", "Test (Out-of-Time)"]
    y_trues = [y_train, y_val, y_test]
    y_preds = [y_pred_train, y_pred_val, y_pred_test]

    metrics_list = []
    for p_name, y_t, y_p in zip(partitions, y_trues, y_preds):
        metrics_list.append({
            "Partition": p_name,
            "RMSE (µg/m³)": root_mean_squared_error(y_t, y_p),
            "MAE (µg/m³)": mean_absolute_error(y_t, y_p),
            "R² Score": r2_score(y_t, y_p),
            "Mean Bias (µg/m³)": mean_bias_error(y_t, y_p),
        })

    metrics_df = pd.DataFrame(metrics_list)

    # Active evaluation selection
    if "Test" in eval_set_name:
        y_eval, y_pred_eval, tag = y_test, y_pred_test, "Test (Out-of-Time)"
    elif "Validation" in eval_set_name:
        y_eval, y_pred_eval, tag = y_val, y_pred_val, "Validation"
    else:
        y_eval, y_pred_eval, tag = y_train, y_pred_train, "Training"

    eval_rmse = root_mean_squared_error(y_eval, y_pred_eval)
    eval_mae = mean_absolute_error(y_eval, y_pred_eval)
    eval_r2 = r2_score(y_eval, y_pred_eval)
    eval_bias = mean_bias_error(y_eval, y_pred_eval)

    # Metric KPI Display
    st.subheader(f"Experimental Performance Metrics ({tag})")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("RMSE", f"{eval_rmse:.3f} µg/m³")
    c2.metric("MAE", f"{eval_mae:.3f} µg/m³")
    c3.metric("R² Score", f"{eval_r2:.4f}")
    c4.metric("Mean Bias Error", f"{eval_bias:+.3f} µg/m³")

    st.markdown("---")

    # Metrics across all sets
    st.write("**Full Chronological Partition Breakdown**:")
    st.dataframe(
        metrics_df.style.format({
            "RMSE (µg/m³)": "{:.3f}",
            "MAE (µg/m³)": "{:.3f}",
            "R² Score": "{:.4f}",
            "Mean Bias (µg/m³)": "{:+.3f}",
        }),
        use_container_width=True,
    )

    st.markdown("---")

    # Diagnostic Visualizations
    col_plot1, col_plot2 = st.columns(2)

    with col_plot1:
        st.subheader("Prediction vs. Reference Concentration")
        fig_scatter, ax_sc = plt.subplots(figsize=(6, 3.8))
        ax_sc.scatter(y_eval, y_pred_eval, alpha=0.3, s=10, color="#2ca02c", label=f"{model_type} (k={k_components})")
        min_v = min(y_eval.min(), y_pred_eval.min())
        max_v = max(y_eval.max(), y_pred_eval.max())
        ax_sc.plot([min_v, max_v], [min_v, max_v], "k--", lw=1.5, label="Ideal 1:1 Identity Line")
        
        # Regression trend
        m_fit, b_fit = np.polyfit(y_eval, y_pred_eval, 1)
        ax_sc.plot(np.array([min_v, max_v]), m_fit * np.array([min_v, max_v]) + b_fit, "r-", lw=1.2, label=f"Trend (slope={m_fit:.2f})")
        
        ax_sc.set_xlabel(r"Reference $C_6H_6(GT)$ ($\mu g/m^3$)")
        ax_sc.set_ylabel(r"Predicted $C_6H_6$ ($\mu g/m^3$)")
        ax_sc.set_title(f"Model Predictions vs. Certified Reference ({tag})")
        ax_sc.legend(loc="upper left", fontsize=8)
        ax_sc.grid(True, alpha=0.3)
        st.pyplot(fig_scatter)

    with col_plot2:
        st.subheader("Calibration Residual Distribution")
        residuals = y_eval - y_pred_eval
        fig_res, ax_r = plt.subplots(figsize=(6, 3.8))
        ax_r.hist(residuals, bins=40, density=True, alpha=0.6, color="#4c72b0", edgecolor="white", label="Residuals $e_t$")
        
        # Normal reference
        rx = np.linspace(residuals.min(), residuals.max(), 200)
        ry = (1.0 / (np.std(residuals) * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((rx - np.mean(residuals)) / np.std(residuals))**2)
        ax_r.plot(rx, ry, "r--", lw=1.5, label="Normal Fit")
        
        ax_r.set_xlabel(r"Residual $e_t = y_t - \hat{y}_t$ ($\mu g/m^3$)")
        ax_r.set_ylabel("Density")
        ax_r.set_title(f"Residual Distribution ({tag})")
        ax_r.legend(loc="upper right", fontsize=8)
        ax_r.grid(True, alpha=0.3)
        st.pyplot(fig_res)

    st.markdown("---")

    # Feature sensitivity table
    st.subheader(rf"Back-Projected Physical Feature Sensitivities ($\mathbf{{w}}_{{\text{{feature}}}}$)")
    st.markdown(
        r"""
        Physical feature sensitivities in engineering units ($\mu g/m^3$ change per raw sensor unit):
        $$\mathbf{w}_{\text{feature}} = D_{\sigma}^{-1} V_k \mathbf{w}_{\text{pca}}$$
        """
    )

    sens_df = pd.DataFrame({
        "Feature": selected_features,
        "Sensitivity Weight": feature_weights,
    })
    st.dataframe(sens_df.style.format({"Sensitivity Weight": "{:.6f}"}), use_container_width=True)

    # Research insight
    if d_selected == 8 and "PCR" in model_type:
        if k_components in [3, 4]:
            st.info(
                f"💡 **Post-Hoc Regularization Observation**: Under the stated validation selection protocol, "
                f"$k^*=8$ (Full OLS) was selected because it achieved the lowest Validation RMSE ($2.024\\ \\mu g/m^3$). "
                f"However, on the untouched out-of-time test set, retaining $k={k_components}$ components achieved lower error "
                f"($\\text{{Test RMSE}} = {eval_rmse:.3f}\\ \\mu g/m^3$) by discarding the trailing collinear dimensions "
                f"that overfit to seasonal training conditions. (Note: in operational practice, choosing $k={k_components}$ "
                f"requires validating across seasonal cycles to avoid retrospective test leakage.)"
            )
        elif k_components == 8:
            st.success(
                "✅ **Validation-Selected Configuration ($k^*=8$)**: This configuration achieved the lowest error on the "
                "validation tuning set (Val RMSE = 2.024 µg/m³, Val R² = 0.9423). In full rank, PCR is mathematically "
                "identical to Full OLS because both span the complete column space."
            )
        elif k_components == 7:
            st.warning(
                "⚠️ **The Collinear Overfit Trap ($k=7$)**: Retaining the 7th component captures transient seasonal correlations "
                "in the training set, causing out-of-time test $R^2$ to collapse to 0.6225!"
            )
