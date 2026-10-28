"""Statistical evaluation, regression performance metrics, and temporal residual analysis.

Computes:
- Core metrics: MAE, RMSE, R-squared, Mean Bias Error (MBE).
- Residual series diagnostics: e_t = y_t - y_hat_t.
- Durbin-Watson statistic for temporal residual autocorrelation.
- Temporal error breakdowns across chronological periods and calendar months.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. Core Regression Metrics
# -----------------------------------------------------------------------------

def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Absolute Error (MAE): (1/n) * sum |y_i - y_hat_i|."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true has {len(yt)}, y_pred has {len(yp)}.")
    if len(yt) == 0:
        return 0.0
    return float(np.mean(np.abs(yt - yp)))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Root Mean Squared Error (RMSE): sqrt((1/n) * sum (y_i - y_hat_i)^2)."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true has {len(yt)}, y_pred has {len(yp)}.")
    if len(yt) == 0:
        return 0.0
    return float(np.sqrt(np.mean((yt - yp)**2)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Coefficient of Determination (R^2): 1 - SS_res / SS_tot."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    if len(yt) != len(yp):
        raise ValueError(f"Length mismatch: y_true has {len(yt)}, y_pred has {len(yp)}.")
    if len(yt) <= 1:
        return 0.0

    ss_res = np.sum((yt - yp)**2)
    ss_tot = np.sum((yt - np.mean(yt))**2)
    if ss_tot < 1e-12:
        return 0.0  # Constant target
    return float(1.0 - (ss_res / ss_tot))


def mean_bias_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Bias Error (MBE): (1/n) * sum (y_hat_i - y_i).

    Positive indicates systematic overestimation; negative indicates underestimation.
    """
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.mean(yp - yt))


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute a dictionary of standard regression evaluation metrics."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    res = yt - yp

    return {
        "n_samples": len(yt),
        "mae": mean_absolute_error(yt, yp),
        "rmse": root_mean_squared_error(yt, yp),
        "r2": r2_score(yt, yp),
        "mbe": mean_bias_error(yt, yp),
        "max_error": float(np.max(np.abs(res))) if len(yt) > 0 else 0.0,
    }


# -----------------------------------------------------------------------------
# 2. Residual Diagnostics & Temporal Stability Analysis
# -----------------------------------------------------------------------------

def compute_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute residual errors: e = y_true - y_pred."""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    return yt - yp


def durbin_watson_statistic(residuals: np.ndarray) -> float:
    """Compute Durbin-Watson statistic for serial correlation in residuals.

    DW = sum_{t=2}^n (e_t - e_{t-1})^2 / sum_{t=1}^n e_t^2.
    - DW ~ 2 indicates no first-order serial correlation.
    - DW < 2 indicates positive serial correlation.
    - DW > 2 indicates negative serial correlation.
    """
    e = np.asarray(residuals, dtype=np.float64).ravel()
    if len(e) < 2:
        return 2.0
    denom = np.sum(e**2)
    if denom < 1e-12:
        return 2.0
    diff = np.diff(e)
    numer = np.sum(diff**2)
    return float(numer / denom)


def temporal_error_breakdown(
    timestamps: pd.Series | np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    freq: str = "M",
) -> pd.DataFrame:
    """Group predictions by temporal period (e.g. Month 'M' or Quarter 'Q') and compute error metrics.

    Parameters
    ----------
    timestamps : pd.Series | np.ndarray
        Chronological timestamps.
    y_true : np.ndarray
        Ground-truth reference target.
    y_pred : np.ndarray
        Calibrated model predictions.
    freq : str, default='M'
        Period frequency ('M' for monthly, 'Q' for quarterly).

    Returns
    -------
    pd.DataFrame
        Table indexed by Period containing sample counts, RMSE, MAE, R^2, Mean Bias,
        and Durbin-Watson statistic.
    """
    ts = pd.to_datetime(timestamps)
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()

    eval_df = pd.DataFrame({
        "Timestamp": ts,
        "y_true": yt,
        "y_pred": yp,
        "residual": yt - yp,
    })

    eval_df["Period"] = eval_df["Timestamp"].dt.to_period(freq)

    records = []
    for period, group in eval_df.groupby("Period"):
        g_yt = group["y_true"].values
        g_yp = group["y_pred"].values
        g_res = group["residual"].values
        records.append({
            "Period": str(period),
            "Count": len(group),
            "RMSE": round(root_mean_squared_error(g_yt, g_yp), 4),
            "MAE": round(mean_absolute_error(g_yt, g_yp), 4),
            "R2": round(r2_score(g_yt, g_yp), 4),
            "Mean_Bias": round(mean_bias_error(g_yt, g_yp), 4),
            "Durbin_Watson": round(durbin_watson_statistic(g_res), 4),
        })

    return pd.DataFrame(records)
