"""Chronological partitioning, feature standardization, and linear calibration models.

Enforces zero-data-leakage protocols:
- Chronological partitions with contiguous out-of-time boundaries.
- Preprocessing transformers fitted strictly on training partition only.
- Solvers utilizing custom QR/SVD linear algebra primitives from airsense.math_core.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as SklearnLinearRegression

from airsense.math_core import (
    matrix_rank,
    solve_least_squares_qr,
    solve_least_squares_svd,
)


# -----------------------------------------------------------------------------
# 1. Chronological Partitioning
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ChronologicalSplit:
    """Container for chronologically partitioned datasets and their boundary timestamps."""
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    def summary(self) -> dict[str, Any]:
        """Return a structured summary of partition sizes and temporal boundaries."""
        n_total = len(self.train_df) + len(self.val_df) + len(self.test_df)
        return {
            "total_rows": n_total,
            "train_rows": len(self.train_df),
            "train_pct": round(len(self.train_df) / n_total * 100, 2),
            "train_window": (self.train_start, self.train_end),
            "val_rows": len(self.val_df),
            "val_pct": round(len(self.val_df) / n_total * 100, 2),
            "val_window": (self.val_start, self.val_end),
            "test_rows": len(self.test_df),
            "test_pct": round(len(self.test_df) / n_total * 100, 2),
            "test_window": (self.test_start, self.test_end),
        }

    def __iter__(self):
        """Allow tuple unpacking: train_df, val_df, test_df = split."""
        return iter((self.train_df, self.val_df, self.test_df))


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.20,
    test_ratio: float = 0.20,
    timestamp_col: str = "Timestamp",
) -> ChronologicalSplit:
    """Split time-series dataframe into contiguous, non-overlapping train/val/test partitions.

    Strict guardrails:
    1. Asserts ratios sum to 1.0.
    2. Validates timestamp column existence and monotonic increasing ordering.
    3. Enforces zero temporal overlap: max(t_train) < min(t_val) and max(t_val) < min(t_test).

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing sorted chronological records.
    train_ratio : float, default=0.60
        Fraction of data assigned to training partition.
    val_ratio : float, default=0.20
        Fraction of data assigned to validation partition.
    test_ratio : float, default=0.20
        Fraction of data assigned to out-of-time test partition.
    timestamp_col : str, default='Timestamp'
        Column name holding timestamps.

    Returns
    -------
    ChronologicalSplit
        Dataclass containing the 3 partitioned dataframes and their boundaries.
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError(
            f"Partition ratios must sum to 1.0; got {train_ratio} + {val_ratio} + {test_ratio} = "
            f"{train_ratio + val_ratio + test_ratio}"
        )

    if timestamp_col not in df.columns:
        raise KeyError(f"Timestamp column '{timestamp_col}' not found in dataframe.")

    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    ts_series = df[timestamp_col]
    if not ts_series.is_monotonic_increasing:
        raise ValueError("Dataframe must be strictly sorted by timestamp before splitting.")

    n_total = len(df)
    n_train = int(np.floor(n_total * train_ratio))
    n_val = int(np.floor(n_total * val_ratio))
    n_test = n_total - n_train - n_val

    if n_train == 0 or n_val == 0 or n_test == 0:
        raise ValueError(f"Partition size is zero: train={n_train}, val={n_val}, test={n_test}.")

    train_df = df.iloc[:n_train].copy().reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].copy().reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].copy().reset_index(drop=True)

    # Boundary timestamps
    t_tr_start = train_df[timestamp_col].iloc[0]
    t_tr_end = train_df[timestamp_col].iloc[-1]
    t_va_start = val_df[timestamp_col].iloc[0]
    t_va_end = val_df[timestamp_col].iloc[-1]
    t_te_start = test_df[timestamp_col].iloc[0]
    t_te_end = test_df[timestamp_col].iloc[-1]

    # Verify temporal non-overlap
    if not (t_tr_end < t_va_start < t_va_end < t_te_start):
        raise ValueError(
            f"Temporal overlap detected: train_end={t_tr_end}, val_start={t_va_start}, "
            f"val_end={t_va_end}, test_start={t_te_start}"
        )

    return ChronologicalSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_start=t_tr_start,
        train_end=t_tr_end,
        val_start=t_va_start,
        val_end=t_va_end,
        test_start=t_te_start,
        test_end=t_te_end,
    )


# -----------------------------------------------------------------------------
# 2. Leakage-Free Standardization Transformer
# -----------------------------------------------------------------------------

class FeatureStandardizer:
    """Standardize features by removing the mean and scaling to unit variance.

    Parameters are computed strictly from training data and applied downstream.
    Guards against zero-variance features by thresholding small standard deviations.
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True) -> None:
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.n_features_: int | None = None

    def fit(self, X: np.ndarray | pd.DataFrame) -> FeatureStandardizer:
        """Compute empirical mean and standard deviation from training data."""
        X_mat = np.asarray(X, dtype=np.float64)
        if X_mat.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_mat.shape}.")

        self.n_features_ = X_mat.shape[1]
        if self.with_mean:
            self.mean_ = np.mean(X_mat, axis=0)
        else:
            self.mean_ = np.zeros(self.n_features_, dtype=np.float64)

        if self.with_std:
            # Sample standard deviation (ddof=1) or population (ddof=0)
            stds = np.std(X_mat, axis=0, ddof=0)
            # Guard against zero-division for constant features
            stds[stds < 1e-12] = 1.0
            self.scale_ = stds
        else:
            self.scale_ = np.ones(self.n_features_, dtype=np.float64)

        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Apply stored standardization parameters to input data."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureStandardizer must be fitted on training data before transform.")

        X_mat = np.asarray(X, dtype=np.float64)
        if X_mat.shape[1] != self.n_features_:
            raise ValueError(
                f"Feature dimension mismatch: fitted on {self.n_features_} features, "
                f"received {X_mat.shape[1]}."
            )

        return (X_mat - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Fit to training data, then transform."""
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Transform standardized scores back to original feature scale."""
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("FeatureStandardizer must be fitted before inverse_transform.")
        Z_mat = np.asarray(Z, dtype=np.float64)
        return Z_mat * self.scale_ + self.mean_


# -----------------------------------------------------------------------------
# 3. Linear Least-Squares Calibration Model
# -----------------------------------------------------------------------------

class LinearCalibrationModel:
    """Linear Ordinary Least Squares calibration model with custom QR and SVD solvers.

    Supports:
    - solver='qr': Numerically stable QR decomposition via Modified Gram-Schmidt.
    - solver='svd': SVD pseudoinverse solver robust to rank-deficiency.
    - solver='sklearn': Reference scikit-learn implementation for verification.
    """

    def __init__(self, solver: str = "qr", fit_intercept: bool = True) -> None:
        if solver not in ("qr", "svd", "sklearn"):
            raise ValueError(f"Unknown solver: '{solver}'. Expected 'qr', 'svd', or 'sklearn'.")
        self.solver = solver
        self.fit_intercept = fit_intercept
        self.weights_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.coef_: np.ndarray | None = None
        self.residuals_: np.ndarray | None = None
        self.rss_: float | None = None
        self.n_features_in_: int | None = None

    def fit(self, X: np.ndarray | pd.DataFrame, y: np.ndarray | pd.Series) -> LinearCalibrationModel:
        """Fit linear regression weights w minimizing ||X w - y||_2^2."""
        X_mat = np.asarray(X, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64).ravel()

        if X_mat.ndim != 2:
            raise ValueError(f"Design matrix X must be 2D; got shape {X_mat.shape}.")
        if X_mat.shape[0] != len(y_vec):
            raise ValueError(f"Sample size mismatch: X has {X_mat.shape[0]} rows, y has {len(y_vec)}.")

        self.n_features_in_ = X_mat.shape[1]

        if self.solver == "sklearn":
            reg = SklearnLinearRegression(fit_intercept=self.fit_intercept)
            reg.fit(X_mat, y_vec)
            self.intercept_ = float(reg.intercept_) if self.fit_intercept else 0.0
            self.coef_ = reg.coef_.copy()
            if self.fit_intercept:
                self.weights_ = np.hstack([[self.intercept_], self.coef_])
            else:
                self.weights_ = self.coef_.copy()
            y_pred = reg.predict(X_mat)
            self.residuals_ = y_vec - y_pred
            self.rss_ = float(np.sum(self.residuals_**2))
            return self

        # Custom QR / SVD mathematical engine
        if self.fit_intercept:
            intercept_col = np.ones((X_mat.shape[0], 1), dtype=np.float64)
            X_augmented = np.hstack([intercept_col, X_mat])
        else:
            X_augmented = X_mat

        if self.solver == "qr":
            try:
                w, residuals, rss = solve_least_squares_qr(X_augmented, y_vec)
            except ValueError:
                # Fallback to SVD if collinearity makes R singular
                w, residuals, rss = solve_least_squares_svd(X_augmented, y_vec)
        else:  # svd
            w, residuals, rss = solve_least_squares_svd(X_augmented, y_vec)

        self.weights_ = w
        self.residuals_ = residuals
        self.rss_ = rss

        if self.fit_intercept:
            self.intercept_ = float(w[0])
            self.coef_ = w[1:].copy()
        else:
            self.intercept_ = 0.0
            self.coef_ = w.copy()

        return self

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict target benzene concentration using fitted linear calibration."""
        if self.weights_ is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        X_mat = np.asarray(X, dtype=np.float64)
        if X_mat.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: fitted on {self.n_features_in_} features, "
                f"received {X_mat.shape[1]}."
            )

        if self.fit_intercept:
            return self.intercept_ + X_mat @ self.coef_
        else:
            return X_mat @ self.coef_
