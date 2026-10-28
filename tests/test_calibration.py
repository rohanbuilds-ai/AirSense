"""Unit and integration tests for calibration, PCA modeling, leakage prevention, and evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airsense.calibration import (
    FeatureStandardizer,
    LinearCalibrationModel,
    chronological_split,
)
from airsense.evaluation import (
    durbin_watson_statistic,
    evaluate_predictions,
    mean_absolute_error,
    mean_bias_error,
    r2_score,
    root_mean_squared_error,
    temporal_error_breakdown,
)
from airsense.pca_model import PCACalibrationModel

PROCESSED_DATA_PATH = Path("data/processed/air_quality_supervised.csv")


# -----------------------------------------------------------------------------
# 1. Tests for Chronological Splitting & Leakage Prevention
# -----------------------------------------------------------------------------

def test_chronological_split_strict_ordering() -> None:
    """Verify that chronological splitting produces non-overlapping contiguous partitions."""
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame({
        "Timestamp": dates,
        "feature1": np.arange(100),
        "target": np.arange(100) * 2.0,
    })

    split = chronological_split(df, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20)

    # 1. Row counts
    assert len(split.train_df) == 60
    assert len(split.val_df) == 20
    assert len(split.test_df) == 20

    # 2. Strict temporal ordering without overlap
    assert split.train_end < split.val_start
    assert split.val_end < split.test_start

    # 3. Exact boundary timestamps
    assert split.train_start == dates[0]
    assert split.train_end == dates[59]
    assert split.val_start == dates[60]
    assert split.val_end == dates[79]
    assert split.test_start == dates[80]
    assert split.test_end == dates[99]

    # 4. Tuple unpacking support
    tr, va, te = split
    assert len(tr) == 60
    assert len(va) == 20
    assert len(te) == 20


def test_chronological_split_invalid_ratios_raises() -> None:
    """Ensure ValueError is raised if partition ratios do not sum to 1.0."""
    df = pd.DataFrame({"Timestamp": pd.date_range("2024-01-01", periods=10, freq="h")})
    with pytest.raises(ValueError, match="must sum to 1.0"):
        chronological_split(df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)


def test_chronological_split_unsorted_raises() -> None:
    """Ensure ValueError is raised if dataframe is not chronologically sorted."""
    dates = pd.date_range("2024-01-01", periods=10, freq="h").tolist()
    dates.reverse()  # Descending order
    df = pd.DataFrame({"Timestamp": dates})
    with pytest.raises(ValueError, match="strictly sorted"):
        chronological_split(df)


def test_feature_standardizer_zero_leakage() -> None:
    """Verify that standardizer parameters are frozen to training statistics."""
    np.random.seed(42)
    # Training distribution: mean=10.0, std=2.0
    X_train = np.random.normal(loc=10.0, scale=2.0, size=(100, 3))
    # Test distribution: shifted mean=20.0, std=5.0
    X_test = np.random.normal(loc=20.0, scale=5.0, size=(50, 3))

    scaler = FeatureStandardizer()
    scaler.fit(X_train)

    train_mean_frozen = scaler.mean_.copy()
    train_scale_frozen = scaler.scale_.copy()

    # Transforming test data must NOT alter scaler parameters
    Z_test = scaler.transform(X_test)

    assert np.allclose(scaler.mean_, train_mean_frozen)
    assert np.allclose(scaler.scale_, train_scale_frozen)

    # Test standardized mean must NOT be zero because test distribution is shifted
    test_z_mean = np.mean(Z_test, axis=0)
    assert not np.allclose(test_z_mean, 0.0, atol=0.5)


# -----------------------------------------------------------------------------
# 2. Tests for Linear Calibration Model
# -----------------------------------------------------------------------------

def test_linear_calibration_qr_matches_sklearn() -> None:
    """Verify custom QR linear calibration matches scikit-learn LinearRegression."""
    np.random.seed(123)
    X = np.random.randn(80, 5)
    true_w = np.array([3.0, -1.5, 2.0, 0.5, -0.8])
    true_b = 4.2
    y = true_b + X @ true_w + 0.1 * np.random.randn(80)

    model_qr = LinearCalibrationModel(solver="qr", fit_intercept=True)
    model_qr.fit(X, y)

    model_sk = LinearCalibrationModel(solver="sklearn", fit_intercept=True)
    model_sk.fit(X, y)

    assert model_qr.intercept_ == pytest.approx(model_sk.intercept_, abs=1e-10)
    assert np.allclose(model_qr.coef_, model_sk.coef_, atol=1e-10)

    # Predictions match
    X_test = np.random.randn(20, 5)
    pred_qr = model_qr.predict(X_test)
    pred_sk = model_sk.predict(X_test)
    assert np.allclose(pred_qr, pred_sk, atol=1e-10)


# -----------------------------------------------------------------------------
# 3. Tests for PCA Calibration Model (PCR)
# -----------------------------------------------------------------------------

def test_pca_calibration_model_pipeline() -> None:
    """Test full PCR pipeline: fit, transform, predict, and back-projected weights."""
    np.random.seed(42)
    n, d = 120, 6
    X = np.random.randn(n, d)
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * np.random.randn(n)

    pcr = PCACalibrationModel(n_components=3, solver="qr")
    pcr.fit(X, y)

    assert pcr.components_.shape == (d, 3)
    assert len(pcr.weights_pca_) == 3
    assert len(pcr.feature_weights_) == d

    # Predict test samples
    X_new = np.random.randn(15, d)
    preds = pcr.predict(X_new)
    assert len(preds) == 15

    # Check that predictions via principal scores match predictions via feature weights
    # y_hat = w_0 + (X_new - mu) @ w_feature
    mu = pcr.scaler_.mean_
    preds_via_features = pcr.intercept_ + (X_new - mu) @ pcr.feature_weights_
    assert np.allclose(preds, preds_via_features, atol=1e-10)


def test_pca_calibration_invalid_components_raises() -> None:
    """Ensure ValueError is raised if n_components is out of valid range [1, d]."""
    X = np.random.randn(20, 4)
    y = np.random.randn(20)

    with pytest.raises(ValueError, match="must be in range"):
        PCACalibrationModel(n_components=0).fit(X, y)

    with pytest.raises(ValueError, match="must be in range"):
        PCACalibrationModel(n_components=5).fit(X, y)


# -----------------------------------------------------------------------------
# 4. Tests for Evaluation Metrics
# -----------------------------------------------------------------------------

def test_evaluation_metrics_exact() -> None:
    """Test MAE, RMSE, R2, and MBE against deterministic vectors."""
    y_true = np.array([3.0, -0.5, 2.0, 7.0])
    y_pred = np.array([2.5, 0.0, 2.0, 8.0])

    # Errors: [-0.5, +0.5, 0.0, +1.0]
    # Abs: [0.5, 0.5, 0.0, 1.0] -> mean = 2.0 / 4 = 0.5
    # Sq: [0.25, 0.25, 0.0, 1.0] -> mean = 1.5 / 4 = 0.375 -> sqrt = 0.612372
    # MBE: (+0.5 + 0.5 + 0.0 + 1.0) / 4 = +0.25
    assert mean_absolute_error(y_true, y_pred) == pytest.approx(0.5)
    assert root_mean_squared_error(y_true, y_pred) == pytest.approx(np.sqrt(0.375))
    assert mean_bias_error(y_true, y_pred) == pytest.approx(0.25)

    # Perfect prediction test
    assert r2_score(y_true, y_true) == pytest.approx(1.0)
    assert root_mean_squared_error(y_true, y_true) == pytest.approx(0.0)


def test_durbin_watson_statistic() -> None:
    """Test Durbin-Watson statistic on uncorrelated vs positively correlated noise."""
    np.random.seed(42)
    # Uncorrelated i.i.d. noise -> DW ~ 2.0
    iid_noise = np.random.randn(500)
    dw_iid = durbin_watson_statistic(iid_noise)
    assert 1.7 < dw_iid < 2.3

    # Strong positive serial correlation: e_t = 0.9 * e_{t-1} + eta_t -> DW << 2
    ser_noise = np.zeros(500)
    for t in range(1, 500):
        ser_noise[t] = 0.9 * ser_noise[t - 1] + np.random.randn()
    dw_ser = durbin_watson_statistic(ser_noise)
    assert dw_ser < 0.6


# -----------------------------------------------------------------------------
# 5. Integration Test: Real AirSense Dataset Pipeline
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not PROCESSED_DATA_PATH.is_file(), reason="Processed dataset not available")
def test_real_airsense_calibration_experiment() -> None:
    """End-to-end integration test verifying chronological calibration on real data."""
    df = pd.read_csv(PROCESSED_DATA_PATH)
    sensor_cols = [
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

    # 1. Chronological split
    split = chronological_split(df, train_ratio=0.60, val_ratio=0.20, test_ratio=0.20)

    X_train = split.train_df[sensor_cols]
    y_train = split.train_df[target_col]
    X_val = split.val_df[sensor_cols]
    y_val = split.val_df[target_col]
    X_test = split.test_df[sensor_cols]
    y_test = split.test_df[target_col]

    # 2. Fit Full OLS
    ols = LinearCalibrationModel(solver="qr", fit_intercept=True)
    ols.fit(X_train, y_train)

    val_preds_ols = ols.predict(X_val)
    val_metrics_ols = evaluate_predictions(y_val.values, val_preds_ols)
    assert val_metrics_ols["r2"] > 0.85

    # 3. Fit PCR with k=4
    pcr = PCACalibrationModel(n_components=4, solver="qr")
    pcr.fit(X_train, y_train)

    val_preds_pcr = pcr.predict(X_val)
    val_metrics_pcr = evaluate_predictions(y_val.values, val_preds_pcr)
    assert val_metrics_pcr["r2"] > 0.80

    # 4. Temporal breakdown table
    monthly_tbl = temporal_error_breakdown(
        split.test_df["Timestamp"], y_test.values, pcr.predict(X_test), freq="M"
    )
    assert len(monthly_tbl) > 0
    assert "RMSE" in monthly_tbl.columns
    assert "Durbin_Watson" in monthly_tbl.columns
