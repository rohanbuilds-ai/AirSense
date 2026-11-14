"""Tests for AirSense FastAPI Gateway Adapter."""

import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_api_health_endpoint():
    """Verify health endpoint reports dataset availability and row counts."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["dataset_loaded"] is True
    assert data["supervised_rows"] == 8991
    assert data["cleaned_rows"] == 9357


def test_api_dataset_summary_endpoint():
    """Verify dataset summary returns provenance and 14 non-date columns."""
    response = client.get("/api/v1/dataset/summary")
    assert response.status_code == 200
    data = response.json()
    assert "provenance" in data
    assert data["counts"]["supervised_usable_rows"] == 8991
    assert data["counts"]["sensor_features_count"] == 8
    assert len(data["columns"]) == 14


def test_api_overview_metrics_endpoint():
    """Verify Overview metrics distinguish validation selection from test results."""
    response = client.get("/api/v1/overview/metrics")
    assert response.status_code == 200
    data = response.json()
    assert data["validation_selected_model"] == "Full OLS / PCR (k*=8)"
    assert abs(data["validation_selected_val_rmse"] - 2.0242) < 1e-3
    assert abs(data["validation_selected_test_rmse"] - 2.8553) < 1e-3
    assert data["post_hoc_best_test_model"] == "Compact PCR (k=3)"
    assert abs(data["post_hoc_test_rmse"] - 2.4146) < 1e-3
    assert abs(data["durbin_watson"] - 0.2062) < 1e-3


def test_api_overview_charts_endpoint():
    """Verify Overview charts return model comparison, PCA variance, and scatter samples."""
    response = client.get("/api/v1/overview/charts")
    assert response.status_code == 200
    data = response.json()
    assert len(data["model_comparison"]) == 8
    assert len(data["pca_variance"]) == 8
    assert len(data["scatter_sample"]) > 100
    assert len(data["timeline_sample"]) > 50
    assert len(data["residuals_distribution"]) == 25


def test_api_dataset_audit_endpoint():
    """Verify dataset audit returns missingness, target stats, and feature correlations."""
    response = client.get("/api/v1/dataset/audit")
    assert response.status_code == 200
    data = response.json()
    assert data["target_distribution"]["count"] == 8991
    assert abs(data["target_distribution"]["mean"] - 10.083) < 1e-2
    assert len(data["feature_correlations"]) == 8
    # PT08.S2(NMHC) has highest correlation ~0.982
    assert data["feature_correlations"][0]["feature"] == "PT08.S2(NMHC)"
    assert abs(data["feature_correlations"][0]["pearson_r"] - 0.9820) < 1e-2


def test_api_paginated_rows_endpoint():
    """Verify paginated dataset rows return expected structure."""
    response = client.get("/api/v1/dataset/rows?limit=25&offset=50")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 9357
    assert data["limit"] == 25
    assert data["offset"] == 50
    assert len(data["rows"]) == 25


def test_api_calibration_benchmark_matrix():
    """Verify benchmark matrix endpoint returns 8 model configurations."""
    response = client.get("/api/v1/calibration/benchmark-matrix")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) == 8


def test_api_calibration_run_ols_baseline():
    """Verify interactive calibration endpoint with baseline 8-feature OLS."""
    payload = {
        "model_type": "OLS",
        "selected_features": [
            "PT08.S1(CO)",
            "PT08.S2(NMHC)",
            "PT08.S3(NOx)",
            "PT08.S4(NO2)",
            "PT08.S5(O3)",
            "T",
            "RH",
            "AH",
        ],
    }
    response = client.post("/api/v1/calibration/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "OLS"
    assert data["feature_count"] == 8
    assert abs(data["val_metrics"]["rmse"] - 2.0242) < 1e-3
    assert abs(data["val_metrics"]["r2"] - 0.9423) < 1e-3
    assert len(data["feature_sensitivities"]) == 8
    assert len(data["scatter_sample"]) > 100
    assert len(data["timeline_sample"]) > 100
    assert len(data["residuals_distribution"]) == 20
    assert data["pca_variance"] is None
    assert "FROZEN" in data["split_info"]["test_partition"]


def test_api_calibration_run_pcr_k3():
    """Verify interactive calibration endpoint with PCR (k=3)."""
    payload = {
        "model_type": "PCR",
        "selected_features": [
            "PT08.S1(CO)",
            "PT08.S2(NMHC)",
            "PT08.S3(NOx)",
            "PT08.S4(NO2)",
            "PT08.S5(O3)",
            "T",
            "RH",
            "AH",
        ],
        "k_components": 3,
    }
    response = client.post("/api/v1/calibration/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "PCR"
    assert data["k_components"] == 3
    assert abs(data["val_metrics"]["rmse"] - 2.8258) < 1e-3
    assert abs(data["val_metrics"]["r2"] - 0.8876) < 1e-3
    assert len(data["pca_variance"]) == 8
    assert abs(data["pca_variance"][2]["cum_variance_pct"] - 94.04) < 1e-1


def test_api_calibration_run_feature_subset():
    """Verify interactive calibration endpoint with custom 3-feature subset."""
    payload = {
        "model_type": "PCR",
        "selected_features": ["PT08.S1(CO)", "PT08.S2(NMHC)", "T"],
        "k_components": 2,
    }
    response = client.post("/api/v1/calibration/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["feature_count"] == 3
    assert data["k_components"] == 2
    assert len(data["feature_sensitivities"]) == 3
    assert any("fewer than 4 features" in w for w in data["warnings"])


def test_api_calibration_run_invalid_components():
    """Verify 400 error when k_components exceeds selected feature count."""
    payload = {
        "model_type": "PCR",
        "selected_features": ["PT08.S1(CO)", "PT08.S2(NMHC)", "T"],
        "k_components": 4,  # only 3 features selected!
    }
    response = client.post("/api/v1/calibration/run", json=payload)
    assert response.status_code == 400
    assert "must be between 1 and 3" in response.json()["detail"]


def test_api_calibration_run_empty_features():
    """Verify 400 error when feature list is empty."""
    payload = {
        "model_type": "OLS",
        "selected_features": [],
    }
    response = client.post("/api/v1/calibration/run", json=payload)
    assert response.status_code == 400
    assert "At least one predictor feature must be selected" in response.json()["detail"]


def test_api_calibration_run_invalid_model_type():
    """Verify 400 error when unsupported model architecture is requested."""
    payload = {
        "model_type": "RANDOM_FOREST",
        "selected_features": ["PT08.S1(CO)"],
    }
    response = client.post("/api/v1/calibration/run", json=payload)
    assert response.status_code == 400
    assert "Unsupported model_type" in response.json()["detail"]


def test_api_calibration_run_leakage_rejection():
    """Verify 400 error when reference analyzer GT column is attempted as input."""
    payload = {
        "model_type": "OLS",
        "selected_features": ["C6H6(GT)", "PT08.S1(CO)"],
    }
    response = client.post("/api/v1/calibration/run", json=payload)
    assert response.status_code == 400
    assert "not a valid predictor" in response.json()["detail"]


# ==============================================================================
# Linear Algebra Lab API Tests
# ==============================================================================

def test_api_math_extract_matrix():
    """Verify matrix extraction from real training partition."""
    payload = {
        "features": ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)"],
        "n_rows": 6,
        "standardize": True,
        "include_intercept": True,
    }
    response = client.post("/api/v1/math/extract-matrix", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["dimensions"] == [6, 4]  # 3 features + 1 intercept
    assert data["col_labels"] == ["Intercept", "PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)"]
    assert len(data["matrix"]) == 6
    assert len(data["target_vector"]) == 6
    assert data["is_standardized"] is True
    # Column 0 is intercept (all 1.0)
    for row in data["matrix"]:
        assert abs(row[0] - 1.0) < 1e-4


def test_api_math_gaussian_elimination_square():
    """Verify step-by-step Gaussian Elimination and LU factorization on a square invertible system."""
    A = [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]]
    payload = {"matrix": A, "compute_lu": True}
    response = client.post("/api/v1/math/gaussian-elimination", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["steps"]) > 0
    assert data["rank"] == 3
    assert data["pivot_columns"] == [0, 1, 2]
    # RREF of invertible matrix is 3x3 identity
    for i in range(3):
        for j in range(3):
            expected = 1.0 if i == j else 0.0
            assert abs(data["final_rref"][i][j] - expected) < 1e-4
    # LU checks
    assert data["lu"] is not None
    assert data["lu"]["is_valid"] is True
    assert data["lu"]["residual_error"] < 1e-8
    assert data["determinant"] is not None


def test_api_math_gaussian_elimination_rectangular():
    """Verify Gaussian Elimination on a tall rectangular matrix."""
    A = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]
    payload = {"matrix": A, "compute_lu": True}
    response = client.post("/api/v1/math/gaussian-elimination", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["rank"] == 2
    assert data["lu"] is None  # LU factorize only applies to square matrices


def test_api_math_subspaces_rank_nullity():
    """Verify column space, null space, and Rank-Nullity Theorem."""
    # 3x3 matrix with rank 2 (row 3 = row 1 + row 2)
    A = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [5.0, 7.0, 9.0]]
    payload = {"matrix": A}
    response = client.post("/api/v1/math/subspaces", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["m_rows"] == 3
    assert data["n_cols"] == 3
    assert data["rank"] == 2
    assert data["nullity"] == 1
    assert data["rank_nullity_satisfied"] is True
    assert data["max_null_residual"] < 1e-10
    assert len(data["column_space_basis"]) == 3
    assert len(data["column_space_basis"][0]) == 2  # 2 basis vectors


def test_api_math_gram_schmidt_comparison():
    """Verify Gram-Schmidt orthogonalization and MGS stability superiority."""
    # Ill-conditioned matrix where CGS loses orthogonality
    eps = 1e-4
    A = [
        [1.0, 1.0, 1.0],
        [eps, 0.0, 0.0],
        [0.0, eps, 0.0],
        [0.0, 0.0, eps],
    ]
    payload = {"matrix": A}
    response = client.post("/api/v1/math/gram-schmidt", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["rank"] == 3
    assert data["loss_mgs"] < 1e-6
    # MGS has lower or equal loss than CGS
    assert data["loss_mgs"] <= data["loss_cgs"]
    assert len(data["dot_product_mgs"]) == 3


def test_api_math_least_squares_projection():
    """Verify QR least-squares, orthogonality of residuals, and condition number squaring."""
    X = [[1.0, 2.0], [2.0, 3.0], [3.0, 5.0], [4.0, 7.0]]
    y = [5.0, 7.0, 11.0, 15.0]
    payload = {"matrix": X, "target_vector": y}
    response = client.post("/api/v1/math/least-squares", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["weights"]) == 2
    assert len(data["projected_target"]) == 4
    assert len(data["residual_vector"]) == 4
    # Orthogonality residual X^T r ~ 0
    assert data["orthogonality_residual"] < 1e-8
    # Condition number squaring: cond(X^T X) ~ cond(X)^2
    assert abs(data["cond_XtX"] - (data["cond_X"] ** 2)) < 1.0
    assert data["cond_ratio"] > 1.0


def test_api_math_spectral_analysis():
    """Verify covariance computation, PSD property, and Jacobi eigenvalue decomposition."""
    payload = {
        "features": ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "T"],
        "use_training_set": True,
    }
    response = client.post("/api/v1/math/spectral", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_symmetric"] is True
    assert data["symmetry_residual"] < 1e-10
    assert data["is_psd"] is True
    assert len(data["eigenvalues"]) == 4
    # Eigenvalues should be sorted descending
    for i in range(len(data["eigenvalues"]) - 1):
        assert data["eigenvalues"][i] >= data["eigenvalues"][i + 1]
    # Residual ||M v - lambda v|| must be very small
    assert data["max_eigenvector_residual"] < 1e-6
    # Cumulative variance should reach 1.0
    assert abs(data["cumulative_variance_ratio"][-1] - 1.0) < 1e-3


# ==============================================================================
# PCA Explorer API Tests
# ==============================================================================

def test_api_pca_analyze_default():
    """Verify standard 8-feature PCA decomposition and Eckart-Young invariant."""
    payload = {
        "features": ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"],
        "standardize": True,
        "n_components": 3,
        "partition": "train",
    }
    response = client.post("/api/v1/pca/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_components"] == 8
    assert data["n_components"] == 3
    assert len(data["variance_table"]) == 8
    # Top 3 components capture approx 94% of variance
    assert abs(data["retained_variance_pct"] - 94.04) < 1.0
    assert abs(data["discarded_variance_pct"] - (100.0 - data["retained_variance_pct"])) < 1e-2
    # Eckart-Young invariant on training partition: empirical == theoretical within 1e-8
    assert abs(data["reconstruction_error_frobenius"] - data["theoretical_error_eckart_young"]) < 1e-8
    # Loadings matrix dimensions
    assert len(data["loadings_matrix"]) == 8
    assert len(data["loadings_matrix"][0]) == 8
    # Scores sample length
    assert len(data["scores_sample"]) == 250
    # Time series sample
    assert len(data["time_series_reconstruction"]) == 48
    # Feature reconstruction errors
    assert len(data["feature_reconstruction"]) == 8


def test_api_pca_analyze_unstandardized():
    """Verify Covariance PCA where unscaled resistance sensors dominate PC1."""
    payload = {
        "features": ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)", "PT08.S5(O3)", "T", "RH", "AH"],
        "standardize": False,
        "n_components": 3,
        "partition": "train",
    }
    response = client.post("/api/v1/pca/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    # In unstandardized PCA, PC1 captures over 85% of total variance due to scale differences
    pc1_var = data["variance_table"][0]["variance_ratio_pct"]
    assert pc1_var > 85.0


def test_api_pca_analyze_validation_partition():
    """Verify out-of-time validation partition transformation with training-fitted scaler."""
    payload = {
        "features": ["PT08.S1(CO)", "PT08.S2(NMHC)", "T"],
        "standardize": True,
        "n_components": 2,
        "partition": "val",
    }
    response = client.post("/api/v1/pca/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["evaluated_partition_rows"] == 1798
    assert data["total_training_rows"] == 5394
    assert len(data["variance_table"]) == 3


def test_api_pca_analyze_component_pair_selection():
    """Verify arbitrary component pair selection for score projection scatter."""
    payload = {
        "features": ["PT08.S1(CO)", "PT08.S2(NMHC)", "PT08.S3(NOx)", "PT08.S4(NO2)"],
        "standardize": True,
        "n_components": 4,
        "score_pc_x": 2,
        "score_pc_y": 3,
    }
    response = client.post("/api/v1/pca/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["score_pc_x_label"] == "PC2"
    assert data["score_pc_y_label"] == "PC3"
    assert len(data["feature_contributions_pc_x"]) == 4
    assert len(data["feature_contributions_pc_y"]) == 4


# ==============================================================================
# Drift Analysis & Sensor Monitoring API Tests
# ==============================================================================

def test_api_drift_analyze_default_ols():
    """Verify default OLS drift analysis, error inflation, and Durbin-Watson diagnostic."""
    payload = {
        "model_type": "OLS",
        "reference_partition": "train",
        "monitoring_partition": "test",
        "selected_feature": "PT08.S1(CO)",
        "rolling_window_hours": 72,
    }
    response = client.post("/api/v1/drift/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "OLS"
    assert data["reference_rows"] == 5394
    assert data["monitoring_rows"] == 1799
    # Reference RMSE is ~0.955, Monitoring RMSE is ~2.855 (inflation ~199%)
    assert abs(data["overall_metrics"]["ref_rmse"] - 0.9547) < 1e-3
    assert abs(data["overall_metrics"]["mon_rmse"] - 2.8553) < 1e-3
    assert data["overall_metrics"]["rmse_inflation_pct"] > 150.0
    # Autocorrelation
    assert data["durbin_watson_full"] < 0.5
    # Timeline points
    assert len(data["timeline_series"]) == 250
    # 14 monthly progression records
    assert len(data["monthly_progression"]) == 14
    # Explainable findings
    assert len(data["explainable_findings"]) >= 4


def test_api_drift_analyze_pcr3():
    """Verify Compact PCR (k=3) drift analysis with superior out-of-time stability."""
    payload = {
        "model_type": "PCR",
        "k_components": 3,
        "reference_partition": "train",
        "monitoring_partition": "test",
    }
    response = client.post("/api/v1/drift/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "PCR"
    assert data["k_components"] == 3
    # PCR (k=3) test RMSE is ~2.415 (better than OLS 2.855)
    assert abs(data["overall_metrics"]["mon_rmse"] - 2.4146) < 1e-3
    assert data["overall_metrics"]["mon_rmse"] < 2.855


def test_api_drift_analyze_feature_distribution_shift():
    """Verify severe distribution drift detection in PT08.S4(NO2) driven by seasonal weather."""
    payload = {
        "selected_feature": "PT08.S4(NO2)",
        "reference_partition": "train",
        "monitoring_partition": "test",
    }
    response = client.post("/api/v1/drift/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    detail = data["selected_feature_detail"]
    assert detail["feature_name"] == "PT08.S4(NO2)"
    # Severe negative shift: Cohen's d < -1.5, KS statistic > 0.5
    assert detail["cohens_d"] < -1.5
    assert detail["ks_statistic"] > 0.5
    assert len(detail["distribution_bins"]) == 20
    # Check that drift summary classifies PT08.S4 as Severe Drift
    s4_summary = next(item for item in data["feature_drift_summary"] if item["feature"] == "PT08.S4(NO2)")
    assert s4_summary["drift_status"] == "Severe Drift"


def test_api_drift_analyze_custom_rolling_window():
    """Verify custom rolling window hours and validation partition monitoring."""
    payload = {
        "model_type": "OLS",
        "reference_partition": "train",
        "monitoring_partition": "val",
        "rolling_window_hours": 168,
        "timeline_subsample": 100,
    }
    response = client.post("/api/v1/drift/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["monitoring_rows"] == 1798
    assert len(data["timeline_series"]) == 100


def test_api_dataset_matrix_preview_default():
    """Verify matrix preview returns real mathematical dimensions, partitions, and standardized values."""
    response = client.get("/api/v1/dataset/matrix-preview?offset=0&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total_usable_rows"] == 8991
    assert data["feature_count"] == 8
    assert data["target_count"] == 1
    assert "8991 x 8" in data["matrix_notation"]
    assert "8991 x 1" in data["target_notation"]
    assert len(data["features"]) == 8
    assert len(data["partitions"]) == 3
    # Check exact partition counts
    part_map = {p["name"]: p["row_count"] for p in data["partitions"]}
    assert part_map["train"] == 5394
    assert part_map["val"] == 1798
    assert part_map["test"] == 1799
    # Check rows
    assert len(data["rows"]) == 10
    first_row = data["rows"][0]
    assert len(first_row["raw_x"]) == 8
    assert len(first_row["standardized_x"]) == 8
    assert isinstance(first_row["raw_y"], float)


def test_api_dataset_matrix_preview_partition_filter():
    """Verify matrix preview filtering by held-out test partition."""
    response = client.get("/api/v1/dataset/matrix-preview?partition=test&limit=5")
    assert response.status_code == 200
    data = response.json()
    assert data["total_rows"] == 1799
    assert len(data["rows"]) == 5
    for r in data["rows"]:
        assert r["partition"] == "test"





