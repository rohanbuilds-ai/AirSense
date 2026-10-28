"""Principal Component Regression (PCR) model for sensor calibration.

Integrates:
- Training-fitted feature standardization.
- Thin SVD PCA dimensionality reduction.
- Numerically stable QR least-squares regression on principal subspaces.
- Projection of principal weights back into physical feature space for interpretation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from airsense.calibration import FeatureStandardizer, LinearCalibrationModel
from airsense.math_core import pca_svd


class PCACalibrationModel:
    """Principal Component Regression (PCR) calibrator.

    Pipeline:
    1. Standardize features: Z = (X - mu_train) / sigma_train.
    2. Dimensionality reduction via SVD: Z = U Sigma V^T, retaining k components.
    3. Principal component projection: T = Z V_k.
    4. Linear least-squares calibration: y = w_0 + T w_pca.
    5. Back-projection: w_feature = D_sigma^{-1} V_k w_pca.

    Parameters
    ----------
    n_components : int
        Number of principal components to retain (1 <= k <= d).
    solver : str, default='qr'
        Least squares solver ('qr', 'svd', or 'sklearn').
    """

    def __init__(self, n_components: int = 4, solver: str = "qr") -> None:
        self.n_components = n_components
        self.solver = solver

        # Fitted attributes
        self.scaler_: FeatureStandardizer | None = None
        self.components_: np.ndarray | None = None  # (d, k)
        self.all_components_: np.ndarray | None = None  # (d, d)
        self.singular_values_: np.ndarray | None = None  # (d,)
        self.eigenvalues_: np.ndarray | None = None  # (d,)
        self.explained_variance_ratio_: np.ndarray | None = None  # (d,)
        self.cumulative_variance_ratio_: np.ndarray | None = None  # (d,)
        self.regressor_: LinearCalibrationModel | None = None
        self.weights_pca_: np.ndarray | None = None  # (k,)
        self.intercept_: float = 0.0
        self.feature_weights_: np.ndarray | None = None  # (d,)
        self.n_features_in_: int | None = None
        self.reconstruction_error_train_: float = 0.0

    def fit(
        self,
        X: np.ndarray | pd.DataFrame,
        y: np.ndarray | pd.Series,
    ) -> PCACalibrationModel:
        """Fit scaler, PCA components, and calibration weights strictly on training data."""
        X_mat = np.asarray(X, dtype=np.float64)
        y_vec = np.asarray(y, dtype=np.float64).ravel()
        n, d = X_mat.shape

        if self.n_components < 1 or self.n_components > d:
            raise ValueError(
                f"n_components={self.n_components} must be in range [1, {d}]."
            )

        self.n_features_in_ = d

        # 1. Fit standardizer on training features
        self.scaler_ = FeatureStandardizer(with_mean=True, with_std=True)
        Z_train = self.scaler_.fit_transform(X_mat)

        # 2. Fit PCA via SVD on standardized training data
        pca_dict = pca_svd(Z_train, n_components=self.n_components)
        self.all_components_ = pca_dict["all_components"]
        self.components_ = pca_dict["components"]  # (d, k)
        self.singular_values_ = pca_dict["singular_values"]
        self.eigenvalues_ = pca_dict["eigenvalues"]
        self.explained_variance_ratio_ = pca_dict["explained_variance_ratio"]
        self.cumulative_variance_ratio_ = pca_dict["cumulative_variance_ratio"]
        self.reconstruction_error_train_ = pca_dict["reconstruction_error"]

        # 3. Project standardized training features to k principal scores: T = Z V_k
        T_train = pca_dict["scores"]  # (n, k)

        # 4. Fit linear calibrator: y = w_0 + T w_pca
        self.regressor_ = LinearCalibrationModel(solver=self.solver, fit_intercept=True)
        self.regressor_.fit(T_train, y_vec)

        self.intercept_ = self.regressor_.intercept_
        self.weights_pca_ = self.regressor_.coef_  # (k,)

        # 5. Back-project weights into original physical feature space for interpretation
        # T = (X - mu) D_sigma^{-1} V_k
        # y_hat = w_0 + (X - mu) D_sigma^{-1} V_k w_pca
        # Let w_feature = D_sigma^{-1} V_k w_pca
        scale = self.scaler_.scale_
        self.feature_weights_ = (self.components_ @ self.weights_pca_) / scale

        return self

    def transform(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Transform input features into principal component coordinates."""
        if self.scaler_ is None or self.components_ is None:
            raise RuntimeError("Model must be fitted before calling transform.")

        Z = self.scaler_.transform(X)
        return Z @ self.components_

    def predict(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict target benzene concentration using PCA-reduced calibration."""
        if self.regressor_ is None:
            raise RuntimeError("Model must be fitted before calling predict.")

        T = self.transform(X)
        return self.regressor_.predict(T)

    def reconstruct(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Reconstruct standardized features from k principal components."""
        T = self.transform(X)
        Z_hat = T @ self.components_.T
        return self.scaler_.inverse_transform(Z_hat)
