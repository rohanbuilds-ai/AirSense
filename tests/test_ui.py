"""Unit tests for the AirSense Streamlit UI components."""

import numpy as np
import pandas as pd
import pytest
from airsense.ui.app import load_datasets
from airsense.ui.views.drift_view import compute_cusum


def test_ui_load_datasets():
    """Verify that cached data loading returns valid dataframes."""
    df_supervised, df_cleaned, df_results = load_datasets()
    assert isinstance(df_supervised, pd.DataFrame)
    assert isinstance(df_cleaned, pd.DataFrame)
    assert isinstance(df_results, pd.DataFrame)
    assert len(df_supervised) > 0
    assert len(df_results) == 8  # k=1..8


def test_compute_cusum_logic():
    """Verify CUSUM drift detection logic."""
    np.random.seed(42)
    # 50 in-control samples followed by 50 drifted samples
    res = np.concatenate([np.random.normal(0, 0.2, 50), np.random.normal(2.0, 0.2, 50)])
    s_pos, s_neg, alarms_pos, alarms_neg = compute_cusum(res, allowance=0.5, threshold=3.0)

    assert len(s_pos) == 100
    assert len(s_neg) == 100
    # Positive excursion should trip alarms after step 50
    assert len(alarms_pos) > 0
    assert alarms_pos[0] >= 50
