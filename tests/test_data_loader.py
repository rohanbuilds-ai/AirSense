"""Unit tests for the AirSense data ingestion and validation pipeline."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from airsense.data_loader import (
    ENVIRONMENTAL_FEATURES,
    EXCLUDED_REFERENCE_COLUMNS,
    PREDICTOR_FEATURES,
    RAW_COLUMNS,
    SENSOR_FEATURES,
    SENTINEL_MISSING_VALUE,
    TARGET_COLUMN,
    clean_air_quality_data,
    detect_csv_dialect,
    get_feature_target_split,
    load_dataset,
    load_raw_air_quality,
    validate_schema,
)

RAW_DATA_PATH = Path("data/raw/AirQualityUCI.csv")


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture
def sample_raw_csv_content() -> str:
    """Generate minimal CSV content mimicking the UCI raw format."""
    return (
        "Date;Time;CO(GT);PT08.S1(CO);NMHC(GT);C6H6(GT);PT08.S2(NMHC);NOx(GT);PT08.S3(NOx);NO2(GT);PT08.S4(NO2);PT08.S5(O3);T;RH;AH;;\n"
        "10/03/2004;18.00.00;2,6;1360;150;11,9;1046;166;1056;113;1692;1268;13,6;48,9;0,7578;;\n"
        "10/03/2004;19.00.00;2;1292;112;9,4;955;103;1174;92;1559;972;13,3;47,7;0,7255;;\n"
        "10/03/2004;20.00.00;-200;-200;-200;-200;-200;-200;-200;-200;-200;-200;-200;-200;-200;;\n"
        ";;;;;;;;;;;;;;;;\n"
    )


@pytest.fixture
def sample_csv_file(tmp_path: Path, sample_raw_csv_content: str) -> Path:
    """Create a temporary CSV file with raw UCI format."""
    csv_file = tmp_path / "mock_air_quality.csv"
    csv_file.write_text(sample_raw_csv_content, encoding="latin-1")
    return csv_file


# -----------------------------------------------------------------------------
# Test Cases
# -----------------------------------------------------------------------------

def test_detect_csv_dialect_synthetic(sample_csv_file: Path) -> None:
    """Test dialect detection on mock European CSV."""
    dialect = detect_csv_dialect(sample_csv_file)
    assert dialect["delimiter"] == ";"
    assert dialect["decimal"] == ","
    assert dialect["has_trailing_delimiters"] is True


def test_detect_csv_dialect_nonexistent_file() -> None:
    """Ensure FileNotFoundError is raised when file does not exist."""
    with pytest.raises(FileNotFoundError):
        detect_csv_dialect("non_existent_file.csv")


def test_load_raw_air_quality_strips_artifacts(sample_csv_file: Path) -> None:
    """Verify that trailing empty columns and empty rows are pruned."""
    raw_df = load_raw_air_quality(sample_csv_file)
    # The mock has 3 data rows, 1 blank trailing row, and 2 empty trailing columns
    assert raw_df.shape == (3, 15)
    assert not any(col.startswith("Unnamed") for col in raw_df.columns)
    assert "Date" in raw_df.columns
    assert "Time" in raw_df.columns


def test_clean_air_quality_sentinel_replacement(sample_csv_file: Path) -> None:
    """Verify that -200 sentinel is converted to NaN across numeric columns."""
    raw_df = load_raw_air_quality(sample_csv_file)
    cleaned_df = clean_air_quality_data(raw_df, drop_missing_target=False)

    # Row index 2 in mock has -200 across all columns
    row_2 = cleaned_df.iloc[2]
    assert np.isnan(row_2["C6H6(GT)"])
    assert np.isnan(row_2["PT08.S1(CO)"])
    assert np.isnan(row_2["T"])

    # Rows 0 and 1 should have valid positive values
    assert cleaned_df.iloc[0]["C6H6(GT)"] == pytest.approx(11.9)
    assert cleaned_df.iloc[0]["T"] == pytest.approx(13.6)


def test_clean_air_quality_timestamp_monotonicity(sample_csv_file: Path) -> None:
    """Verify timestamp parsing and monotonic ordering."""
    raw_df = load_raw_air_quality(sample_csv_file)
    cleaned_df = clean_air_quality_data(raw_df, drop_missing_target=False)

    assert "Timestamp" in cleaned_df.columns
    assert cleaned_df["Timestamp"].is_monotonic_increasing
    assert cleaned_df["Timestamp"].duplicated().sum() == 0
    assert cleaned_df["Timestamp"].iloc[0] == pd.Timestamp("2004-03-10 18:00:00")
    assert cleaned_df["Timestamp"].iloc[1] == pd.Timestamp("2004-03-10 19:00:00")


def test_drop_missing_target_flag(sample_csv_file: Path) -> None:
    """Verify drop_missing_target filters out target NaN rows."""
    raw_df = load_raw_air_quality(sample_csv_file)
    supervised_df = clean_air_quality_data(raw_df, drop_missing_target=True)

    # Row 2 had -200 target, so only 2 rows should remain
    assert len(supervised_df) == 2
    assert supervised_df[TARGET_COLUMN].isna().sum() == 0


def test_feature_target_split_safety(sample_csv_file: Path) -> None:
    """Verify that feature extraction strictly forbids reference columns."""
    raw_df = load_raw_air_quality(sample_csv_file)
    supervised_df = clean_air_quality_data(raw_df, drop_missing_target=True)

    X, y, ts = get_feature_target_split(supervised_df, include_environmental=True)

    assert X.shape == (2, 8)
    assert len(y) == 2
    assert len(ts) == 2

    # Verify no (GT) columns exist in X
    for col in X.columns:
        assert not col.endswith("(GT)")

    # Verify target name
    assert y.name == TARGET_COLUMN


def test_feature_target_split_raises_on_nan_target(sample_csv_file: Path) -> None:
    """Verify ValueError is raised if target contains NaNs during split."""
    raw_df = load_raw_air_quality(sample_csv_file)
    cleaned_with_nans = clean_air_quality_data(raw_df, drop_missing_target=False)

    with pytest.raises(ValueError, match="contains .* NaN values"):
        get_feature_target_split(cleaned_with_nans)


def test_validate_schema_errors() -> None:
    """Test schema validation on malformed dataframes."""
    # Missing required target
    bad_df = pd.DataFrame({"PT08.S1(CO)": [100.0], "Timestamp": [pd.Timestamp.now()]})
    with pytest.raises(ValueError, match="missing required columns"):
        validate_schema(bad_df, is_cleaned=True)

    # Missing Timestamp in cleaned df
    cols = [TARGET_COLUMN] + PREDICTOR_FEATURES
    bad_df_no_ts = pd.DataFrame({c: [1.0] for c in cols})
    with pytest.raises(ValueError, match="must contain 'Timestamp' column"):
        validate_schema(bad_df_no_ts, is_cleaned=True)


# -----------------------------------------------------------------------------
# Integration Tests on Real Raw Dataset (if present)
# -----------------------------------------------------------------------------

@pytest.mark.skipif(not RAW_DATA_PATH.is_file(), reason="Real raw dataset not present")
def test_real_dataset_integrity() -> None:
    """Integration test verifying full real UCI dataset ingestion and properties."""
    full_df = load_dataset(RAW_DATA_PATH, drop_missing_target=False)
    assert full_df.shape == (9357, 16)
    assert full_df["Timestamp"].min() == pd.Timestamp("2004-03-10 18:00:00")
    assert full_df["Timestamp"].max() == pd.Timestamp("2005-04-04 14:00:00")
    assert full_df["Timestamp"].is_monotonic_increasing
    assert full_df["Timestamp"].duplicated().sum() == 0

    # Hourly continuity check
    diffs = full_df["Timestamp"].diff().dropna()
    assert (diffs == pd.Timedelta(hours=1)).all()

    # Target missingness check
    assert full_df[TARGET_COLUMN].isna().sum() == 366

    # Supervised split check
    supervised_df = load_dataset(RAW_DATA_PATH, drop_missing_target=True)
    assert supervised_df.shape == (8991, 16)
    X, y, ts = get_feature_target_split(supervised_df, include_environmental=True)
    assert X.shape == (8991, 8)
    assert y.shape == (8991,)
    assert not X.isna().any().any()
    assert not y.isna().any()
