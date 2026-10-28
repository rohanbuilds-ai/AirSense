"""Reusable data ingestion, validation, and cleaning pipeline for the UCI Air Quality dataset.

This module enforces strict data contracts:
- Dialect detection (semicolon delimiters, European decimal commas).
- Pruning of trailing empty artifact columns/rows.
- Timestamp parsing and chronological ordering validation.
- Conversion of official UCI missing value sentinel (-200) to NaN.
- Isolation of predictor features and exclusion of reference analyzer (GT) measurements.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Domain Constants & Schema Definitions
# -----------------------------------------------------------------------------

# Primary prediction target: Reference Benzene concentration
TARGET_COLUMN: str = "C6H6(GT)"

# 5 metal-oxide chemical sensor array measurements
SENSOR_FEATURES: list[str] = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
]

# Ambient meteorological variables
ENVIRONMENTAL_FEATURES: list[str] = [
    "T",   # Temperature in Celsius
    "RH",  # Relative Humidity %
    "AH",  # Absolute Humidity
]

# Authorized predictor feature set (8 features total)
PREDICTOR_FEATURES: list[str] = SENSOR_FEATURES + ENVIRONMENTAL_FEATURES

# Certified reference analyzer columns ending in (GT) that MUST be excluded from predictors
EXCLUDED_REFERENCE_COLUMNS: list[str] = [
    "CO(GT)",
    "NMHC(GT)",
    "NOx(GT)",
    "NO2(GT)",
]

# All expected raw columns in the official UCI Air Quality CSV
RAW_COLUMNS: list[str] = [
    "Date",
    "Time",
    "CO(GT)",
    "PT08.S1(CO)",
    "NMHC(GT)",
    "C6H6(GT)",
    "PT08.S2(NMHC)",
    "NOx(GT)",
    "PT08.S3(NOx)",
    "NO2(GT)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH",
]

# Missing value sentinel defined in UCI documentation
SENTINEL_MISSING_VALUE: float = -200.0


# -----------------------------------------------------------------------------
# Dialect Detection and Raw Ingestion
# -----------------------------------------------------------------------------

def detect_csv_dialect(file_path: str | Path, num_lines: int = 5) -> dict[str, Any]:
    """Inspect the raw file bytes/lines to verify CSV dialect without assumptions.

    Parameters
    ----------
    file_path : str | Path
        Path to the raw CSV file.
    num_lines : int, default=5
        Number of lines to sample for delimiter and decimal inspection.

    Returns
    -------
    dict[str, Any]
        Dictionary with detected 'delimiter', 'decimal', 'encoding', and 'has_trailing_delimiters'.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    encoding = "latin-1"
    with path.open("r", encoding=encoding) as f:
        sample_lines = [f.readline() for _ in range(num_lines)]

    if not sample_lines:
        raise ValueError(f"File {path} is empty.")

    header = sample_lines[0]
    # Semicolon detection
    delimiter = ";" if ";" in header else ","

    # Decimal detection: examine numbers in first data row
    first_data_line = sample_lines[1] if len(sample_lines) > 1 else ""
    # Semicolon-delimited files in Europe typically use comma as decimal
    decimal = "," if "," in first_data_line and delimiter == ";" else "."

    has_trailing_delimiters = header.strip().endswith(delimiter + delimiter)

    return {
        "delimiter": delimiter,
        "decimal": decimal,
        "encoding": encoding,
        "has_trailing_delimiters": has_trailing_delimiters,
    }


def load_raw_air_quality(file_path: str | Path) -> pd.DataFrame:
    """Read the raw UCI Air Quality CSV file with correct dialect and boundary handling.

    Parameters
    ----------
    file_path : str | Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Raw dataframe with stripped headers and removed artifact columns/rows.
    """
    dialect = detect_csv_dialect(file_path)
    df = pd.read_csv(
        file_path,
        sep=dialect["delimiter"],
        decimal=dialect["decimal"],
        encoding=dialect["encoding"],
    )

    # Clean header whitespace
    df.columns = df.columns.str.strip()

    # Drop artifact columns (e.g. Unnamed: 15, Unnamed: 16 created by trailing ';;')
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    df = df.dropna(how="all", axis=1)

    # Drop blank trailing rows where Date is NaN or empty
    if "Date" in df.columns:
        df = df.dropna(subset=["Date"])
        df = df[df["Date"].astype(str).str.strip() != ""]

    return df


# -----------------------------------------------------------------------------
# Cleaning, Timestamps, and Sentinel Conversion
# -----------------------------------------------------------------------------

def clean_air_quality_data(
    df: pd.DataFrame,
    drop_missing_target: bool = False,
) -> pd.DataFrame:
    """Clean the raw Air Quality dataframe.

    Steps:
    1. Parse and combine 'Date' and 'Time' into a single UTC datetime 'Timestamp'.
    2. Verify chronological monotonicity and check duplicate timestamps.
    3. Replace -200 sentinel values with np.nan across all numeric columns.
    4. Cast all sensor, weather, and reference columns to float64.
    5. Optionally drop rows where target C6H6(GT) is missing.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe loaded by load_raw_air_quality.
    drop_missing_target : bool, default=False
        If True, rows where C6H6(GT) is NaN are dropped for supervised evaluation.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe with sorted 'Timestamp' column and numeric columns as float64.
    """
    df = df.copy()

    # 1. Parse timestamps: format is DD/MM/YYYY and HH.MM.SS
    if "Date" in df.columns and "Time" in df.columns:
        # Time strings sometimes use dots instead of colons (e.g. '18.00.00')
        raw_ts = df["Date"].astype(str).str.strip() + " " + df["Time"].astype(str).str.strip()
        df["Timestamp"] = pd.to_datetime(raw_ts, format="%d/%m/%Y %H.%M.%S", errors="coerce")

        if df["Timestamp"].isna().any():
            raise ValueError("Found unparseable date/time entries in dataset.")

        # Ensure chronological ordering
        df = df.sort_values("Timestamp").reset_index(drop=True)

        # Check duplicate timestamps
        dup_count = df["Timestamp"].duplicated().sum()
        if dup_count > 0:
            raise ValueError(f"Found {dup_count} duplicate timestamps in dataset.")

    # 2. Identify numeric columns
    numeric_cols = [
        col for col in df.columns
        if col not in ["Date", "Time", "Timestamp"]
    ]

    # 3. Replace sentinel -200 with NaN and cast to float64
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        # Replace sentinel -200 (both integer and float representations)
        df[col] = df[col].replace(SENTINEL_MISSING_VALUE, np.nan)
        df[col] = df[col].astype(np.float64)

    # 4. Filter missing target rows if requested
    if drop_missing_target:
        if TARGET_COLUMN not in df.columns:
            raise KeyError(f"Target column '{TARGET_COLUMN}' not present in dataframe.")
        df = df.dropna(subset=[TARGET_COLUMN]).reset_index(drop=True)

    return df


def load_dataset(
    file_path: str | Path,
    drop_missing_target: bool = False,
) -> pd.DataFrame:
    """Convenience high-level loader: reads, cleans, and validates dataset.

    Parameters
    ----------
    file_path : str | Path
        Path to raw CSV file.
    drop_missing_target : bool, default=False
        Whether to drop rows with missing target C6H6(GT).

    Returns
    -------
    pd.DataFrame
        Cleaned, schema-verified dataframe.
    """
    raw_df = load_raw_air_quality(file_path)
    cleaned_df = clean_air_quality_data(raw_df, drop_missing_target=drop_missing_target)
    validate_schema(cleaned_df, is_cleaned=True)
    return cleaned_df


# -----------------------------------------------------------------------------
# Feature / Target Partitioning & Leakage Prevention
# -----------------------------------------------------------------------------

def get_feature_target_split(
    df: pd.DataFrame,
    include_environmental: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Extract design matrix features X, target vector y, and timestamps.

    Strict guardrails:
    - Never includes any reference analyzer columns ending in (GT) in X.
    - Raises error if target y contains NaNs, enforcing supervised evaluation safety.
    - Only includes physical gas sensors (5 features) or sensors + weather (8 features).

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    include_environmental : bool, default=True
        If True, includes T, RH, AH (8 features total).
        If False, includes only the 5 metal oxide sensors (PT08.S1 to PT08.S5).

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series]
        (X, y, timestamps)
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target '{TARGET_COLUMN}' missing from dataframe.")

    feature_cols = PREDICTOR_FEATURES if include_environmental else SENSOR_FEATURES

    # Ensure no reference columns accidentally present in feature_cols
    for col in feature_cols:
        if col.endswith("(GT)"):
            raise ValueError(f"CRITICAL ERROR: Reference column '{col}' cannot be a feature!")

    # Verify all feature columns exist
    missing_cols = set(feature_cols) - set(df.columns)
    if missing_cols:
        raise KeyError(f"Missing required feature columns: {missing_cols}")

    y = df[TARGET_COLUMN].copy()
    if y.isna().any():
        nan_count = y.isna().sum()
        raise ValueError(
            f"Target '{TARGET_COLUMN}' contains {nan_count} NaN values. "
            f"Filter rows with drop_missing_target=True before splitting."
        )

    X = df[feature_cols].copy()
    timestamps = df["Timestamp"].copy() if "Timestamp" in df.columns else pd.Series(df.index)

    return X, y, timestamps


# -----------------------------------------------------------------------------
# Schema & Contract Validation
# -----------------------------------------------------------------------------

def validate_schema(df: pd.DataFrame, is_cleaned: bool = True) -> bool:
    """Validate that the dataframe satisfies domain requirements.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe to validate.
    is_cleaned : bool, default=True
        Whether df is expected to have 'Timestamp' and float64 numeric columns.

    Returns
    -------
    bool
        True if schema is valid, raises ValueError or AssertionError otherwise.
    """
    # Check essential target and sensors
    required_cols = [TARGET_COLUMN] + SENSOR_FEATURES + ENVIRONMENTAL_FEATURES
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Schema violation: missing required columns {missing}")

    if is_cleaned:
        if "Timestamp" not in df.columns:
            raise ValueError("Schema violation: cleaned dataframe must contain 'Timestamp' column.")
        if not pd.api.types.is_datetime64_any_dtype(df["Timestamp"]):
            raise TypeError("Schema violation: 'Timestamp' column must be datetime64.")
        for col in required_cols:
            if not pd.api.types.is_float_dtype(df[col]):
                raise TypeError(f"Schema violation: column '{col}' must be float64.")

    return True
