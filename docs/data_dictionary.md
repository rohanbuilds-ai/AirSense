# AirSense: Dataset Dictionary & Schema Specification

## 1. Dataset Overview

- **Source**: UCI Machine Learning Repository — [Air Quality Dataset](https://archive.ics.uci.edu/dataset/360/air+quality)
- **Deployment Location**: Roadside field site in an Italian city with significant automotive traffic.
- **Duration**: March 10, 2004 (18:00:00) to April 4, 2005 (14:00:00).
- **Sampling Frequency**: Continuous hourly measurements ($9,357$ valid consecutive hourly slots spanning $389.8$ days).
- **Measurement Hardware**:
  1. An array of **5 solid-state chemical gas sensors** based on metal-oxide semiconductors (MOS / PT08 series).
  2. Certified **reference analyzers** (optical/chemical ground-truth instruments operated by environmental protection authorities).
  3. Meteorological sensors for ambient **Temperature**, **Relative Humidity**, and **Absolute Humidity**.

---

## 2. Complete Field Specifications

| Column Name | Physical Quantity | Measurement Unit | Hardware / Principle | Missing % | AirSense Role | Excluded from Predictors? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`Date`** | Calendar Date | DD/MM/YYYY | System Clock | 0.0% | Temporal Index | **Yes** (converted to `Timestamp`) |
| **`Time`** | Sampling Time | HH.MM.SS | System Clock | 0.0% | Temporal Index | **Yes** (converted to `Timestamp`) |
| **`Timestamp`** | Unified UTC Timestamp | ISO-8601 | Derived | 0.0% | Ordering & Split Key | **Yes** (not a linear feature) |
| **`PT08.S1(CO)`** | Carbon Monoxide Sensor Response | Sensor Resistance ($\Omega$) | Tin Oxide ($SnO_2$) MOS | 3.91% | **Predictor Feature** | **No** (Authorized input) |
| **`PT08.S2(NMHC)`** | Hydrocarbon Sensor Response | Sensor Resistance ($\Omega$) | Titania ($TiO_2$) MOS | 3.91% | **Predictor Feature** | **No** (Authorized input) |
| **`PT08.S3(NOx)`** | Nitrogen Oxides Sensor Response | Sensor Resistance ($\Omega$) | Tungsten Oxide ($WO_3$) MOS | 3.91% | **Predictor Feature** | **No** (Authorized input) |
| **`PT08.S4(NO2)`** | Nitrogen Dioxide Sensor Response | Sensor Resistance ($\Omega$) | Tungsten Oxide ($WO_3$) MOS | 3.91% | **Predictor Feature** | **No** (Authorized input) |
| **`PT08.S5(O3)`** | Ozone Sensor Response | Sensor Resistance ($\Omega$) | Indium Oxide ($In_2O_3$) MOS | 3.91% | **Predictor Feature** | **No** (Authorized input) |
| **`T`** | Ambient Temperature | Degrees Celsius (°C) | Thermistor | 3.91% | **Predictor Feature** | **No** (Authorized input) |
| **`RH`** | Relative Humidity | Percent (%) | Capacitive Sensor | 3.91% | **Predictor Feature** | **No** (Authorized input) |
| **`AH`** | Absolute Humidity | $g / m^3$ | Derived Physical Variable | 3.91% | **Predictor Feature** | **No** (Authorized input) |
| **`C6H6(GT)`** | Reference Benzene Concentration | $\mu g / m^3$ | Certified Chromatographic Analyzer | 3.91% | **Prediction Target ($y$)** | **No** (Target variable) |
| **`CO(GT)`** | Reference Carbon Monoxide | $mg / m^3$ | Certified NDIR Reference Analyzer | 17.99% | Ground-Truth Reference | **YES (EXCLUDED)** |
| **`NMHC(GT)`** | Non-Methane Hydrocarbons | $\mu g / m^3$ | Certified Reference Analyzer | 90.23% | Ground-Truth Reference | **YES (EXCLUDED)** |
| **`NOx(GT)`** | Nitrogen Oxides | $ppb$ | Certified Chemiluminescence Analyzer | 17.52% | Ground-Truth Reference | **YES (EXCLUDED)** |
| **`NO2(GT)`** | Nitrogen Dioxide | $\mu g / m^3$ | Certified Chemiluminescence Analyzer | 17.55% | Ground-Truth Reference | **YES (EXCLUDED)** |

---

## 3. Mandatory Reference Exclusion Policy

All reference measurements ending in `(GT)` (Ground Truth) **except** the designated target `C6H6(GT)` are strictly forbidden from being used as predictive features.

### Rationale:
1. **Target Contamination & Physical Purpose**: The fundamental research goal of AirSense is calibrating an array of low-cost, imperfect chemical sensors to replace an expensive certified laboratory analyzer. Feeding other reference laboratory analyzers (`CO(GT)`, `NOx(GT)`, `NO2(GT)`) as inputs would defeat the physical purpose of sensor calibration and create unrealistic predictive performance.
2. **Defective Hardware & Massive Missingness**: The `NMHC(GT)` reference analyzer failed early in the trial, resulting in **90.23% missing values** (8,443 missing hours out of 9,357). Using it would discard almost the entire dataset.
3. **Data Leakage**: Co-pollutant reference concentrations measured by high-grade instruments introduce uncalibrated ground-truth information not available to deployed low-cost sensor nodes.

---

## 4. Missing Value Sentinel (`-200`) and Cleaning Decisions

- **Sentinel Representation**: The UCI documentation specifies that missing field values are encoded as `-200` (or `-200.0`).
- **Physical Impossibility**: Sensor resistance, pollutant concentrations, and absolute humidity cannot take negative values. Any occurrence of `-200` is immediately converted to IEEE 754 `NaN`.
- **Coincident Missingness Discovery**:
  - The 5 gas sensors, 3 meteorological variables, and target `C6H6(GT)` have **identically 366 missing values** ($3.91\%$).
  - Crucially, all 366 missing values occur **simultaneously at the exact same hourly timestamps** (indicating complete power/logging outages of the monitoring station).
  - Therefore, filtering out missing target instances (`drop_missing_target=True`) yields an unbroken supervised dataset of **8,991 rows** with **zero missing values** across all 8 input features. No arbitrary interpolation or synthetic imputation is required for the supervised calibration matrix.

---

## 5. Storage and File Formats

- **Raw Data**: Stored in `data/raw/AirQualityUCI.csv` (read-only, never modified).
- **Cleaned Dataset**: Stored in `data/processed/air_quality_cleaned.csv` ($9,357$ rows $\times$ $16$ columns, containing parsed timestamps and `NaN` placeholders).
- **Supervised Evaluation Dataset**: Stored in `data/processed/air_quality_supervised.csv` ($8,991$ rows $\times$ $16$ columns, strictly non-null target observations).
