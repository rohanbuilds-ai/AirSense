"""Unit tests for the Statistical Inference & Probability Laboratory."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from airsense.statistical_inference import (
    binomial_to_normal_approximation,
    chi_squared_contingency_test,
    confidence_interval_mean,
    diagnose_assumptions,
    map_gaussian_regression,
    margin_of_error_curve,
    mle_normal,
    multivariate_normal_fit,
    one_sample_t_test,
    one_way_anova,
    paired_t_test,
    simulate_clt,
    simulate_power_curve,
    two_sample_t_test,
)

PROCESSED_DATA_PATH = Path("data/processed/air_quality_supervised.csv")


# -----------------------------------------------------------------------------
# 1. Tests for Unit 3: Sampling, CLT & Distributions
# -----------------------------------------------------------------------------

def test_simulate_clt_convergence() -> None:
    """Test that sample means converge to true population mean and variance scales as sigma^2/n."""
    res = simulate_clt(population_distribution="exponential", sample_sizes=[10, 50, 200], n_simulations=500, seed=42)

    assert res["distribution"] == "exponential"
    # True mean is 2.0
    for n in [10, 50, 200]:
        means = res["sample_means"][n]
        assert np.mean(means) == pytest.approx(2.0, abs=0.15)
        # Variance of sample mean should be sigma^2 / n = 4.0 / n
        expected_var = 4.0 / n
        actual_var = np.var(means)
        assert actual_var == pytest.approx(expected_var, rel=0.25)


def test_binomial_to_normal_rule_of_thumb() -> None:
    """Verify Binomial to Normal approximation conditions and error bounds."""
    # Satisfies np >= 10 and n(1-p) >= 10: n=100, p=0.3 -> np=30, n(1-p)=70
    res = binomial_to_normal_approximation(n=100, p=0.3)
    assert res["rule_satisfied"] is True
    assert res["mean"] == pytest.approx(30.0)
    assert res["variance"] == pytest.approx(21.0)
    assert res["max_approximation_error"] < 0.02  # Close approximation

    # Fails rule: n=10, p=0.1 -> np=1 < 10
    res_fail = binomial_to_normal_approximation(n=10, p=0.1)
    assert res_fail["rule_satisfied"] is False


# -----------------------------------------------------------------------------
# 2. Tests for Point Estimation, MLE & Bayesian MAP
# -----------------------------------------------------------------------------

def test_mle_normal_bessel_correction() -> None:
    """Verify MLE variance bias vs Bessel's unbiased sample variance."""
    np.random.seed(42)
    data = np.random.normal(loc=5.0, scale=2.0, size=50)
    mle = mle_normal(data)

    assert mle["mu_mle"] == pytest.approx(np.mean(data))
    # MLE variance is (n-1)/n * s^2
    expected_mle_var = (49.0 / 50.0) * mle["s2_unbiased"]
    assert mle["sigma2_mle"] == pytest.approx(expected_mle_var, rel=1e-12)
    # Theoretical bias is -(1/n) * s^2
    assert mle["empirical_difference"] == pytest.approx(mle["theoretical_bias"], rel=1e-12)


def test_map_gaussian_regression_ridge_equivalence() -> None:
    """Verify that Bayesian MAP with Gaussian prior is analytically identical to Ridge regression."""
    np.random.seed(99)
    n, d = 40, 4
    X = np.random.randn(n, d)
    true_w = np.array([2.0, -1.0, 0.5, 1.2])
    y = X @ true_w + 0.5 * np.random.randn(n)

    prior_var = 2.0  # tau^2
    noise_var = 1.0  # sigma^2
    # Equivalent Ridge penalty: lambda = sigma^2 / tau^2 = 1.0 / 2.0 = 0.5
    res = map_gaussian_regression(X, y, prior_variance=prior_var, noise_variance=noise_var)

    assert res["equivalent_ridge_lambda"] == pytest.approx(0.5)

    # Compute Ridge analytical directly: (X^T X + lambda I)^-1 X^T y
    ridge_direct = np.linalg.solve(X.T @ X + 0.5 * np.eye(d), X.T @ y)
    assert np.allclose(res["w_map"], ridge_direct, atol=1e-12)

    # When prior variance -> infinity (no prior regularization), MAP converges to OLS
    res_uninformative = map_gaussian_regression(X, y, prior_variance=1e8, noise_variance=1.0)
    assert np.allclose(res_uninformative["w_map"], res["w_ols"], atol=1e-6)


# -----------------------------------------------------------------------------
# 3. Tests for Confidence Intervals & MOE Scaling
# -----------------------------------------------------------------------------

def test_confidence_interval_small_vs_large_sample() -> None:
    """Verify Student's t for small samples vs z for large samples."""
    np.random.seed(42)
    small_sample = np.random.normal(loc=10.0, scale=2.0, size=15)
    large_sample = np.random.normal(loc=10.0, scale=2.0, size=100)

    ci_small = confidence_interval_mean(small_sample, confidence=0.95, method="auto")
    assert ci_small["method_used"] == "t"
    assert ci_small["df"] == 14

    ci_large = confidence_interval_mean(large_sample, confidence=0.95, method="auto")
    assert ci_large["method_used"] == "z"
    assert ci_large["df"] is None

    # Both intervals must contain the sample mean at their center
    assert (ci_small["ci_lower"] + ci_small["ci_upper"]) / 2.0 == pytest.approx(ci_small["sample_mean"])
    assert (ci_large["ci_lower"] + ci_large["ci_upper"]) / 2.0 == pytest.approx(ci_large["sample_mean"])


def test_margin_of_error_inverse_sqrt_scaling() -> None:
    """Verify MOE scales inversely with sqrt(n)."""
    curve = margin_of_error_curve(sample_std=4.0, sample_sizes=[100, 400], confidence=0.95)
    moe_100 = curve.loc[curve["n"] == 100, "Margin_of_Error"].values[0]
    moe_400 = curve.loc[curve["n"] == 400, "Margin_of_Error"].values[0]

    # Quadrupling sample size (100 -> 400) cuts margin of error roughly in half
    ratio = moe_100 / moe_400
    assert ratio == pytest.approx(2.0, rel=0.05)


# -----------------------------------------------------------------------------
# 4. Tests for Hypothesis Testing & Duality
# -----------------------------------------------------------------------------

def test_one_sample_t_test_and_ci_duality() -> None:
    """Verify the fundamental duality: H_0 rejected at level alpha <=> popmean outside (1-alpha) CI."""
    np.random.seed(42)
    data = np.random.normal(loc=15.0, scale=2.0, size=25)

    # Test 1: True mean is 15.0 -> Should NOT reject H_0: mu = 15.0
    res_null_true = one_sample_t_test(data, popmean=15.0, alpha=0.05)
    assert res_null_true["reject_null"] is False
    assert res_null_true["hypothesized_in_ci"] is True
    assert res_null_true["duality_satisfied"] is True

    # Test 2: Hypothesize mean is 25.0 -> Far outside CI, MUST reject H_0
    res_null_false = one_sample_t_test(data, popmean=25.0, alpha=0.05)
    assert res_null_false["reject_null"] is True
    assert res_null_false["hypothesized_in_ci"] is False
    assert res_null_false["duality_satisfied"] is True


def test_two_sample_welch_t_test() -> None:
    """Verify two-sample Welch's t-test matches SciPy."""
    np.random.seed(123)
    g1 = np.random.normal(loc=10.0, scale=2.0, size=30)
    g2 = np.random.normal(loc=12.0, scale=4.0, size=35)  # Unequal variances!

    res = two_sample_t_test(g1, g2, equal_var=False, alpha=0.05)
    ref = stats.ttest_ind(g1, g2, equal_var=False)

    assert res["t_statistic"] == pytest.approx(ref.statistic, rel=1e-10)
    assert res["p_value"] == pytest.approx(ref.pvalue, rel=1e-10)
    assert res["reject_null"] is True  # Groups differ significantly


def test_paired_t_test() -> None:
    """Verify paired t-test on matched before/after observations."""
    np.random.seed(42)
    before = np.random.normal(loc=20.0, scale=3.0, size=25)
    # After has a systematic reduction of 1.5 units
    after = before - 1.5 + np.random.normal(loc=0.0, scale=0.5, size=25)

    res = paired_t_test(before, after, alpha=0.05)
    ref = stats.ttest_rel(after, before)

    assert res["t_statistic"] == pytest.approx(ref.statistic, rel=1e-10)
    assert res["p_value"] == pytest.approx(ref.pvalue, rel=1e-10)
    assert res["mean_diff"] == pytest.approx(-1.5, abs=0.3)
    assert res["reject_null"] is True


# -----------------------------------------------------------------------------
# 5. Tests for One-Way ANOVA & Chi-Squared Contingency Tests
# -----------------------------------------------------------------------------

def test_one_way_anova_ss_partitioning_and_eta_squared() -> None:
    """Verify SS partitioning: SS_total = SS_between + SS_within, and F-statistic matches SciPy."""
    np.random.seed(42)
    g1 = np.random.normal(loc=5.0, scale=1.5, size=20)
    g2 = np.random.normal(loc=8.0, scale=1.5, size=25)
    g3 = np.random.normal(loc=12.0, scale=1.5, size=20)

    res = one_way_anova([g1, g2, g3], alpha=0.05)
    ref = stats.f_oneway(g1, g2, g3)

    assert res["F_statistic"] == pytest.approx(ref.statistic, rel=1e-10)
    assert res["p_value"] == pytest.approx(ref.pvalue, rel=1e-10)

    # Invariant: SS_total == SS_between + SS_within
    assert res["ss_total"] == pytest.approx(res["ss_between"] + res["ss_within"], rel=1e-12)
    # Invariant: eta_squared == SS_between / SS_total
    assert res["eta_squared"] == pytest.approx(res["ss_between"] / res["ss_total"], rel=1e-12)
    assert 0.0 < res["eta_squared"] < 1.0


def test_chi_squared_contingency_cochran_and_cramers_v() -> None:
    """Verify Chi-squared contingency test, Cochran condition, and Cramér's V."""
    # Contingency table: Day Type (Weekday vs Weekend) x Pollution Level (Low vs High)
    table = np.array([
        [150, 50],   # Weekdays: 150 low, 50 high
        [30, 70],    # Weekends: 30 low, 70 high
    ])

    res = chi_squared_contingency_test(table, alpha=0.05)
    ref_chi2, ref_p, ref_dof, ref_exp = stats.chi2_contingency(table, correction=False)

    assert res["chi2_statistic"] == pytest.approx(ref_chi2, rel=1e-10)
    assert res["p_value"] == pytest.approx(ref_p, rel=1e-10)
    assert res["df"] == ref_dof
    assert np.allclose(res["expected_table"], ref_exp, atol=1e-10)

    # All expected cells >= 5
    assert res["assumption_satisfied"] is True
    # Significant association
    assert res["reject_null"] is True
    assert 0.0 < res["cramers_v"] <= 1.0


# -----------------------------------------------------------------------------
# 6. Tests for Assumption Diagnostics & Power Simulations
# -----------------------------------------------------------------------------

def test_diagnose_assumptions_catches_non_normality() -> None:
    """Verify assumption diagnostics correctly detect skewed non-normal distributions."""
    np.random.seed(42)
    exp_data = np.random.exponential(scale=2.0, size=200)
    diag = diagnose_assumptions(exp_data)

    assert diag["is_normal_05"] is False
    assert diag["skewness"] > 1.0


def test_simulate_power_curve_type1_error() -> None:
    """Verify that when effect size d=0.0 (Null is True), empirical rejection rate ~ alpha (0.05)."""
    curve = simulate_power_curve(effect_sizes=[0.0, 0.8], sample_sizes=[50], alpha=0.05, n_simulations=500, seed=42)

    type1 = curve.loc[curve["Effect_Size_d"] == 0.0, "Type_I_Error"].values[0]
    # For alpha=0.05 with 500 simulations, Type I error should be roughly between 0.02 and 0.08
    assert 0.02 <= type1 <= 0.08

    # When d=0.8 (large effect), power should be very high (> 0.95 for n=50)
    power = curve.loc[curve["Effect_Size_d"] == 0.8, "Empirical_Power"].values[0]
    assert power > 0.95
