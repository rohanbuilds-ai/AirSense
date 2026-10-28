"""Statistical Inference & Educational Probability Laboratory for AirSense.

Implements syllabus requirements for Unit 3 & Unit 4:
- Sampling distributions and Central Limit Theorem (CLT) simulations.
- Point estimation (MLE for normal, unbiased vs biased variance, MSE).
- Bayesian MAP estimation with Gaussian conjugate priors (Ridge equivalence).
- Interval estimation (z-test, Student's t-test, margin of error scaling).
- Hypothesis testing (one-sample, two-sample Welch/Student, paired t-tests).
- Duality between hypothesis tests and confidence intervals.
- One-Way ANOVA with eta-squared (eta^2) effect size.
- Chi-squared test of independence with Cramér's V and assumption checks.
- Statistical assumption diagnostics (Shapiro-Wilk, Levene, serial autocorrelation).
- Type I / Type II error and statistical power simulations.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# -----------------------------------------------------------------------------
# 1. Unit 3: Sampling, Central Limit Theorem & Distributions
# -----------------------------------------------------------------------------

def simulate_clt(
    population_distribution: str = "exponential",
    sample_sizes: list[int] | None = None,
    n_simulations: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Simulate the Central Limit Theorem (CLT) on skewed or non-normal parent distributions.

    Demonstrates that as sample size n increases (typically n >= 30), the sampling
    distribution of the sample mean bar{X}_n converges to a Normal distribution
    N(mu, sigma^2 / n), even when the parent population is highly non-normal.

    Parameters
    ----------
    population_distribution : str, default='exponential'
        Parent distribution ('exponential', 'uniform', or 'lognormal').
    sample_sizes : list[int] | None, default=None
        List of sample sizes to simulate (default: [2, 5, 15, 30, 100]).
    n_simulations : int, default=1000
        Number of independent samples drawn for each sample size.
    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    dict[str, Any]
        Dictionary with 'sample_means', 'parent_samples', 'pop_mean', 'pop_std',
        and normality test p-values for each n.
    """
    rng = np.random.default_rng(seed)
    sizes = sample_sizes or [2, 5, 15, 30, 100]

    # Draw parent population benchmark
    n_parent = 50000
    if population_distribution == "exponential":
        parent = rng.exponential(scale=2.0, size=n_parent)
        mu_true = 2.0
        sigma_true = 2.0
    elif population_distribution == "uniform":
        parent = rng.uniform(low=0.0, high=10.0, size=n_parent)
        mu_true = 5.0
        sigma_true = np.sqrt(100.0 / 12.0)
    elif population_distribution == "lognormal":
        parent = rng.lognormal(mean=0.0, sigma=0.8, size=n_parent)
        mu_true = float(np.exp(0.0 + 0.8**2 / 2.0))
        sigma_true = float(np.sqrt((np.exp(0.8**2) - 1.0) * np.exp(2 * 0.0 + 0.8**2)))
    else:
        raise ValueError(f"Unsupported population distribution: '{population_distribution}'.")

    means_by_n: dict[int, np.ndarray] = {}
    normality_pvals: dict[int, float] = {}

    for n in sizes:
        if population_distribution == "exponential":
            samples = rng.exponential(scale=2.0, size=(n_simulations, n))
        elif population_distribution == "uniform":
            samples = rng.uniform(low=0.0, high=10.0, size=(n_simulations, n))
        else:  # lognormal
            samples = rng.lognormal(mean=0.0, sigma=0.8, size=(n_simulations, n))

        sample_means = np.mean(samples, axis=1)
        means_by_n[n] = sample_means

        # Test normality of sampling distribution (D'Agostino-Pearson test)
        stat, pval = stats.normaltest(sample_means)
        normality_pvals[n] = float(pval)

    return {
        "distribution": population_distribution,
        "parent_samples": parent[:5000],
        "true_mean": mu_true,
        "true_std": sigma_true,
        "sample_sizes": sizes,
        "sample_means": means_by_n,
        "normality_pvalues": normality_pvals,
    }


def binomial_to_normal_approximation(
    n: int,
    p: float,
) -> dict[str, Any]:
    """Demonstrate the Normal approximation to the Binomial distribution B(n, p).

    Evaluates the standard rule of thumb: n*p >= 10 and n*(1-p) >= 10.
    Compares exact Binomial PMF with continuity-corrected Normal density N(np, np(1-p)).

    Parameters
    ----------
    n : int
        Number of independent trials.
    p : float
        Probability of success in each trial.

    Returns
    -------
    dict[str, Any]
        Dictionary with exact PMF, normal PDF, mean, variance, and rule-of-thumb flag.
    """
    if n <= 0:
        raise ValueError("Number of trials n must be positive.")
    if not (0.0 < p < 1.0):
        raise ValueError("Probability p must be in (0, 1).")

    mu = n * p
    var = n * p * (1.0 - p)
    std = np.sqrt(var)

    rule_satisfied = bool((n * p >= 10) and (n * (1.0 - p) >= 10))

    k_vals = np.arange(max(0, int(mu - 4 * std)), min(n + 1, int(mu + 4 * std) + 1))
    exact_pmf = stats.binom.pmf(k_vals, n, p)

    # Continuity corrected normal approximation: P(k - 0.5 <= X <= k + 0.5)
    approx_pmf = stats.norm.cdf(k_vals + 0.5, loc=mu, scale=std) - stats.norm.cdf(k_vals - 0.5, loc=mu, scale=std)
    max_abs_diff = float(np.max(np.abs(exact_pmf - approx_pmf)))

    return {
        "n": n,
        "p": p,
        "mean": mu,
        "variance": var,
        "std": std,
        "rule_satisfied": rule_satisfied,
        "k_values": k_vals,
        "exact_pmf": exact_pmf,
        "approx_pmf": approx_pmf,
        "max_approximation_error": max_abs_diff,
    }


# -----------------------------------------------------------------------------
# 2. Point Estimation, Maximum Likelihood & MAP (Bayesian-Ridge)
# -----------------------------------------------------------------------------

def mle_normal(data: np.ndarray) -> dict[str, Any]:
    """Compute the analytical Maximum Likelihood Estimators (MLE) for a Gaussian sample.

    Demonstrates the Bias-Variance tradeoff and Bessel's correction:
    - mu_MLE = (1/n) sum x_i  (Unbiased: E[mu_MLE] = mu)
    - sigma^2_MLE = (1/n) sum (x_i - mu_hat)^2  (Biased: E[sigma^2_MLE] = ((n-1)/n) * sigma^2)
    - s^2 = (1/(n-1)) sum (x_i - mu_hat)^2  (Unbiased sample variance)

    Parameters
    ----------
    data : np.ndarray
        1D sample array.

    Returns
    -------
    dict[str, Any]
        Dictionary with mu_mle, sigma2_mle, s2_unbiased, and sample size n.
    """
    x = np.asarray(data, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        raise ValueError("MLE estimation requires at least 2 observations.")

    mu_mle = float(np.mean(x))
    deviations = x - mu_mle
    sigma2_mle = float(np.sum(deviations**2) / n)
    s2_unbiased = float(np.sum(deviations**2) / (n - 1))
    bias = sigma2_mle - s2_unbiased

    return {
        "n": n,
        "mu_mle": mu_mle,
        "sigma2_mle": sigma2_mle,
        "sigma_mle": float(np.sqrt(sigma2_mle)),
        "s2_unbiased": s2_unbiased,
        "s_unbiased": float(np.sqrt(s2_unbiased)),
        "theoretical_bias": float(-(1.0 / n) * s2_unbiased),
        "empirical_difference": bias,
    }


def map_gaussian_regression(
    X: np.ndarray,
    y: np.ndarray,
    prior_variance: float = 1.0,
    noise_variance: float = 1.0,
) -> dict[str, Any]:
    """Compute Maximum A Posteriori (MAP) parameter estimation in linear regression.

    Specified Bayesian Prior:
    w ~ N(0, tau^2 * I) where tau^2 = prior_variance.
    Likelihood:
    y | X, w ~ N(X w, sigma^2 * I) where sigma^2 = noise_variance.

    Analytically proves the equivalence between Bayesian Gaussian MAP and L2 Ridge Regularization:
    w_MAP = (X^T X + lambda * I)^{-1} X^T y, where lambda = sigma^2 / tau^2.

    Parameters
    ----------
    X : np.ndarray
        Design matrix (n, d).
    y : np.ndarray
        Target vector (n,).
    prior_variance : float, default=1.0
        Prior parameter variance tau^2.
    noise_variance : float, default=1.0
        Observation noise variance sigma^2.

    Returns
    -------
    dict[str, Any]
        Dictionary with w_map, posterior_covariance, equivalent_ridge_lambda, and w_ols.
    """
    X_mat = np.asarray(X, dtype=np.float64)
    y_vec = np.asarray(y, dtype=np.float64).ravel()
    n, d = X_mat.shape

    if prior_variance <= 0 or noise_variance <= 0:
        raise ValueError("Variances must be strictly positive.")

    # Equivalent Ridge regularization penalty: lambda = sigma^2 / tau^2
    ridge_lambda = noise_variance / prior_variance

    XtX = X_mat.T @ X_mat
    Xty = X_mat.T @ y_vec

    # Posterior precision = (1 / sigma^2) * X^T X + (1 / tau^2) * I
    # w_MAP = (X^T X + (sigma^2 / tau^2) I)^{-1} X^T y
    reg_matrix = XtX + ridge_lambda * np.eye(d)
    w_map = np.linalg.solve(reg_matrix, Xty)

    # Posterior covariance: Sigma_w = sigma^2 * (X^T X + lambda * I)^{-1}
    posterior_cov = noise_variance * np.linalg.inv(reg_matrix)

    # Standard OLS solution for comparison
    w_ols, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)

    return {
        "w_map": w_map,
        "posterior_covariance": posterior_cov,
        "equivalent_ridge_lambda": ridge_lambda,
        "prior_variance": prior_variance,
        "noise_variance": noise_variance,
        "w_ols": w_ols,
        "shrinkage_norm_ratio": float(np.linalg.norm(w_map) / (np.linalg.norm(w_ols) + 1e-12)),
    }


# -----------------------------------------------------------------------------
# 3. Confidence Intervals & Margin of Error Behaviour
# -----------------------------------------------------------------------------

def confidence_interval_mean(
    data: np.ndarray,
    confidence: float = 0.95,
    method: str = "auto",
) -> dict[str, Any]:
    """Compute confidence interval for population mean using Student's t or Normal z method.

    Parameters
    ----------
    data : np.ndarray
        Sample values.
    confidence : float, default=0.95
        Confidence level (0 < confidence < 1).
    method : str, default='auto'
        'auto' uses Student's t for n < 30 and z for n >= 30.
        't' enforces Student's t distribution.
        'z' enforces Standard Normal distribution.

    Returns
    -------
    dict[str, Any]
        Dictionary with sample_mean, std_err, margin_of_error, ci_lower, ci_upper, method, df.
    """
    x = np.asarray(data, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        raise ValueError("Confidence interval requires at least 2 observations.")
    if not (0.0 < confidence < 1.0):
        raise ValueError("Confidence level must be between 0 and 1.")

    sample_mean = float(np.mean(x))
    sample_std = float(np.std(x, ddof=1))
    std_err = sample_std / np.sqrt(n)
    alpha = 1.0 - confidence

    chosen_method = method
    if method == "auto":
        chosen_method = "t" if n < 30 else "z"

    if chosen_method == "t":
        df = n - 1
        critical_val = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
    elif chosen_method == "z":
        df = None
        critical_val = float(stats.norm.ppf(1.0 - alpha / 2.0))
    else:
        raise ValueError(f"Unknown method '{method}'; choose 'auto', 't', or 'z'.")

    margin_of_error = critical_val * std_err
    ci_lower = sample_mean - margin_of_error
    ci_upper = sample_mean + margin_of_error

    return {
        "n": n,
        "sample_mean": sample_mean,
        "sample_std": sample_std,
        "std_err": std_err,
        "confidence": confidence,
        "alpha": alpha,
        "critical_value": critical_val,
        "margin_of_error": margin_of_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "method_used": chosen_method,
        "df": df,
    }


def margin_of_error_curve(
    sample_std: float,
    sample_sizes: list[int] | None = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Demonstrate the inverse square-root scaling of Margin of Error: MOE proportional to 1 / sqrt(n).

    Demonstrates the fundamental sample size rule: To cut the margin of error in half (2x precision),
    sample size must be quadrupled (4x sample size).

    Parameters
    ----------
    sample_std : float
        Standard deviation s of the population.
    sample_sizes : list[int] | None
        List of sample sizes (default: [10, 25, 50, 100, 250, 500, 1000, 4000]).
    confidence : float, default=0.95
        Confidence level.

    Returns
    -------
    pd.DataFrame
        Table with n, Critical Value, Standard Error, and Margin of Error.
    """
    sizes = sample_sizes or [10, 25, 50, 100, 250, 500, 1000, 4000]
    records = []
    alpha = 1.0 - confidence

    for n in sizes:
        df = n - 1
        t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=df))
        se = sample_std / np.sqrt(n)
        moe = t_crit * se
        records.append({
            "n": n,
            "sqrt_n": round(np.sqrt(n), 2),
            "Critical_t": round(t_crit, 3),
            "Std_Error": round(se, 4),
            "Margin_of_Error": round(moe, 4),
        })

    return pd.DataFrame(records)


def multivariate_normal_fit(X: np.ndarray) -> dict[str, Any]:
    """Fit parameters (mu, Sigma) of a Multivariate Normal distribution and compute Mahalanobis distances.

    Mahalanobis distance squared: D_M^2(x) = (x - mu)^T Sigma^{-1} (x - mu) ~ Chi^2(d).

    Parameters
    ----------
    X : np.ndarray
        Data matrix of shape (n, d).

    Returns
    -------
    dict[str, Any]
        Dictionary with mean_vector, covariance_matrix, mahalanobis_distances, and chi2_pval.
    """
    X_mat = np.asarray(X, dtype=np.float64)
    n, d = X_mat.shape

    mu = np.mean(X_mat, axis=0)
    dev = X_mat - mu
    cov = (dev.T @ dev) / (n - 1)

    cov_inv = np.linalg.pinv(cov)
    # Compute Mahalanobis distance squared for each observation
    d2 = np.sum((dev @ cov_inv) * dev, axis=1)

    # Theoretical chi-squared CDF p-values
    p_vals = 1.0 - stats.chi2.cdf(d2, df=d)

    return {
        "n_samples": n,
        "n_dimensions": d,
        "mean_vector": mu,
        "covariance_matrix": cov,
        "mahalanobis_sq": d2,
        "p_values": p_vals,
    }


# -----------------------------------------------------------------------------
# 4. Unit 4: Hypothesis Testing & Duality with Confidence Intervals
# -----------------------------------------------------------------------------

def one_sample_t_test(
    data: np.ndarray,
    popmean: float,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> dict[str, Any]:
    """Execute One-Sample Student's t-test and verify duality with the confidence interval.

    Hypotheses:
    H_0: mu = popmean
    H_1: mu != popmean (two-sided)

    Duality Principle:
    A two-sided test rejects H_0 at significance level alpha IF AND ONLY IF
    popmean falls outside the (1 - alpha) Confidence Interval!

    Parameters
    ----------
    data : np.ndarray
        Sample values.
    popmean : float
        Hypothesized population mean mu_0.
    alpha : float, default=0.05
        Significance level.
    alternative : str, default='two-sided'
        'two-sided', 'less', or 'greater'.

    Returns
    -------
    dict[str, Any]
        Dictionary containing t_statistic, p_value, df, reject_null, effect_size (Cohen's d),
        ci_lower, ci_upper, and duality_verified flag.
    """
    x = np.asarray(data, dtype=np.float64).ravel()
    n = len(x)
    if n < 2:
        raise ValueError("t-test requires at least 2 samples.")

    sample_mean = float(np.mean(x))
    sample_std = float(np.std(x, ddof=1))
    std_err = sample_std / np.sqrt(n)

    t_stat = (sample_mean - popmean) / std_err
    df = n - 1

    res = stats.ttest_1samp(x, popmean=popmean, alternative=alternative)
    p_val = float(res.pvalue)

    reject_null = bool(p_val < alpha)

    # Cohen's d effect size: (bar{x} - mu_0) / s
    cohens_d = float((sample_mean - popmean) / sample_std)

    # Compute (1 - alpha) CI
    ci_info = confidence_interval_mean(x, confidence=1.0 - alpha, method="t")
    in_ci = bool(ci_info["ci_lower"] <= popmean <= ci_info["ci_upper"])

    # Duality check: reject_null == (popmean not in CI) for two-sided tests
    duality_satisfied = (reject_null != in_ci) if alternative == "two-sided" else None

    return {
        "n": n,
        "sample_mean": sample_mean,
        "hypothesized_mean": popmean,
        "t_statistic": float(t_stat),
        "p_value": p_val,
        "df": df,
        "alpha": alpha,
        "reject_null": reject_null,
        "cohens_d": cohens_d,
        "ci_lower": ci_info["ci_lower"],
        "ci_upper": ci_info["ci_upper"],
        "hypothesized_in_ci": in_ci,
        "duality_satisfied": duality_satisfied,
    }


def two_sample_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
    equal_var: bool = False,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Execute Two-Sample t-test (Welch's or Student's) comparing two independent groups.

    Hypotheses:
    H_0: mu_1 = mu_2
    H_1: mu_1 != mu_2

    Parameters
    ----------
    group1, group2 : np.ndarray
        Samples from group 1 and group 2.
    equal_var : bool, default=False
        If False, uses Welch's t-test with Welch-Satterthwaite degrees of freedom
        (recommended for chemical sensors with heterogeneous variance).
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict[str, Any]
        Dictionary with t_statistic, p_value, df, cohens_d, and reject_null.
    """
    g1 = np.asarray(group1, dtype=np.float64).ravel()
    g2 = np.asarray(group2, dtype=np.float64).ravel()
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        raise ValueError("Both groups must contain at least 2 observations.")

    m1, m2 = float(np.mean(g1)), float(np.mean(g2))
    s1, s2 = float(np.std(g1, ddof=1)), float(np.std(g2, ddof=1))

    # Pooled standard deviation for Cohen's d
    s_pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    cohens_d = (m1 - m2) / s_pooled

    res = stats.ttest_ind(g1, g2, equal_var=equal_var)
    t_stat = float(res.statistic)
    p_val = float(res.pvalue)

    if equal_var:
        df = n1 + n2 - 2
    else:
        # Welch-Satterthwaite formula
        v1 = s1**2 / n1
        v2 = s2**2 / n2
        df = (v1 + v2)**2 / (v1**2 / (n1 - 1) + v2**2 / (n2 - 1))

    return {
        "n1": n1,
        "n2": n2,
        "mean1": m1,
        "mean2": m2,
        "std1": s1,
        "std2": s2,
        "t_statistic": t_stat,
        "p_value": p_val,
        "df": float(df),
        "cohens_d": float(cohens_d),
        "equal_var": equal_var,
        "alpha": alpha,
        "reject_null": bool(p_val < alpha),
    }


def paired_t_test(
    before: np.ndarray,
    after: np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Execute Paired Student's t-test on matched differences d_i = after_i - before_i.

    Parameters
    ----------
    before, after : np.ndarray
        Paired observation vectors of identical length.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict[str, Any]
        Dictionary with t_statistic, p_value, mean_diff, std_diff, cohens_d, and reject_null.
    """
    b = np.asarray(before, dtype=np.float64).ravel()
    a = np.asarray(after, dtype=np.float64).ravel()
    if len(b) != len(a):
        raise ValueError(f"Paired vectors must have same length; got {len(b)} and {len(a)}.")

    diff = a - b
    res = one_sample_t_test(diff, popmean=0.0, alpha=alpha)
    return {
        "n_pairs": len(b),
        "mean_diff": float(np.mean(diff)),
        "std_diff": float(np.std(diff, ddof=1)),
        "t_statistic": res["t_statistic"],
        "p_value": res["p_value"],
        "df": res["df"],
        "cohens_d": res["cohens_d"],
        "reject_null": res["reject_null"],
    }


# -----------------------------------------------------------------------------
# 5. One-Way ANOVA & Categorical Chi-Squared Contingency Tests
# -----------------------------------------------------------------------------

def one_way_anova(
    groups: list[np.ndarray],
    group_names: list[str] | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Execute One-Way Analysis of Variance (ANOVA) and compute effect size eta^2.

    Hypotheses:
    H_0: mu_1 = mu_2 = ... = mu_k  (all group population means are equal)
    H_1: At least one group population mean differs.

    Partitions total sum of squares: SS_total = SS_between + SS_within.
    Effect Size: eta^2 = SS_between / SS_total (proportion of variance explained by factor).

    Parameters
    ----------
    groups : list[np.ndarray]
        List of k 1D numeric arrays.
    group_names : list[str] | None
        Optional labels for groups.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict[str, Any]
        Dictionary containing F_statistic, p_value, df_between, df_within,
        SS_between, SS_within, SS_total, eta_squared, and reject_null.
    """
    k = len(groups)
    if k < 2:
        raise ValueError("ANOVA requires at least 2 groups.")

    clean_groups = [np.asarray(g, dtype=np.float64).ravel() for g in groups]
    for g in clean_groups:
        if len(g) < 2:
            raise ValueError("Every group must contain at least 2 samples.")

    all_data = np.concatenate(clean_groups)
    grand_mean = float(np.mean(all_data))
    N = len(all_data)

    ss_between = float(np.sum([len(g) * (np.mean(g) - grand_mean)**2 for g in clean_groups]))
    ss_within = float(np.sum([np.sum((g - np.mean(g))**2) for g in clean_groups]))
    ss_total = ss_between + ss_within

    df_between = k - 1
    df_within = N - k

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    f_stat = ms_between / ms_within
    p_val = float(1.0 - stats.f.cdf(f_stat, df_between, df_within))

    eta_squared = ss_between / ss_total if ss_total > 0 else 0.0

    return {
        "k_groups": k,
        "n_total": N,
        "group_means": [float(np.mean(g)) for g in clean_groups],
        "grand_mean": grand_mean,
        "F_statistic": float(f_stat),
        "p_value": p_val,
        "df_between": df_between,
        "df_within": df_within,
        "ss_between": ss_between,
        "ss_within": ss_within,
        "ss_total": ss_total,
        "ms_between": ms_between,
        "ms_within": ms_within,
        "eta_squared": float(eta_squared),
        "alpha": alpha,
        "reject_null": bool(p_val < alpha),
    }


def chi_squared_contingency_test(
    contingency_table: np.ndarray | pd.DataFrame,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Execute Pearson's Chi-Squared Contingency Test of Independence with Cramér's V.

    Hypotheses:
    H_0: Row factor and Column factor are statistically independent.
    H_1: There is a statistically significant association between the two factors.

    Assumption Check:
    Verifies that expected cell counts E_ij >= 5.0 (Cochran's condition).

    Effect Size:
    Cramer's V = sqrt( chi^2 / (N * min(r-1, c-1)) ).

    Parameters
    ----------
    contingency_table : np.ndarray | pd.DataFrame
        2D matrix of observed counts O_ij.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    dict[str, Any]
        Dictionary with chi2_statistic, p_value, df, expected_frequencies,
        cramers_v, assumption_satisfied, and reject_null.
    """
    table = np.asarray(contingency_table, dtype=np.float64)
    if table.ndim != 2:
        raise ValueError(f"Contingency table must be 2D; got shape {table.shape}.")

    r, c = table.shape
    if r < 2 or c < 2:
        raise ValueError("Contingency table must be at least 2x2.")

    chi2_stat, p_val, dof, expected = stats.chi2_contingency(table, correction=False)

    N = float(np.sum(table))
    min_dim = min(r - 1, c - 1)
    cramers_v = float(np.sqrt(chi2_stat / (N * min_dim))) if (N * min_dim) > 0 else 0.0

    # Cochran assumption: all expected counts should be >= 5
    min_expected = float(np.min(expected))
    assumption_satisfied = bool(min_expected >= 5.0)

    return {
        "observed_table": table,
        "expected_table": expected,
        "min_expected_count": min_expected,
        "assumption_satisfied": assumption_satisfied,
        "chi2_statistic": float(chi2_stat),
        "p_value": float(p_val),
        "df": dof,
        "cramers_v": cramers_v,
        "alpha": alpha,
        "reject_null": bool(p_val < alpha),
    }


# -----------------------------------------------------------------------------
# 6. Assumption Diagnostics (Normality, Homoscedasticity, Independence)
# -----------------------------------------------------------------------------

def diagnose_assumptions(
    data: np.ndarray,
    groups: list[np.ndarray] | None = None,
) -> dict[str, Any]:
    """Diagnose parametric assumptions: Normality, Equality of Variances, and Serial Independence.

    1. Normality:
       - D'Agostino-Pearson omnibus test: K^2 test combining skewness and kurtosis.
       - Shapiro-Wilk test (if n <= 5000).
    2. Homoscedasticity (if groups provided):
       - Levene's test (robust to non-normal distributions).
       - Bartlett's test.
    3. Serial Independence:
       - Durbin-Watson statistic (tests for first-order autocorrelation).

    Parameters
    ----------
    data : np.ndarray
        1D array of sample or residual values.
    groups : list[np.ndarray] | None
        Optional list of groups to test homoscedasticity.

    Returns
    -------
    dict[str, Any]
        Structured dictionary of diagnostic test statistics and p-values.
    """
    x = np.asarray(data, dtype=np.float64).ravel()
    n = len(x)

    # 1. Normality
    skewness = float(stats.skew(x))
    kurt = float(stats.kurtosis(x))  # excess kurtosis (normal = 0)
    norm_stat, norm_pval = stats.normaltest(x)

    shapiro_pval = None
    if n <= 5000:
        _, shapiro_pval = stats.shapiro(x)
        shapiro_pval = float(shapiro_pval)

    # 2. Homoscedasticity across groups
    levene_pval = None
    bartlett_pval = None
    if groups is not None and len(groups) >= 2:
        clean_g = [np.asarray(g, dtype=np.float64).ravel() for g in groups]
        _, lev_p = stats.levene(*clean_g)
        levene_pval = float(lev_p)
        try:
            _, bart_p = stats.bartlett(*clean_g)
            bartlett_pval = float(bart_p)
        except Exception:
            bartlett_pval = None

    # 3. Serial Autocorrelation
    dw = float(np.sum(np.diff(x)**2) / (np.sum(x**2) + 1e-12))

    return {
        "n_samples": n,
        "skewness": skewness,
        "excess_kurtosis": kurt,
        "dagostino_pvalue": float(norm_pval),
        "is_normal_05": bool(norm_pval >= 0.05),
        "shapiro_pvalue": shapiro_pval,
        "levene_pvalue": levene_pval,
        "is_homoscedastic_05": bool(levene_pval >= 0.05) if levene_pval is not None else None,
        "durbin_watson": dw,
        "has_serial_correlation": bool(abs(dw - 2.0) > 0.5),
    }


# -----------------------------------------------------------------------------
# 7. Type I & Type II Errors and Statistical Power Simulations
# -----------------------------------------------------------------------------

def simulate_power_curve(
    effect_sizes: list[float] | None = None,
    sample_sizes: list[int] | None = None,
    alpha: float = 0.05,
    n_simulations: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """Simulate statistical power (1 - beta) as a function of sample size n and effect size Cohen's d.

    Empirically illustrates:
    - Type I error rate under H_0 (d = 0.0): P(Reject H_0 | H_0 is true) == alpha (5%).
    - Type II error rate under H_1 (d > 0.0): P(Fail to reject H_0 | H_1 is true) == beta.
    - Power = 1 - beta: Probability of correctly rejecting false null hypothesis.

    Parameters
    ----------
    effect_sizes : list[float] | None
        List of Cohen's d values (default: [0.0, 0.2, 0.5, 0.8]).
        (0.0 = Null is True, 0.2 = small, 0.5 = medium, 0.8 = large).
    sample_sizes : list[int] | None
        List of sample sizes n (default: [10, 25, 50, 100, 200]).
    alpha : float, default=0.05
        Significance level.
    n_simulations : int, default=1000
        Number of Monte Carlo trials per condition.
    seed : int, default=42
        Random seed.

    Returns
    -------
    pd.DataFrame
        Table with Effect_Size, Sample_Size, Empirical_Power, Type_I_Error, and Type_II_Error.
    """
    rng = np.random.default_rng(seed)
    effects = effect_sizes or [0.0, 0.2, 0.5, 0.8]
    sizes = sample_sizes or [10, 25, 50, 100, 200]

    records = []
    for d in effects:
        for n in sizes:
            # Under H_1: X ~ N(d, 1.0)
            # Under H_0: mu_0 = 0.0
            data_matrix = rng.normal(loc=d, scale=1.0, size=(n_simulations, n))
            # Perform two-sided t-test against popmean = 0.0
            sample_means = np.mean(data_matrix, axis=1)
            sample_stds = np.std(data_matrix, axis=1, ddof=1)
            t_stats = sample_means / (sample_stds / np.sqrt(n))
            df = n - 1

            p_values = 2.0 * (1.0 - stats.t.cdf(np.abs(t_stats), df=df))
            rejections = np.sum(p_values < alpha)
            power_or_type1 = float(rejections / n_simulations)

            if np.isclose(d, 0.0):
                type1_err = power_or_type1
                power = None
                type2_err = None
            else:
                type1_err = None
                power = power_or_type1
                type2_err = 1.0 - power

            records.append({
                "Effect_Size_d": d,
                "Sample_Size_n": n,
                "Empirical_Power": round(power, 4) if power is not None else None,
                "Type_I_Error": round(type1_err, 4) if type1_err is not None else None,
                "Type_II_Error": round(type2_err, 4) if type2_err is not None else None,
            })

    return pd.DataFrame(records)
