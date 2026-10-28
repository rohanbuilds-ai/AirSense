# AirSense: Statistical Methods, Probability & Inference Manual

This document provides theoretical foundations, derivations, formulas, and university viva explanations for **Unit 3 (Probability & Estimation)** and **Unit 4 (Statistical Inference & Hypothesis Testing)** as integrated into the **AirSense Laboratory**.

---

## Unit 3: Probability, Random Variables & Estimation

### 1. Population, Sample & The Sampling Distribution
* **Target Population**: All theoretical hourly operating states of roadside chemical gas sensor arrays under atmospheric exposure.
* **Empirical Sample**: The finite set of $n = 8,991$ hourly observations recorded between March 2004 and April 2005.
* **Sampling Distribution**: The probability distribution of a sample statistic (e.g., sample mean $\bar{X}_n$) over hypothetical repeated realizations of size $n$ drawn from the population.

### 2. The Central Limit Theorem (CLT)
Let $X_1, X_2, \dots, X_n$ be independent, identically distributed (i.i.d.) random variables with finite population mean $\mu$ and finite variance $\sigma^2 > 0$. Then the standardized sample mean converges in distribution to the standard normal distribution as $n \to \infty$:
$$Z_n = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1)$$

* **Practical Rule of Thumb**: For sample sizes $n \ge 30$, the sampling distribution of $\bar{X}_n$ is approximately normal $\mathcal{N}(\mu, \sigma^2/n)$, even when the underlying parent distribution is highly skewed (e.g. exponential or lognormal sensor concentrations).

### 3. Point Estimation & Mean Squared Error (MSE)
For any point estimator $\hat{\theta}$ of an unknown population parameter $\theta$:
* **Bias**: $\text{Bias}(\hat{\theta}) = \mathbb{E}[\hat{\theta}] - \theta$
* **Variance**: $\text{Var}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \mathbb{E}[\hat{\theta}])^2]$
* **Mean Squared Error Decomposition**:
  $$\text{MSE}(\hat{\theta}) = \mathbb{E}[(\hat{\theta} - \theta)^2] = \text{Bias}(\hat{\theta})^2 + \text{Var}(\hat{\theta})$$

### 4. Maximum Likelihood Estimation (MLE) for the Normal Distribution
Given an i.i.d. sample $x_1, \dots, x_n \sim \mathcal{N}(\mu, \sigma^2)$, the log-likelihood function is:
$$\ln L(\mu, \sigma^2) = -\frac{n}{2} \ln(2\pi) - \frac{n}{2} \ln(\sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (x_i - \mu)^2$$

Setting the partial derivatives to zero yields:
1. **MLE for the Mean** (Unbiased):
   $$\frac{\partial \ln L}{\partial \mu} = \frac{1}{\sigma^2}\sum_{i=1}^n (x_i - \mu) = 0 \implies \hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i = \bar{x}$$
   $$\mathbb{E}[\hat{\mu}_{\text{MLE}}] = \mu \implies \text{Bias}(\hat{\mu}_{\text{MLE}}) = 0$$

2. **MLE for the Variance** (Biased):
   $$\frac{\partial \ln L}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2(\sigma^2)^2}\sum_{i=1}^n (x_i - \bar{x})^2 = 0 \implies \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2$$
   $$\mathbb{E}[\hat{\sigma}^2_{\text{MLE}}] = \frac{n-1}{n}\sigma^2 \ne \sigma^2 \implies \text{Bias}(\hat{\sigma}^2_{\text{MLE}}) = -\frac{1}{n}\sigma^2$$
   To correct this downward bias, we apply **Bessel's correction**, dividing by $n - 1$ degrees of freedom to obtain the unbiased sample variance $s^2$:
   $$s^2 = \frac{1}{n-1}\sum_{i=1}^n (x_i - \bar{x})^2, \quad \mathbb{E}[s^2] = \sigma^2$$

### 5. Maximum A Posteriori (MAP) Estimation: The Bayesian-Ridge Bridge
In linear regression, consider the Bayesian formulation:
* **Likelihood**: $\mathbf{y} \mid X, \mathbf{w} \sim \mathcal{N}(X\mathbf{w}, \sigma^2 I)$
  $$p(\mathbf{y} \mid X, \mathbf{w}) \propto \exp\left( -\frac{1}{2\sigma^2} \|\mathbf{y} - X\mathbf{w}\|_2^2 \right)$$
* **Conjugate Prior**: Zero-mean isotropic Gaussian prior $\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)$
  $$p(\mathbf{w}) \propto \exp\left( -\frac{1}{2\tau^2} \|\mathbf{w}\|_2^2 \right)$$

By Bayes' theorem, the posterior log-density is:
$$\ln p(\mathbf{w} \mid X, \mathbf{y}) \propto \ln p(\mathbf{y} \mid X, \mathbf{w}) + \ln p(\mathbf{w}) = -\frac{1}{2\sigma^2} \|\mathbf{y} - X\mathbf{w}\|_2^2 - \frac{1}{2\tau^2} \|\mathbf{w}\|_2^2 + \text{const}$$

Maximizing the posterior is equivalent to minimizing:
$$\arg\max_{\mathbf{w}} \ln p(\mathbf{w} \mid X, \mathbf{y}) \equiv \arg\min_{\mathbf{w}} \left( \|\mathbf{y} - X\mathbf{w}\|_2^2 + \frac{\sigma^2}{\tau^2} \|\mathbf{w}\|_2^2 \right)$$
Setting $\lambda = \frac{\sigma^2}{\tau^2}$, this is **analytically identical to L2 Ridge Regularization**:
$$\mathbf{w}_{\text{MAP}} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}, \quad \text{where } \lambda = \frac{\sigma^2}{\tau^2}$$
* When prior variance $\tau^2 \to \infty$ ($\lambda \to 0$, uninformative prior), $\mathbf{w}_{\text{MAP}} \to \mathbf{w}_{\text{OLS}}$.
* When prior variance $\tau^2 \to 0$ ($\lambda \to \infty$, strong shrinkage), $\mathbf{w}_{\text{MAP}} \to \mathbf{0}$.

### 6. Confidence Intervals & Margin of Error (MOE)
For a population mean $\mu$ at confidence level $1 - \alpha$:
* **Large Samples ($n \ge 30$, Normal $z$-method)**:
  $$\text{CI}_{1-\alpha} = \left[ \bar{x} - z_{\alpha/2} \frac{s}{\sqrt{n}}, \quad \bar{x} + z_{\alpha/2} \frac{s}{\sqrt{n}} \right]$$
* **Small Samples ($n < 30$, Student's $t$-method with $df = n - 1$)**:
  $$\text{CI}_{1-\alpha} = \left[ \bar{x} - t_{\alpha/2, n-1} \frac{s}{\sqrt{n}}, \quad \bar{x} + t_{\alpha/2, n-1} \frac{s}{\sqrt{n}} \right]$$

#### Margin of Error Scaling:
$$\text{MOE} = t_{\alpha/2} \frac{s}{\sqrt{n}} \propto \frac{1}{\sqrt{n}}$$
**The Quadrupling Law**: Because precision scales with $\sqrt{n}$, cutting the margin of error in half ($2\times$ precision) requires **$4\times$ the sample size**.

---

## Unit 4: Hypothesis Testing & Inferential Decision Making

### 1. The Decision-Theoretic Framework

| Reality $\downarrow$ \ Decision $\rightarrow$ | Do Not Reject $H_0$ | Reject $H_0$ (Claim Effect) |
| :--- | :--- | :--- |
| **$H_0$ is True** (No effect) | Correct Decision ($1 - \alpha$) | **Type I Error ($\alpha$)** (False Positive) |
| **$H_0$ is False** (Real effect) | **Type II Error ($\beta$)** (False Negative) | **Statistical Power ($1 - \beta$)** (True Positive) |

* **Significance Level ($\alpha$)**: $P(\text{Reject } H_0 \mid H_0 \text{ is true})$, fixed a priori (typically $\alpha = 0.05$ or $0.01$).
* **Statistical Power ($1 - \beta$)**: The probability of correctly rejecting a false null hypothesis. Power increases monotonically with:
  1. Larger sample size $n$.
  2. Larger effect size (Cohen's $d$).
  3. Lower observation noise $\sigma$.
  4. Higher allowed significance level $\alpha$.

### 2. Duality Between Two-Sided Hypothesis Tests & Confidence Intervals
**Theorem**: A two-sided hypothesis test of $H_0: \mu = \mu_0$ versus $H_1: \mu \ne \mu_0$ rejects $H_0$ at significance level $\alpha$ **if and only if** the hypothesized value $\mu_0$ falls strictly outside the $(1 - \alpha)$ Confidence Interval:
$$\text{Reject } H_0 \text{ at level } \alpha \iff \mu_0 \notin \left[ \bar{x} - t_{\alpha/2} \frac{s}{\sqrt{n}}, \quad \bar{x} + t_{\alpha/2} \frac{s}{\sqrt{n}} \right]$$

### 3. One-Way Analysis of Variance (ANOVA)
Tests equality of $k \ge 2$ independent group means:
$$H_0: \mu_1 = \mu_2 = \cdots = \mu_k \quad \text{vs.} \quad H_1: \text{At least one mean differs}$$

Partitions total variation via the Fundamental ANOVA Identity:
$$SS_{\text{total}} = SS_{\text{between}} + SS_{\text{within}}$$
$$SS_{\text{between}} = \sum_{j=1}^k n_j (\bar{x}_j - \bar{x}_{\text{grand}})^2, \quad df_B = k - 1$$
$$SS_{\text{within}} = \sum_{j=1}^k \sum_{i=1}^{n_j} (x_{ij} - \bar{x}_j)^2, \quad df_W = N - k$$

The $F$-statistic is the ratio of explained variance to unexplained variance:
$$F = \frac{MS_{\text{between}}}{MS_{\text{within}}} = \frac{SS_B / (k - 1)}{SS_W / (N - k)} \sim F(k - 1, N - k)$$

* **Effect Size ($\eta^2$, Eta-Squared)**:
  $$\eta^2 = \frac{SS_{\text{between}}}{SS_{\text{total}}}$$
  Quantifies the fraction of total variance attributable to group membership ($0 \le \eta^2 \le 1$).

### 4. Pearson's Chi-Squared Contingency Test of Independence
Evaluates categorical associations across an $r \times c$ contingency table:
$$H_0: \text{Row factor and Column factor are statistically independent}$$

Under $H_0$, expected cell frequencies are:
$$E_{ij} = \frac{R_i \cdot C_j}{N}$$
The test statistic measures squared deviations:
$$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}} \sim \chi^2_{(r-1)(c-1)}$$

* **Cochran's Assumption Check**: Valid only when all expected cell counts $E_{ij} \ge 5.0$.
* **Effect Size (Cramér's $V$)**:
  $$V = \sqrt{\frac{\chi^2}{N \cdot \min(r - 1, c - 1)}}$$
  Ranges from $0$ (no association) to $1$ (complete deterministic association).

---

## 3. Correct Interpretation of $p$-values (Viva Defense Key)

> [!CAUTION]
> **What a $p$-value IS**:
> The probability of observing a test statistic as extreme as, or more extreme than, the value actually observed, assuming that the null hypothesis $H_0$ is strictly true:
> $$p\text{-value} = P(T \ge t_{\text{obs}} \mid H_0)$$

> [!WARNING]
> **What a $p$-value IS NOT**:
> 1. It is **NOT** the probability that the null hypothesis is true: $p \ne P(H_0 \mid \text{data})$.
> 2. It is **NOT** the probability that the alternative hypothesis is true.
> 3. It does **NOT** measure the magnitude or practical importance of an effect (a tiny, trivial difference can produce $p < 0.001$ with large $n$).
