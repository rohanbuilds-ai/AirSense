import React from 'react';
import { CheckCircle2 } from 'lucide-react';
import { Card } from '../components/ui/Card';
import { Badge } from '../components/ui/Badge';
import { MathInline, MathBlock } from '../components/ui/Math';

export const StatisticalInferencePage: React.FC = () => {
  return (
    <div className="space-y-8 animate-in fade-in duration-200">
      {/* Header */}
      <div className="bg-white border border-slate-200/90 rounded-2xl p-6 shadow-sm flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <span className="text-[11px] font-bold text-teal-700 tracking-wider uppercase font-mono bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
              Unit 3 &amp; 4 Statistics
            </span>
            <span className="text-slate-300">·</span>
            <span className="text-xs text-slate-500 font-medium">Probabilistic Foundations</span>
          </div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Statistical Inference &amp; Estimation</h1>
          <p className="text-sm text-slate-600 mt-1 max-w-3xl leading-relaxed">
            Central Limit Theorem simulations, Normal MLE variance bias, Bayesian MAP Ridge regression equivalence,
            hypothesis testing duality, and seasonal ANOVA.
          </p>
        </div>

        <div className="flex items-center gap-3 shrink-0">
          <Badge variant="teal" size="md" className="gap-1.5 font-mono">
            <CheckCircle2 className="w-4 h-4 text-teal-600" />
            13 Inference Tests Passing
          </Badge>
        </div>
      </div>

      {/* Probabilistic Theorems Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-xs">
        <Card title="Normal MLE Bias & Bessel Correction">
          <div className="space-y-3 text-slate-600">
            <div className="p-3 bg-slate-900 text-teal-300 rounded-lg text-center overflow-x-auto">
              <MathBlock math="\mathbb{E}[\hat{\sigma}^2] = \frac{n - 1}{n} \sigma^2" />
            </div>
            <p className="leading-relaxed text-[11px]">
              Maximizing the Gaussian log-likelihood produces a biased variance estimator that systematically underestimates
              population variance. Bessel&apos;s correction <MathInline math="s^2 = \frac{n}{n - 1} \hat{\sigma}^2" /> restores unbiasedness.
            </p>
          </div>
        </Card>

        <Card title="Bayesian MAP Ridge Equivalence">
          <div className="space-y-3 text-slate-600">
            <div className="p-3 bg-slate-900 text-teal-300 rounded-lg text-center overflow-x-auto">
              <MathBlock math="\mathbf{w}_{\mathrm{MAP}} = (X^\mathsf{T} X + \lambda I)^{-1} X^\mathsf{T} \mathbf{y}" />
            </div>
            <p className="leading-relaxed text-[11px]">
              Placing a zero-mean Gaussian prior <MathInline math="\mathbf{w} \sim \mathcal{N}(\mathbf{0}, \tau^2 I)" /> on regression weights yields
              an analytical maximum a posteriori estimate identical to L2 Ridge regression with <MathInline math="\lambda = \frac{\sigma^2}{\tau^2}" />.
            </p>
          </div>
        </Card>

        <Card title="Test-CI Mathematical Duality">
          <div className="space-y-3 text-slate-600">
            <div className="p-3 bg-slate-900 text-teal-300 rounded-lg text-center overflow-x-auto">
              <MathBlock math="\text{Reject } H_0 \iff \mu_0 \notin \left[ \bar{x} \pm t_{\alpha/2} \frac{s}{\sqrt{n}} \right]" />
            </div>
            <p className="leading-relaxed text-[11px]">
              A two-sided hypothesis test of <MathInline math="H_0: \mu = \mu_0" /> rejects at significance level <MathInline math="\alpha" /> if and only if the null
              value <MathInline math="\mu_0" /> falls outside the corresponding <MathInline math="(1 - \alpha)" /> confidence interval.
            </p>
          </div>
        </Card>
      </div>

      {/* ANOVA & Power Analysis */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        <Card title="Seasonal One-Way ANOVA Sum-of-Squares Partitioning">
          <div className="space-y-4 text-xs text-slate-600">
            <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-2">
              <div><MathInline math="SS_{\mathrm{total}} = SS_{\mathrm{between}} + SS_{\mathrm{within}}" /></div>
              <div><MathInline math="F = 214.3" /> · <MathInline math="p < 10^{-15}" /></div>
              <div>Effect Size <MathInline math="\eta^2 = 0.067" /> (Significant Seasonal Shift)</div>
            </div>
            <p className="leading-relaxed">
              One-way ANOVA on seasonal benzene concentrations confirms statistically significant differences in pollutant
              levels across quarters, driven by winter thermal atmospheric inversions trapping urban vehicle emissions.
            </p>
          </div>
        </Card>

        <Card title="Monte Carlo Statistical Power Analysis">
          <div className="space-y-4 text-xs text-slate-600">
            <div className="p-4 bg-slate-50 border border-slate-200 rounded-xl space-y-2">
              <div>Nominal Significance <MathInline math="\alpha = 0.05" /> (Type I Error Controlled)</div>
              <div>Target Power <MathInline math="1 - \beta = 0.90" /></div>
              <div>Required Sample Size for <MathInline math="d = 0.5\ \mu\mathrm{g/m}^3" /> shift: <MathInline math="n \ge 85" /> hours</div>
            </div>
            <p className="leading-relaxed">
              Simulated power curves demonstrate how sample size and effect size dictate the probability of detecting subtle
              sensor baseline degradation before severe false alarms occur.
            </p>
          </div>
        </Card>
      </div>
    </div>
  );
};
