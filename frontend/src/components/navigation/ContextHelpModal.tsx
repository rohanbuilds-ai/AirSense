import React from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { X, BookOpen, AlertTriangle } from 'lucide-react';
import { modalBackdropVariants, modalContentVariants } from '../../lib/motion';
import { MathInline } from '../ui/Math';

interface ContextHelpModalProps {
  isOpen: boolean;
  onClose: () => void;
  currentPath: string;
}

const helpContent: Record<string, { title: string; concept: React.ReactNode; purpose: React.ReactNode; outcome: React.ReactNode; caution: React.ReactNode }> = {
  '/': {
    title: 'Overview & Benchmark Evaluation',
    concept: 'Zero-leakage chronological evaluation of full-rank OLS vs. Principal Component Regression (PCR).',
    purpose: 'Understand whether in-sample fit translates into long-term out-of-time calibration stability on field gas sensors.',
    outcome: (
      <span>
        Full OLS was selected on validation data (Val RMSE 2.024 <MathInline math="\mu\mathrm{g/m}^3" />). Post-hoc test comparison shows Compact PCR (<MathInline math="k = 3" />) regularizes against drift (Test RMSE 2.415 <MathInline math="\mu\mathrm{g/m}^3" />).
      </span>
    ),
    caution: (
      <span>
        Do not claim <MathInline math="k = 3" /> is the validation-selected model; it is a post-hoc empirical observation of PCA shrinkage.
      </span>
    ),
  },
  '/data-quality': {
    title: 'Dataset Audit & Missingness Ingestion',
    concept: 'Detection of CSV dialect (semicolon, European comma decimal), replacement of sentinel -200 with IEEE NaN.',
    purpose: 'Identify sensor dropouts, examine benzene target availability, and guarantee zero feature-target leakage.',
    outcome: 'Discovered exactly 366 coincident blackout hours dropping all sensors and target together. Pruning leaves 8,991 usable supervised rows.',
    caution: 'Never include certified analyzer reference columns (GT) as predictive features; they are ground-truth validation targets.',
  },
  '/calibration': {
    title: 'Sensor Calibration & PCR Workspace',
    concept: 'QR factorization via Modified Gram-Schmidt and Principal Component Regression with physical weight back-projection.',
    purpose: 'Interactively configure feature subsets and PCA components to observe changes in validation RMSE and feature attribution.',
    outcome: 'Back-projected physical weights reveal chemical proxy dynamics (e.g. PT08.S2 NMHC proxy dominating target prediction).',
    caution: 'The test partition (1,799 hours) remains frozen during interactive tuning to prevent test data snooping.',
  },
  '/linear-algebra': {
    title: 'Linear Algebra Laboratory',
    concept: 'From-scratch Gaussian elimination with partial pivoting, canonical RREF, Rank-Nullity Theorem, and Modified Gram-Schmidt QR.',
    purpose: 'Prove fundamental linear algebra invariants without relying on high-level library shortcuts.',
    outcome: (
      <span>
        MGS achieves <MathInline math="7.07 \times 10^6\times" /> superior numerical stability over Classical Gram-Schmidt on ill-conditioned sensor matrices.
      </span>
    ),
    caution: (
      <span>
        Normal equations (<MathInline math="X^\mathsf{T} X" />) square the condition number to <MathInline math="\kappa(X^\mathsf{T} X) \approx 3.58 \times 10^8" />; custom QR avoids precision destruction.
      </span>
    ),
  },
  '/pca-explorer': {
    title: 'PCA Explorer & Spectral SVD',
    concept: 'Centred feature matrix SVD equivalence to covariance spectral decomposition, Jacobi eigenvalue algorithm, and Eckart-Young low-rank bounds.',
    purpose: 'Eliminate severe multicollinearity among solid-state metal-oxide sensors without throwing away physical sensor channels.',
    outcome: (
      <span>
        First 3 principal components capture 94.04% of total dataset variance; condition number drops from <MathInline math="1.80 \times 10^2" /> to 14.3.
      </span>
    ),
    caution: 'PCA is unsupervised; components with large variance are not guaranteed to align with the benzene target.',
  },
  '/drift-analysis': {
    title: 'Sensor Drift & Quality Control Dynamics',
    concept: 'Durbin-Watson serial correlation, cumulative sum (CUSUM) change detection, and empirical residual bias progression.',
    purpose: 'Detect when solid-state semiconductor sensors lose calibration fidelity in real-world ambient field deployments.',
    outcome: (
      <span>
        Residual mean bias shifts by -4.13 <MathInline math="\mu\mathrm{g/m}^3" /> over 13 months, with strong serial autocorrelation (<MathInline math="DW = 0.2062" />).
      </span>
    ),
    caution: 'Sensor resistance aging cannot be decoupled from annual ambient weather cycles without sealed zero/span chamber tests.',
  },
  '/statistical-inference': {
    title: 'Statistical Inference & Hypothesis Testing',
    concept: 'MLE normal estimators, Central Limit Theorem Monte Carlo simulations, Bayesian MAP with conjugate priors, and dual-interval hypothesis tests.',
    purpose: 'Bridge textbook probability theory with physical sensor error distributions in an engineering setting.',
    outcome: 'Student-t confidence intervals provide conservative coverage on small calibration batches; normal z-tests apply to operational streams.',
    caution: 'Time-series residual observations violate i.i.d. independence assumptions due to diurnal pollution cycles.',
  },
  '/methodology-viva': {
    title: 'Methodology & University Viva Dossier',
    concept: 'Examiner-focused Concept -> Purpose -> Outcome defense pack and anticipated questions.',
    purpose: 'Equip the researcher to defend mathematical foundations and scientific boundaries before an academic panel.',
    outcome: 'Complete coverage of Units 1 to 4 with zero mathematical gaps.',
    caution: 'Always emphasize academic honesty regarding limitations (effective sample size, confounding, non-linear kinetics).',
  },
};

export const ContextHelpModal: React.FC<ContextHelpModalProps> = ({ isOpen, onClose, currentPath }) => {
  const content = helpContent[currentPath] || helpContent['/'];

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
          <motion.div
            variants={modalBackdropVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            onClick={onClose}
            className="fixed inset-0 bg-slate-950/60 backdrop-blur-xs"
          />

          <motion.div
            variants={modalContentVariants}
            initial="initial"
            animate="animate"
            exit="exit"
            className="bg-white border border-slate-200/90 rounded-2xl shadow-xl max-w-xl w-full p-6 relative z-10 overflow-hidden"
          >
            <div className="flex items-center justify-between pb-4 border-b border-slate-100">
              <div className="flex items-center gap-2.5">
                <div className="p-2 bg-teal-50 text-teal-700 rounded-xl border border-teal-100">
                  <BookOpen className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-bold text-slate-900 text-sm tracking-tight">{content.title}</h3>
                  <p className="text-[11px] text-slate-500 font-medium">Scientific Context &amp; Mathematical Notes</p>
                </div>
              </div>
              <button
                type="button"
                onClick={onClose}
                className="p-1.5 rounded-xl text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors cursor-pointer"
                aria-label="Close notes dialog"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            <div className="py-4 space-y-3.5 text-xs">
              <div>
                <span className="font-bold text-slate-700 uppercase text-[10px] tracking-wider block mb-1 font-mono">
                  Mathematical Concept
                </span>
                <p className="text-slate-600 leading-relaxed bg-slate-50/80 p-3 rounded-xl border border-slate-200/70">
                  {content.concept}
                </p>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                <div>
                  <span className="font-bold text-slate-700 uppercase text-[10px] tracking-wider block mb-1 font-mono">
                    Scientific Purpose
                  </span>
                  <p className="text-slate-600 leading-relaxed bg-slate-50/80 p-3 rounded-xl border border-slate-200/70">
                    {content.purpose}
                  </p>
                </div>
                <div>
                  <span className="font-bold text-slate-700 uppercase text-[10px] tracking-wider block mb-1 font-mono">
                    Empirical Outcome
                  </span>
                  <p className="text-slate-600 leading-relaxed bg-slate-50/80 p-3 rounded-xl border border-slate-200/70">
                    {content.outcome}
                  </p>
                </div>
              </div>

              <div className="p-3.5 bg-amber-50/70 border border-amber-200/80 rounded-xl flex items-start gap-2.5 text-amber-950">
                <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                <div>
                  <span className="font-bold text-[10px] uppercase tracking-wider block mb-0.5 font-mono text-amber-900">
                    Academic Viva Defense Boundary
                  </span>
                  <p className="text-[11px] leading-relaxed text-amber-900">{content.caution}</p>
                </div>
              </div>
            </div>

            <div className="pt-3 border-t border-slate-100 flex justify-end">
              <button
                type="button"
                onClick={onClose}
                className="px-4 py-2 bg-slate-900 hover:bg-slate-800 active:scale-95 text-white rounded-xl text-xs font-semibold transition-all cursor-pointer shadow-sm"
              >
                Dismiss Notes
              </button>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  );
};
