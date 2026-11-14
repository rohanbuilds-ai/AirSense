import React from 'react';
import { Layers, Shuffle, GitCommit, Split, ArrowDownToLine, Activity } from 'lucide-react';

export interface WorkflowStep {
  id: number;
  title: string;
  shortTitle: string;
  unit: string;
  icon: React.ReactNode;
  summary: string;
}

export const WORKFLOW_STEPS: WorkflowStep[] = [
  {
    id: 1,
    title: '1. Real Data Matrix Representation',
    shortTitle: '1. Data Matrix',
    unit: 'Unit 1',
    icon: <Layers className="w-4 h-4" />,
    summary: 'Design matrix X in R^(m x d) & target y from UCI Air Quality',
  },
  {
    id: 2,
    title: '2. Gaussian Elimination & LU',
    shortTitle: '2. Elimination & LU',
    unit: 'Unit 1',
    icon: <Shuffle className="w-4 h-4" />,
    summary: 'Row operations, canonical RREF, and PA = LU factorization',
  },
  {
    id: 3,
    title: '3. Vector Spaces & Subspaces',
    shortTitle: '3. Subspaces & Rank',
    unit: 'Unit 1',
    icon: <GitCommit className="w-4 h-4" />,
    summary: 'Column space C(A), null space N(A), & Rank-Nullity theorem',
  },
  {
    id: 4,
    title: '4. Gram-Schmidt Orthogonalization',
    shortTitle: '4. Gram-Schmidt',
    unit: 'Unit 2',
    icon: <Split className="w-4 h-4" />,
    summary: 'Classical vs Modified Gram-Schmidt numerical stability comparison',
  },
  {
    id: 5,
    title: '5. Projections & Least Squares',
    shortTitle: '5. Least Squares',
    unit: 'Unit 2',
    icon: <ArrowDownToLine className="w-4 h-4" />,
    summary: 'QR regression, orthogonal projection P, & condition number squaring',
  },
  {
    id: 6,
    title: '6. Spectral Decomposition & PCA',
    shortTitle: '6. Spectral & PCA',
    unit: 'Unit 2',
    icon: <Activity className="w-4 h-4" />,
    summary: 'Covariance matrix S, Jacobi eigenvalues, & variance explained',
  },
];

interface StepNavigationProps {
  activeStep: number;
  onSelectStep: (stepId: number) => void;
}

export const StepNavigation: React.FC<StepNavigationProps> = ({
  activeStep,
  onSelectStep,
}) => {
  return (
    <div className="bg-white border border-slate-200/90 rounded-2xl p-2.5 shadow-xs">
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-2">
        {WORKFLOW_STEPS.map((step) => {
          const isActive = activeStep === step.id;
          return (
            <button
              key={step.id}
              onClick={() => onSelectStep(step.id)}
              className={`text-left p-3 rounded-xl transition-all relative flex flex-col justify-between ${
                isActive
                  ? 'bg-slate-900 text-white shadow-md ring-1 ring-slate-800'
                  : 'bg-slate-50/70 hover:bg-slate-100 text-slate-700 hover:text-slate-900 border border-slate-200/60'
              }`}
            >
              <div className="flex items-center justify-between w-full mb-2">
                <span
                  className={`p-1.5 rounded-lg ${
                    isActive ? 'bg-teal-500/20 text-teal-300' : 'bg-white text-slate-500 border border-slate-200/70'
                  }`}
                >
                  {step.icon}
                </span>
                <span
                  className={`text-[9px] font-mono px-1.5 py-0.5 rounded ${
                    isActive
                      ? 'bg-slate-800 text-teal-400 border border-teal-500/30'
                      : 'bg-white text-slate-500 border border-slate-200'
                  }`}
                >
                  {step.unit}
                </span>
              </div>

              <div>
                <div
                  className={`text-xs font-bold leading-snug line-clamp-1 ${
                    isActive ? 'text-white' : 'text-slate-800'
                  }`}
                >
                  {step.shortTitle}
                </div>
                <div
                  className={`text-[10px] mt-1 line-clamp-1 ${
                    isActive ? 'text-slate-300' : 'text-slate-500'
                  }`}
                >
                  {step.summary}
                </div>
              </div>

              {isActive && (
                <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-6 h-1 bg-teal-400 rounded-full" />
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
};
