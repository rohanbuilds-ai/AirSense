import React, { useState } from 'react';
import { BookOpen, Target, Cpu, Lightbulb, ChevronDown, ChevronUp } from 'lucide-react';
import { MathInline } from '../ui/Math';

interface MathConceptCardProps {
  concept: React.ReactNode;
  purpose: React.ReactNode;
  operation: React.ReactNode;
  airsenseInterpretation: React.ReactNode;
  formula?: React.ReactNode;
  defaultOpen?: boolean;
}

export const MathConceptCard: React.FC<MathConceptCardProps> = ({
  concept,
  purpose,
  operation,
  airsenseInterpretation,
  formula,
  defaultOpen = true,
}) => {
  const [isOpen, setIsOpen] = useState(defaultOpen);

  return (
    <div className="bg-white border border-slate-200/90 rounded-2xl shadow-xs overflow-hidden transition-all">
      {/* Header bar */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-5 py-3.5 bg-slate-50/80 hover:bg-slate-100/80 border-b border-slate-200/80 flex items-center justify-between text-left transition-colors"
      >
        <div className="flex items-center gap-2.5">
          <span className="p-1.5 rounded-lg bg-teal-50 text-teal-700 border border-teal-200">
            <BookOpen className="w-4 h-4" />
          </span>
          <div>
            <span className="text-xs font-bold text-slate-900 tracking-tight">
              Mathematical Foundation &amp; Pedagogical Theory
            </span>
            <span className="text-[11px] text-slate-500 block">
              Concept &middot; Purpose &middot; Operation &middot; Calibration Interpretation
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {formula && (
            <span className="hidden sm:inline-block font-mono text-[11px] px-2.5 py-1 rounded bg-slate-900 text-teal-300 font-medium">
              {typeof formula === 'string' && (formula.includes('\\') || formula.includes('^') || formula.includes('_')) ? (
                <MathInline math={formula} />
              ) : (
                formula
              )}
            </span>
          )}
          <span className="p-1 rounded-md text-slate-400 hover:text-slate-600">
            {isOpen ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </span>
        </div>
      </button>

      {/* Expandable Content Grid */}
      {isOpen && (
        <div className="p-5 grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Concept */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/70 space-y-2">
            <div className="flex items-center gap-2 text-xs font-bold text-slate-900">
              <span className="p-1 rounded bg-indigo-50 text-indigo-700">
                <BookOpen className="w-3.5 h-3.5" />
              </span>
              1. Mathematical Concept
            </div>
            <p className="text-xs text-slate-600 leading-relaxed font-sans">
              {concept}
            </p>
          </div>

          {/* Purpose */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/70 space-y-2">
            <div className="flex items-center gap-2 text-xs font-bold text-slate-900">
              <span className="p-1 rounded bg-teal-50 text-teal-700">
                <Target className="w-3.5 h-3.5" />
              </span>
              2. Scientific Purpose
            </div>
            <p className="text-xs text-slate-600 leading-relaxed font-sans">
              {purpose}
            </p>
          </div>

          {/* Operation */}
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200/70 space-y-2">
            <div className="flex items-center gap-2 text-xs font-bold text-slate-900">
              <span className="p-1 rounded bg-emerald-50 text-emerald-700">
                <Cpu className="w-3.5 h-3.5" />
              </span>
              3. Algorithmic Operation
            </div>
            <p className="text-xs text-slate-600 leading-relaxed font-sans">
              {operation}
            </p>
          </div>

          {/* AirSense Interpretation */}
          <div className="p-4 rounded-xl bg-amber-50/50 border border-amber-200/70 space-y-2">
            <div className="flex items-center gap-2 text-xs font-bold text-amber-950">
              <span className="p-1 rounded bg-amber-100 text-amber-800">
                <Lightbulb className="w-3.5 h-3.5" />
              </span>
              4. Interpretation for AirSense Calibration
            </div>
            <p className="text-xs text-amber-900 leading-relaxed font-sans">
              {airsenseInterpretation}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
