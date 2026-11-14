import React, { useMemo } from 'react';
import katex from 'katex';

interface MathProps {
  math: string;
  className?: string;
  ariaLabel?: string;
}

/**
 * MathInline: Renders inline LaTeX mathematics with KaTeX.
 * Example: <MathInline math="X \in \mathbb{R}^{n \times 8}" />
 */
export const MathInline: React.FC<MathProps> = ({ math, className = '', ariaLabel }) => {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        displayMode: false,
        throwOnError: false,
        strict: false,
      });
    } catch {
      return math;
    }
  }, [math]);

  return (
    <span
      className={`inline-block align-baseline font-serif ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
      aria-label={ariaLabel || math}
    />
  );
};

/**
 * MathBlock: Renders centered display-mode LaTeX equations with KaTeX.
 * Keeps equations responsive and scrollable on small mobile screens.
 * Example: <MathBlock math="z_{ij} = \frac{x_{ij} - \mu_{j,\mathrm{train}}}{\sigma_{j,\mathrm{train}}}" />
 */
export const MathBlock: React.FC<MathProps> = ({ math, className = '', ariaLabel }) => {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, {
        displayMode: true,
        throwOnError: false,
        strict: false,
      });
    } catch {
      return math;
    }
  }, [math]);

  return (
    <div
      className={`overflow-x-auto py-2 text-center text-slate-800 ${className}`}
      dangerouslySetInnerHTML={{ __html: html }}
      aria-label={ariaLabel || math}
    />
  );
};
