import React from 'react';

interface MatrixViewerProps {
  matrix: number[][];
  rowLabels?: string[];
  colLabels?: string[];
  highlightPivots?: Array<[number, number]>;
  highlightRow?: number | null;
  highlightCol?: number | null;
  title?: React.ReactNode;
  badge?: React.ReactNode;
  precision?: number;
  maxDisplayRows?: number;
  maxDisplayCols?: number;
  caption?: React.ReactNode;
  emptyMessage?: string;
}

export const MatrixViewer: React.FC<MatrixViewerProps> = ({
  matrix,
  rowLabels,
  colLabels,
  highlightPivots = [],
  highlightRow = null,
  highlightCol = null,
  title,
  badge,
  precision = 3,
  maxDisplayRows = 16,
  maxDisplayCols = 8,
  caption,
  emptyMessage = 'No matrix data',
}) => {
  if (!matrix || matrix.length === 0 || !matrix[0]) {
    return (
      <div className="p-4 border border-slate-200 rounded-xl bg-slate-50 text-xs text-slate-500 font-mono text-center">
        {emptyMessage}
      </div>
    );
  }

  const m = matrix.length;
  const n = matrix[0].length;

  const displayRows = matrix.slice(0, maxDisplayRows);
  const displayRowLabels = rowLabels ? rowLabels.slice(0, maxDisplayRows) : undefined;
  const displayColLabels = colLabels ? colLabels.slice(0, maxDisplayCols) : undefined;

  const isPivot = (r: number, c: number) => {
    return highlightPivots.some(([pr, pc]) => pr === r && pc === c);
  };

  const formatVal = (val: number | undefined): string => {
    if (val === undefined || isNaN(val)) return '-';
    // Clean up negative zero
    const rounded = Math.abs(val) < 1e-10 ? 0 : val;
    // For integers or near integers
    if (Math.abs(rounded - Math.round(rounded)) < 1e-6 && Math.abs(rounded) < 1000) {
      return rounded.toFixed(1);
    }
    return rounded.toFixed(precision);
  };

  return (
    <div className="bg-white border border-slate-200/90 rounded-xl p-4 shadow-sm space-y-3">
      {/* Header */}
      {(title || badge) && (
        <div className="flex items-center justify-between pb-2 border-b border-slate-100">
          <div className="flex items-center gap-2">
            {title && <span className="text-xs font-bold text-slate-800">{title}</span>}
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-100 text-slate-600 font-medium">
              {m} &times; {n}
            </span>
          </div>
          {badge && (
            <span className="text-[10px] font-mono font-semibold px-2 py-0.5 rounded bg-teal-50 text-teal-700 border border-teal-200">
              {badge}
            </span>
          )}
        </div>
      )}

      {/* Matrix Body with bracket border styling */}
      <div className="relative overflow-x-auto py-1">
        <div className="inline-block min-w-full">
          <div className="relative border-l-2 border-r-2 border-slate-400 px-3 py-1.5 rounded-sm">
            <table className="border-collapse text-right font-mono text-xs w-full">
              {/* Optional Column Headers */}
              {displayColLabels && (
                <thead>
                  <tr>
                    {displayRowLabels && <th className="p-1.5 text-left text-[10px] text-slate-400 font-sans">#</th>}
                    {displayColLabels.map((col, cIdx) => (
                      <th
                        key={cIdx}
                        className={`p-1.5 text-[10px] font-semibold tracking-tight ${
                          highlightCol === cIdx ? 'text-teal-700 bg-teal-50/60 rounded' : 'text-slate-500'
                        }`}
                      >
                        {col}
                      </th>
                    ))}
                    {n > maxDisplayCols && (
                      <th className="p-1.5 text-[10px] text-slate-400 font-sans">&hellip;</th>
                    )}
                  </tr>
                </thead>
              )}

              <tbody>
                {displayRows.map((row, rIdx) => {
                  const isRowHighlighted = highlightRow === rIdx;
                  return (
                    <tr
                      key={rIdx}
                      className={`transition-colors ${
                        isRowHighlighted ? 'bg-amber-50/70 border-y border-amber-200' : 'hover:bg-slate-50/80'
                      }`}
                    >
                      {/* Row Label */}
                      {displayRowLabels && (
                        <td className="p-1.5 text-left text-[10px] text-slate-400 font-sans whitespace-nowrap">
                          {displayRowLabels[rIdx] || `r_${rIdx + 1}`}
                        </td>
                      )}

                      {/* Cells */}
                      {row.slice(0, maxDisplayCols).map((val, cIdx) => {
                        const pivot = isPivot(rIdx, cIdx);
                        const isColHighlighted = highlightCol === cIdx;
                        return (
                          <td
                            key={cIdx}
                            className={`p-1.5 transition-all text-xs ${
                              pivot
                                ? 'bg-teal-100/90 text-teal-900 font-bold border border-teal-300 rounded shadow-xs'
                                : isRowHighlighted
                                ? 'text-amber-950 font-semibold'
                                : isColHighlighted
                                ? 'bg-teal-50/40 text-slate-900'
                                : Math.abs(val) < 1e-10
                                ? 'text-slate-300'
                                : 'text-slate-700'
                            }`}
                          >
                            {formatVal(val)}
                          </td>
                        );
                      })}

                      {n > maxDisplayCols && (
                        <td className="p-1.5 text-slate-400 text-center">&hellip;</td>
                      )}
                    </tr>
                  );
                })}

                {m > maxDisplayRows && (
                  <tr>
                    <td
                      colSpan={(displayColLabels ? displayColLabels.length : n) + (displayRowLabels ? 1 : 0)}
                      className="p-1.5 text-center text-[10px] text-slate-400 font-sans"
                    >
                      &vellip; ({m - maxDisplayRows} additional rows truncated for display)
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Caption / note */}
      {caption && (
        <p className="text-[11px] text-slate-500 font-sans pt-1 border-t border-slate-100 leading-tight">
          {caption}
        </p>
      )}
    </div>
  );
};
