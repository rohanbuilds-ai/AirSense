import React from 'react';
import { Loader2 } from 'lucide-react';

interface LoadingSpinnerProps {
  label?: string;
}

export const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({
  label = 'Loading scientific telemetry...',
}) => {
  return (
    <div className="flex flex-col items-center justify-center p-16 text-center min-h-[320px]">
      <div className="relative flex items-center justify-center mb-4">
        <div className="w-12 h-12 rounded-full border-2 border-teal-500/20 animate-ping absolute" />
        <div className="w-10 h-10 rounded-full border-2 border-teal-500/30 flex items-center justify-center bg-teal-50/50">
          <Loader2 className="w-5 h-5 text-teal-600 animate-spin" />
        </div>
      </div>
      <p className="text-xs font-semibold text-slate-700 tracking-wide uppercase font-mono">{label}</p>
      <p className="text-[11px] text-slate-400 mt-1">Connecting to FastAPI in-memory Python mathematical engine</p>
    </div>
  );
};
