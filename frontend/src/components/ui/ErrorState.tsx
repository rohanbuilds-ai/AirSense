import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';

interface ErrorStateProps {
  title?: string;
  message: string;
  onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
  title = 'Telemetry Gateway Disconnected',
  message,
  onRetry,
}) => {
  return (
    <div className="bg-white border border-rose-200/90 rounded-2xl p-8 text-center max-w-md mx-auto my-12 shadow-sm">
      <div className="inline-flex p-3 bg-rose-50 border border-rose-100 rounded-2xl text-rose-600 mb-4 shadow-2xs">
        <AlertCircle className="w-6 h-6" />
      </div>
      <h3 className="font-bold text-slate-900 text-base mb-1.5">{title}</h3>
      <p className="text-xs text-slate-600 mb-6 font-mono leading-relaxed bg-slate-50 p-3 rounded-xl border border-slate-200/70 text-left overflow-x-auto">
        {message}
      </p>
      {onRetry && (
        <button
          onClick={onRetry}
          type="button"
          className="inline-flex items-center gap-2 px-4 py-2.5 bg-slate-900 hover:bg-slate-800 active:scale-95 text-white rounded-xl text-xs font-semibold transition-all shadow-sm cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Retry Connection</span>
        </button>
      )}
    </div>
  );
};
