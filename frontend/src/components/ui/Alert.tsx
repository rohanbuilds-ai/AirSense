import React from 'react';
import { AlertCircle, AlertTriangle, CheckCircle2, Info } from 'lucide-react';
import { twMerge } from 'tailwind-merge';

interface AlertProps {
  variant?: 'info' | 'warning' | 'danger' | 'success';
  title?: string;
  children: React.ReactNode;
  className?: string;
}

export const Alert: React.FC<AlertProps> = ({
  variant = 'info',
  title,
  children,
  className,
}) => {
  const icons = {
    info: <Info className="w-4 h-4 text-teal-600 shrink-0 mt-0.5" />,
    warning: <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />,
    danger: <AlertCircle className="w-4 h-4 text-rose-600 shrink-0 mt-0.5" />,
    success: <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />,
  };

  const variantStyles = {
    info: 'bg-teal-50/50 border-teal-200/90 text-teal-950',
    warning: 'bg-amber-50/60 border-amber-200/90 text-amber-950',
    danger: 'bg-rose-50/60 border-rose-200/90 text-rose-950',
    success: 'bg-emerald-50/60 border-emerald-200/90 text-emerald-950',
  };

  return (
    <div
      role="alert"
      className={twMerge(
        'border rounded-xl p-4 flex items-start gap-3 text-xs leading-relaxed transition-all shadow-2xs',
        variantStyles[variant],
        className
      )}
    >
      {icons[variant]}
      <div className="flex-1 min-w-0">
        {title && (
          <h4 className="font-bold text-xs tracking-tight uppercase mb-1 text-slate-900 flex items-center gap-2">
            {title}
          </h4>
        )}
        <div className="text-slate-700 text-xs leading-normal">{children}</div>
      </div>
    </div>
  );
};
