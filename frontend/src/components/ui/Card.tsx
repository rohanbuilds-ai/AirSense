import React from 'react';
import { twMerge } from 'tailwind-merge';

interface CardProps extends Omit<React.HTMLAttributes<HTMLDivElement>, 'title'> {
  title?: React.ReactNode;
  subtitle?: React.ReactNode;
  action?: React.ReactNode;
  hoverable?: boolean;
}

export const Card: React.FC<CardProps> = ({
  className,
  title,
  subtitle,
  action,
  hoverable = false,
  children,
  ...props
}) => {
  return (
    <div
      className={twMerge(
        'bg-white border border-slate-200/90 rounded-2xl shadow-xs transition-all duration-200 overflow-hidden',
        hoverable && 'hover:shadow-md hover:border-slate-300 hover:-translate-y-0.5',
        className
      )}
      {...props}
    >
      {(title || subtitle || action) && (
        <div className="px-6 py-4.5 border-b border-slate-100 flex items-center justify-between gap-4">
          <div>
            {title && (
              <h3 className="font-semibold text-slate-900 text-sm tracking-tight flex items-center gap-2">
                {title}
              </h3>
            )}
            {subtitle && <p className="text-xs text-slate-500 mt-0.5 leading-relaxed">{subtitle}</p>}
          </div>
          {action && <div className="flex items-center gap-2 shrink-0">{action}</div>}
        </div>
      )}
      <div className="p-6">{children}</div>
    </div>
  );
};
