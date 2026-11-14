import React from 'react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'outline' | 'teal' | 'navy';
  size?: 'sm' | 'md';
  dot?: boolean;
}

export const Badge: React.FC<BadgeProps> = ({
  className,
  variant = 'default',
  size = 'md',
  dot = false,
  children,
  ...props
}) => {
  const baseStyles = 'inline-flex items-center font-medium rounded-full transition-colors whitespace-nowrap';

  const sizeStyles = {
    sm: 'text-[11px] px-2 py-0.5 gap-1.5',
    md: 'text-xs px-2.5 py-1 gap-1.5',
  };

  const variantStyles = {
    default: 'bg-slate-100 text-slate-700 border border-slate-200/90',
    teal: 'bg-teal-50 text-teal-800 border border-teal-200/90',
    info: 'bg-sky-50 text-sky-800 border border-sky-200/90',
    success: 'bg-emerald-50 text-emerald-800 border border-emerald-200/90',
    warning: 'bg-amber-50 text-amber-900 border border-amber-200/90',
    danger: 'bg-rose-50 text-rose-800 border border-rose-200/90',
    outline: 'border border-slate-300 text-slate-700 bg-white/50 backdrop-blur-xs',
    navy: 'bg-slate-900 text-slate-200 border border-slate-700',
  };

  const dotColors = {
    default: 'bg-slate-400',
    teal: 'bg-teal-500',
    info: 'bg-sky-500',
    success: 'bg-emerald-500',
    warning: 'bg-amber-500',
    danger: 'bg-rose-500',
    outline: 'bg-slate-400',
    navy: 'bg-teal-400',
  };

  return (
    <span
      className={twMerge(clsx(baseStyles, sizeStyles[size], variantStyles[variant], className))}
      {...props}
    >
      {dot && <span className={clsx('w-1.5 h-1.5 rounded-full shrink-0', dotColors[variant])} />}
      {children}
    </span>
  );
};
