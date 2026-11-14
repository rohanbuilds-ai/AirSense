import React from 'react';
import { Badge } from './Badge';

interface MetricCardProps {
  label: React.ReactNode;
  value: React.ReactNode;
  unit?: React.ReactNode;
  badge?: {
    text: string;
    variant?: 'default' | 'success' | 'warning' | 'danger' | 'info' | 'teal';
  };
  subtitle?: React.ReactNode;
  secondaryValue?: {
    label: string;
    value: React.ReactNode;
  };
  icon?: React.ReactNode;
  trend?: {
    direction: 'up' | 'down' | 'neutral';
    text: string;
  };
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  unit,
  badge,
  subtitle,
  secondaryValue,
  icon,
}) => {
  return (
    <div className="bg-white border border-slate-200/90 rounded-2xl p-5 shadow-xs flex flex-col justify-between hover:border-slate-300 hover:shadow-sm hover:-translate-y-0.5 transition-all duration-200 group">
      <div>
        <div className="flex items-start justify-between gap-3 mb-2">
          <span className="text-[11px] font-bold text-slate-500 uppercase tracking-wider font-mono">
            {label}
          </span>
          {icon && (
            <div className="text-slate-400 p-1.5 bg-slate-50 border border-slate-100 rounded-lg group-hover:text-teal-600 group-hover:bg-teal-50/50 group-hover:border-teal-100 transition-colors">
              {icon}
            </div>
          )}
        </div>

        <div className="flex items-baseline gap-1.5 my-1.5">
          <span className="text-2xl font-bold tracking-tight text-slate-900 font-mono">
            {value}
          </span>
          {unit && (
            <span className="text-xs font-semibold text-slate-500 font-sans">
              {unit}
            </span>
          )}
        </div>
      </div>

      <div className="flex items-center justify-between gap-2 mt-3 pt-3 border-t border-slate-100">
        <div className="min-w-0">
          {subtitle && (
            <p className="text-[11px] text-slate-500 truncate" title={typeof subtitle === 'string' ? subtitle : undefined}>
              {subtitle}
            </p>
          )}
          {secondaryValue && (
            <p className="text-[11px] font-medium text-slate-600 truncate">
              {secondaryValue.label}: <span className="font-mono text-slate-800 font-semibold">{secondaryValue.value}</span>
            </p>
          )}
        </div>

        {badge && (
          <Badge variant={badge.variant || 'default'} size="sm" className="shrink-0 font-mono text-[10px]">
            {badge.text}
          </Badge>
        )}
      </div>
    </div>
  );
};
