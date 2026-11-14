import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { HelpCircle, Database, AlertCircle, Menu } from 'lucide-react';
import { Badge } from '../ui/Badge';
import { ContextHelpModal } from './ContextHelpModal';
import { apiClient } from '../../services/apiClient';

const routeTitles: Record<string, string> = {
  '/': 'Overview',
  '/dataset': 'Dataset & Matrix',
  '/data-quality': 'Dataset & Matrix',
  '/linear-algebra': 'Linear Algebra Lab',
  '/calibration': 'Calibration Results',
  '/pca-explorer': 'PCA Explorer',
  '/drift-analysis': 'Drift & Statistics',
  '/statistical-inference': 'Statistical Inference',
  '/walkthrough': 'Project Walkthrough',
  '/methodology-viva': 'Project Walkthrough',
};

interface TopBarProps {
  onToggleMobileMenu?: () => void;
}

export const TopBar: React.FC<TopBarProps> = ({ onToggleMobileMenu }) => {
  const location = useLocation();
  const [isHelpOpen, setIsHelpOpen] = useState(false);
  const [isApiOnline, setIsApiOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let mounted = true;
    const checkStatus = async () => {
      try {
        const res = await apiClient.checkHealth();
        if (mounted) {
          setIsApiOnline(res.status === 'healthy');
        }
      } catch {
        if (mounted) {
          setIsApiOnline(false);
        }
      }
    };

    checkStatus();
    const interval = setInterval(checkStatus, 20000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const pageTitle = routeTitles[location.pathname] || 'Workspace';

  return (
    <>
      <header className="h-16 bg-white/90 backdrop-blur-md border-b border-slate-200/80 px-4 sm:px-6 lg:px-8 flex items-center justify-between sticky top-0 z-30 transition-all">
        {/* Left: Mobile Toggle & Breadcrumbs */}
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onToggleMobileMenu}
            className="lg:hidden p-2 text-slate-500 hover:text-slate-800 hover:bg-slate-100 rounded-xl transition-colors cursor-pointer"
            aria-label="Toggle navigation menu"
          >
            <Menu className="w-5 h-5" />
          </button>

          <div className="flex items-center gap-2 text-xs">
            <span className="font-medium text-slate-400 hidden sm:inline">AirSense</span>
            <span className="text-slate-300 hidden sm:inline">/</span>
            <h1 className="text-sm font-bold text-slate-900 tracking-tight">{pageTitle}</h1>
          </div>
        </div>

        {/* Right: Dataset Context & Status Badges */}
        <div className="flex items-center gap-3">
          {/* Dataset Context Pill */}
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-slate-50 border border-slate-200/90 rounded-xl text-xs text-slate-600">
            <Database className="w-3.5 h-3.5 text-teal-600" />
            <span className="font-semibold text-slate-700">UCI Air Quality</span>
            <span className="text-slate-300">·</span>
            <span className="text-slate-500 font-mono text-[11px]">8,991 Supervised Hours</span>
          </div>

          {/* Backend Connection Indicator with pulsing dot */}
          {isApiOnline === null ? (
            <Badge variant="outline" size="sm" className="font-mono text-[11px]">
              <span className="w-1.5 h-1.5 rounded-full bg-slate-400 animate-pulse mr-1" />
              Connecting...
            </Badge>
          ) : isApiOnline ? (
            <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono font-medium bg-emerald-50 text-emerald-800 border border-emerald-200/80">
              <span className="relative flex h-2 w-2">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
              </span>
              <span className="hidden sm:inline">Gateway</span> Online
            </div>
          ) : (
            <Badge variant="danger" size="sm" className="gap-1.5 font-mono text-[11px]">
              <AlertCircle className="w-3 h-3 text-rose-600" />
              Gateway Offline
            </Badge>
          )}

          {/* Context Help Button */}
          <button
            type="button"
            onClick={() => setIsHelpOpen(true)}
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-slate-50 hover:bg-slate-100 hover:border-slate-300 border border-slate-200/90 rounded-xl transition-all cursor-pointer shadow-2xs"
            title="View mathematical formulas and scientific notes for this workspace"
          >
            <HelpCircle className="w-3.5 h-3.5 text-teal-600" />
            <span className="hidden sm:inline">Scientific Notes</span>
          </button>
        </div>
      </header>

      <ContextHelpModal
        isOpen={isHelpOpen}
        onClose={() => setIsHelpOpen(false)}
        currentPath={location.pathname}
      />
    </>
  );
};
