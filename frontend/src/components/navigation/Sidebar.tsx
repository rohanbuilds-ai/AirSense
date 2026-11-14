import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Database,
  Binary,
  Target,
  Network,
  TrendingDown,
  GraduationCap,
  Activity,
  X,
} from 'lucide-react';

interface NavItem {
  to: string;
  label: string;
  icon: React.ReactNode;
}

const navItems: NavItem[] = [
  { to: '/', label: 'Overview', icon: <LayoutDashboard className="w-4 h-4" /> },
  { to: '/dataset', label: 'Dataset & Matrix', icon: <Database className="w-4 h-4" /> },
  { to: '/linear-algebra', label: 'Linear Algebra Lab', icon: <Binary className="w-4 h-4" /> },
  { to: '/pca-explorer', label: 'PCA Explorer', icon: <Network className="w-4 h-4" /> },
  { to: '/calibration', label: 'Calibration Results', icon: <Target className="w-4 h-4" /> },
  { to: '/drift-analysis', label: 'Drift & Statistics', icon: <TrendingDown className="w-4 h-4" /> },
  { to: '/walkthrough', label: 'Project Walkthrough', icon: <GraduationCap className="w-4 h-4" /> },
];

interface SidebarProps {
  isMobileOpen?: boolean;
  onClose?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ isMobileOpen = false, onClose }) => {
  return (
    <>
      {/* Mobile Backdrop */}
      {isMobileOpen && (
        <div
          role="presentation"
          onClick={onClose}
          className="fixed inset-0 bg-slate-950/70 backdrop-blur-xs z-40 lg:hidden transition-opacity"
        />
      )}

      <aside
        className={`fixed lg:sticky top-0 left-0 z-50 lg:z-30 w-68 bg-[#090D16] border-r border-slate-800/80 flex flex-col justify-between shrink-0 h-screen select-none transition-transform duration-300 ease-out ${
          isMobileOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        {/* Brand & Header */}
        <div>
          <div className="p-5 border-b border-slate-800/80 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-teal-500 to-cyan-600 flex items-center justify-center text-white shadow-md shadow-teal-500/20 ring-1 ring-teal-400/30">
                <Activity className="w-5 h-5" />
              </div>
              <div>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-white text-base tracking-tight font-sans">AirSense</span>
                  <span className="text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded bg-teal-500/10 text-teal-400 border border-teal-500/20">
                    v0.2
                  </span>
                </div>
                <p className="text-[10px] text-slate-400 tracking-tight font-medium">Scientific Analytics Suite</p>
              </div>
            </div>

            {/* Mobile Close Button */}
            {onClose && (
              <button
                type="button"
                onClick={onClose}
                className="lg:hidden p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-slate-800/60"
                aria-label="Close navigation sidebar"
              >
                <X className="w-5 h-5" />
              </button>
            )}
          </div>

          {/* Navigation list */}
          <nav className="p-3 space-y-1 overflow-y-auto max-h-[calc(100vh-190px)]">
            <div className="px-3 pt-3 pb-2 text-[10px] font-bold tracking-wider text-slate-500 uppercase font-mono">
              Scientific Workspaces
            </div>

            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                onClick={onClose}
                className={({ isActive }) =>
                  `group relative flex items-center justify-between px-3 py-2.5 rounded-xl text-xs transition-all duration-150 ${
                    isActive
                      ? 'bg-teal-500/15 text-teal-300 font-semibold shadow-2xs border border-teal-500/20'
                      : 'text-slate-400 hover:text-slate-100 hover:bg-slate-800/50 font-medium'
                  }`
                }
              >
                {({ isActive }) => (
                  <>
                    <div className="flex items-center gap-2.5 min-w-0">
                      <span
                        className={`transition-colors shrink-0 ${
                          isActive ? 'text-teal-400' : 'text-slate-500 group-hover:text-slate-300'
                        }`}
                      >
                        {item.icon}
                      </span>
                      <span className="truncate">{item.label}</span>
                    </div>

                    {/* Active accent pill */}
                    {isActive && (
                      <span className="absolute left-0 top-2 bottom-2 w-0.5 bg-teal-400 rounded-r-full shadow-sm shadow-teal-400" />
                    )}
                  </>
                )}
              </NavLink>
            ))}
          </nav>
        </div>
      </aside>
    </>
  );
};
