# AirSense: Scientific UI/UX Design System Specification

**Product Name**: AirSense  
**Descriptor**: *Mathematical Sensor Calibration & Drift Analysis*  
**Platform**: Responsive Single Page Web Application (React 18 + TypeScript + Vite + Tailwind CSS + Motion for React)  
**Backend Gateway**: Python FastAPI REST Adapter (`/api/v1/`)  
**Design Philosophy**: A restrained, high-density scientific computing interface prioritizing mathematical clarity, data integrity, transparent physical interpretation, and responsive motion feedback.

---

## 1. Visual Identity & Design Principles

### Core Principles
1. **Light-First Analytical Surfaces with Dark Accent Shell**:
   - Clean, warm-neutral / cool-white canvas (`#F8FAFC`) with crisp elevated card containers (`#FFFFFF`).
   - Premium dark midnight navigation sidebar (`#090D16`, `border-r border-slate-800/80`) providing strong structural anchoring without competing with white analytical surfaces.
   - Visually distinctive dark-navy hero surface on the Overview page (`bg-gradient-to-br from-[#090D16] via-[#0F172A] to-[#111C33]` with subtle technical line grid).
2. **Typography with Mathematical Distinction**:
   - **Primary Sans**: `Inter` (weights: 400, 500, 600, 700) for UI controls, navigation labels, and scientific prose.
   - **Monospace Telemetry**: `JetBrains Mono` / `Fira Code` for numerical values, matrix notation, formulas, timestamps, and error tolerances.
3. **Restrained, Meaningful Color Semantics**:
   - **Deep Slate / Navy**: Primary text (`#0F172A`, `#1E293B`, `#334155`) ensuring high contrast (WCAG AAA).
   - **Restrained Teal / Cyan**: Selected models, active workspaces, and key scientific signals (`#0D9488`, `#0F766E`, `#14B8A6`).
   - **Subtle Slate Borders**: Soft structural boundaries (`#E2E8F0`, `#F1F5F9`) avoiding heavy visual framing.
   - **Amber Only for Genuine Cautions**: Environmental confounding, Durbin-Watson serial autocorrelation, and missing sentinels (`#D97706`).
   - **Rose for Verified Errors & Blackouts**: Severe collinearity, power outages, and failed requests (`#E11D48`).
4. **Anti-Patterns Strictly Avoided**:
   - No generic AI landing-page gradients or glowing nebulae everywhere.
   - No emoji-based navigation icons.
   - No oversized decorative headings that waste analytical screen real estate.
   - No misleading real-time telemetry tickers (the UCI dataset is a historical 13-month field research record).

---

## 2. Color Palette & Design Tokens

| Token Name | Hex Code | Tailwind Utility | Semantic Purpose |
|---|---|---|---|
| `surface-ground` | `#F8FAFC` | `bg-slate-50` / `bg-[#F8FAFC]` | Application canvas |
| `surface-card` | `#FFFFFF` | `bg-white` | Analytical cards, tables, modal containers |
| `surface-subtle` | `#F1F5F9` | `bg-slate-100` | Table headers, secondary stat strips |
| `surface-sidebar` | `#090D16` | `bg-[#090D16]` | Persistent dark navigation sidebar |
| `surface-hero` | `#0F172A` | `bg-gradient-to-br` | Overview centerpiece hero surface |
| `border-subtle` | `#E2E8F0` | `border-slate-200` | Standard card and table borders |
| `text-primary` | `#0F172A` | `text-slate-900` | Section headings, primary metrics |
| `text-secondary` | `#475569` | `text-slate-600` | Descriptions, captions, table headers |
| `text-muted` | `#94A3B8` | `text-slate-400` | Axis tick labels, metadata, units |
| `brand-teal` | `#0D9488` | `text-teal-700`, `bg-teal-50` | Active navigation, selected model, 1:1 line |
| `status-success` | `#059669` | `text-emerald-700`, `bg-emerald-50` | Valid observations, passing invariant tests |
| `status-warning` | `#D97706` | `text-amber-700`, `bg-amber-50` | Confounding warnings, high missingness |
| `status-danger` | `#E11D48` | `text-rose-700`, `bg-rose-50` | Station blackouts, collinear traps |

---

## 3. Motion Design System (`motion/react`)

AirSense implements a restrained, high-performance motion system defined in `frontend/src/lib/motion.ts`:

### 3.1 Core Animation Variants
- **`pageVariants`**: Gentle fade-and-rise entrance on route navigation (`opacity: 0 -> 1`, `y: 8 -> 0`, duration: 280ms, easing: `easeOutCubic`).
- **`staggerContainer` & `staggerItem`**: Staggered appearance of metric card groups (stagger: 40ms, delay: 20ms).
- **`cardHoverVariants`**: Subtle 2px elevation and soft shadow on hover (`transition: 180ms ease-out`).
- **`tabContentVariants`**: Smooth transitions between visualization tabs (`opacity: 0 -> 1`, `y: 6 -> 0`, duration: 200ms).
- **`accordionVariants`**: Smooth height expansion and collapse for mathematical derivations.
- **`modalBackdropVariants` & `modalContentVariants`**: Scale and fade entrance for the Scientific Notes modal dialog.

### 3.2 Motion Guidelines & Reduced Motion
- **Scientific Stability**: Time-series charts and numerical outputs remain completely static once rendered to ensure data readability.
- **Accessibility (`prefers-reduced-motion`)**: When reduced motion is requested, all CSS and motion animations collapse to `0.01ms` duration via media query in `index.css`.
- **Skeleton Shimmer**: Reusable `Skeleton` component displays CSS-based shimmer without triggering continuous layout reflows.

---

## 4. Reusable UI Components

### 4.1 `MetricCard`
- **Purpose**: Displays key performance indicators (KPIs) in compact, high-density grids.
- **Features**: Label, monospace value, physical unit ($\mu g/m^3$, hours), status badge, secondary context (e.g. $R^2$), and subtle hover elevation.

### 4.2 `Card`
- **Purpose**: Standard container for charts, tables, and scientific explanations.
- **Features**: Left-aligned bold title, subtle sub-caption, optional action toolbar (pagination, filters), and optional hover elevation.

### 4.3 `Badge`
- **Variants**: `teal`, `default`, `info`, `success`, `warning`, `danger`, `outline`, `navy`.
- **Features**: Dot status indicator, monospace formatting, and rounded-full pill design.

### 4.4 `Alert`
- **Variants**: `info`, `warning`, `danger`, `success`.
- **Purpose**: Essential scientific caveats, autocorrelation diagnostics, and regulatory data-quality rules.

### 4.5 `Skeleton`
- **Variants**: `rectangular`, `text`, `circular`.
- **Purpose**: Dimensionally stable loading states for tables and KPI cards.

---

## 5. Navigation & Layout Architecture

### Persistent Dark Sidebar (`Sidebar.tsx`)
- **Location**: Left screen edge, fixed sticky position, 272px (`w-68`).
- **Surface**: `#090D16` with `#1E293B` borders.
- **Active Indicator**: Cyan glowing pill and `bg-teal-500/15` container.
- **Mobile Drawer**: Slide-in drawer with backdrop overlay for screens $< 1024$px.

### Top Navigation Bar (`TopBar.tsx`)
- **Breadcrumbs**: `AirSense / {PageTitle}`.
- **Dataset Context**: `UCI Air Quality · 8,991 Supervised Hours`.
- **Live Health Status**: Pulsing green dot indicator reflecting real-time `/api/v1/health` connectivity.
- **Scientific Notes**: Direct modal access to syllabus concepts, mathematical purposes, and viva defense caveats.

---

## 6. Responsive Breakpoint Matrix

| Viewport | Target Device | Navigation Mode | Main Content Width |
| :--- | :--- | :--- | :--- |
| **$\ge 1440$px** | Large Desktop | Persistent Sidebar (`w-68`) | `max-w-7xl`, 2-column analytical grids |
| **$1280$px** | Standard Laptop | Persistent Sidebar (`w-68`) | `max-w-7xl`, 2-column analytical grids |
| **$768$px** | Tablet | Persistent Sidebar / Overlay | Single-column stack, horizontal table scroll |
| **$390$px** | Mobile Smartphone | Collapsible Hamburger Drawer | Single-column stack, responsive KPI cards |
