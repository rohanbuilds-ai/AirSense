import React from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { MainLayout } from './layouts/MainLayout';
import { OverviewPage } from './pages/OverviewPage';
import { DatasetMatrixPage } from './pages/DatasetMatrixPage';
import { LinearAlgebraPage } from './pages/LinearAlgebraPage';
import { CalibrationPage } from './pages/CalibrationPage';
import { PcaExplorerPage } from './pages/PcaExplorerPage';
import { DriftAnalysisPage } from './pages/DriftAnalysisPage';
import { StatisticalInferencePage } from './pages/StatisticalInferencePage';
import { ProjectWalkthroughPage } from './pages/ProjectWalkthroughPage';

export const App: React.FC = () => {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<MainLayout />}>
          <Route path="/" element={<OverviewPage />} />
          <Route path="/dataset" element={<DatasetMatrixPage />} />
          <Route path="/data-quality" element={<DatasetMatrixPage />} />
          <Route path="/linear-algebra" element={<LinearAlgebraPage />} />
          <Route path="/calibration" element={<CalibrationPage />} />
          <Route path="/pca-explorer" element={<PcaExplorerPage />} />
          <Route path="/drift-analysis" element={<DriftAnalysisPage />} />
          <Route path="/statistical-inference" element={<StatisticalInferencePage />} />
          <Route path="/walkthrough" element={<ProjectWalkthroughPage />} />
          <Route path="/methodology-viva" element={<ProjectWalkthroughPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
};
