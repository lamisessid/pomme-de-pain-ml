import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/layout/Layout';
import { AppInfoProvider } from './context/AppInfo';
import { PageLoader } from './components/ui/Loader';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Forecasts = lazy(() => import('./pages/Forecasts'));
const Losses = lazy(() => import('./pages/Losses'));
const Comparison = lazy(() => import('./pages/Comparison'));
const Alerts = lazy(() => import('./pages/Alerts'));
const Products = lazy(() => import('./pages/Products'));
const Assistant = lazy(() => import('./pages/Assistant'));
const About = lazy(() => import('./pages/About'));

function App() {
  return (
    <AppInfoProvider>
      <Router>
        <Layout>
          <Suspense fallback={<PageLoader />}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/forecasts" element={<Forecasts />} />
              <Route path="/losses" element={<Losses />} />
              <Route path="/comparison" element={<Comparison />} />
              <Route path="/alerts" element={<Alerts />} />
              <Route path="/products" element={<Products />} />
              <Route path="/assistant" element={<Assistant />} />
              <Route path="/about" element={<About />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </Suspense>
        </Layout>
      </Router>
    </AppInfoProvider>
  );
}

export default App;
