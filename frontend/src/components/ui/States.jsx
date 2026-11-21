import React from 'react';
import { AlertCircle, RefreshCw } from 'lucide-react';
import { PageLoader } from './Loader';

export const ErrorState = ({ error, onRetry }) => (
  <div className="min-h-[300px] flex items-center justify-center">
    <div className="text-center max-w-md">
      <AlertCircle className="h-10 w-10 text-red-500 mx-auto" />
      <p className="mt-3 font-medium text-gray-900">Impossible de charger les données</p>
      <p className="mt-1 text-sm text-gray-600">{error?.message}</p>
      <p className="mt-1 text-xs text-gray-500">Vérifiez que l'API tourne (voir README).</p>
      {onRetry && (
        <button onClick={onRetry} className="btn-secondary mt-4 inline-flex items-center">
          <RefreshCw className="h-4 w-4 mr-2" /> Réessayer
        </button>
      )}
    </div>
  </div>
);

/** Affiche le loader, l'erreur ou le contenu selon l'état d'un useApi. */
export const AsyncContent = ({ state, children }) => {
  if (state.loading && !state.data) return <PageLoader />;
  if (state.error) return <ErrorState error={state.error} onRetry={state.reload} />;
  return children(state.data);
};

export const PageHeader = ({ title, subtitle, actions }) => (
  <div className="flex flex-col md:flex-row md:items-end md:justify-between gap-4 mb-6">
    <div>
      <h1 className="text-2xl font-bold text-gray-900">{title}</h1>
      {subtitle && <p className="text-gray-600 mt-1">{subtitle}</p>}
    </div>
    {actions && <div className="flex items-center gap-2">{actions}</div>}
  </div>
);

export const PeriodSelect = ({ value, onChange, options }) => (
  <select
    value={value}
    onChange={(e) => onChange(Number(e.target.value))}
    className="border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white"
  >
    {options.map((o) => (
      <option key={o} value={o}>{o} jours</option>
    ))}
  </select>
);
