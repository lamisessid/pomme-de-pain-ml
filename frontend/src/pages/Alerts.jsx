import React, { useState } from 'react';
import toast from 'react-hot-toast';
import { CheckCircle2, RotateCcw, PackageX, Trash2, CalendarHeart } from 'lucide-react';

import { api } from '../api/client';
import useApi from '../hooks/useApi';
import { useAppInfo } from '../context/AppInfo';
import { Card } from '../components/ui/Card';
import { AsyncContent, PageHeader } from '../components/ui/States';
import SeverityBadge from '../components/ui/SeverityBadge';
import { ALERT_TYPES, fmtDate, fmtEur } from '../utils/format';

const ICONS = { rupture: PackageX, gaspillage: Trash2, evenement: CalendarHeart };
const ICON_COLORS = { rupture: 'text-amber-600 bg-amber-50', gaspillage: 'text-red-600 bg-red-50', evenement: 'text-indigo-600 bg-indigo-50' };

const Alerts = () => {
  const [status, setStatus] = useState('active');
  const [type, setType] = useState('');
  const { refreshAlerts } = useAppInfo();
  const state = useApi(() => api.alerts(status), [status]);
  const stats = useApi(() => api.alertStats(), [state.data]);

  const toggle = async (alert) => {
    try {
      await (alert.resolved ? api.reopenAlert(alert.id) : api.resolveAlert(alert.id));
      toast.success(alert.resolved ? 'Alerte réouverte' : 'Alerte résolue');
      state.reload();
      refreshAlerts();
    } catch (e) {
      toast.error(e.message);
    }
  };

  const tabs = [['active', 'Actives'], ['resolved', 'Résolues'], ['all', 'Toutes']];

  return (
    <>
      <PageHeader title="Alertes" subtitle="Générées à partir des prévisions des 3 prochains jours et du calendrier SQY" />

      {stats.data && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <Card><p className="text-sm text-gray-600">Actives</p><p className="text-2xl font-bold mt-1">{stats.data.active}</p></Card>
          <Card><p className="text-sm text-gray-600">Critiques / élevées</p><p className="text-2xl font-bold mt-1 text-red-600">{(stats.data.by_severity.critical || 0) + (stats.data.by_severity.high || 0)}</p></Card>
          <Card><p className="text-sm text-gray-600">Impact estimé</p><p className="text-2xl font-bold mt-1">{fmtEur(stats.data.estimated_impact)}</p></Card>
          <Card><p className="text-sm text-gray-600">Résolues</p><p className="text-2xl font-bold mt-1 text-green-600">{stats.data.resolved}</p></Card>
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
        <div className="inline-flex rounded-lg border border-gray-300 bg-white p-1">
          {tabs.map(([key, label]) => (
            <button key={key} onClick={() => setStatus(key)}
                    className={`px-3 py-1.5 text-sm rounded-md ${status === key ? 'bg-primary-600 text-white' : 'text-gray-700 hover:bg-gray-50'}`}>
              {label}
            </button>
          ))}
        </div>
        <select value={type} onChange={(e) => setType(e.target.value)} className="border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white">
          <option value="">Tous les types</option>
          {Object.entries(ALERT_TYPES).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
        </select>
      </div>

      <AsyncContent state={state}>
        {(alerts) => {
          const list = type ? alerts.filter((a) => a.type === type) : alerts;
          if (!list.length) return <Card className="text-center text-gray-500">Aucune alerte.</Card>;
          return (
            <div className="space-y-3">
              {list.map((a) => {
                const Icon = ICONS[a.type];
                return (
                  <Card key={a.id} className={a.resolved ? 'opacity-60' : ''}>
                    <div className="flex items-start gap-4">
                      <div className={`p-2 rounded-lg shrink-0 ${ICON_COLORS[a.type]}`}><Icon className="h-5 w-5" /></div>
                      <div className="flex-1 min-w-0">
                        <div className="flex flex-wrap items-center gap-2">
                          <h3 className="font-semibold text-gray-900">{a.title}</h3>
                          <SeverityBadge severity={a.severity} />
                          <span className="text-xs text-gray-500">{ALERT_TYPES[a.type]} · {fmtDate(a.date)}</span>
                        </div>
                        <p className="text-sm text-gray-600 mt-1">{a.description}</p>
                        <p className="text-sm text-gray-900 mt-2"><span className="font-medium">Action :</span> {a.suggested_action}</p>
                      </div>
                      <div className="text-right shrink-0">
                        {a.estimated_impact != null && <p className="text-sm font-semibold tabular-nums">{fmtEur(a.estimated_impact, true)}</p>}
                        <button onClick={() => toggle(a)} className="btn-secondary mt-2 inline-flex items-center text-sm py-1.5">
                          {a.resolved ? <><RotateCcw className="h-4 w-4 mr-1" /> Rouvrir</> : <><CheckCircle2 className="h-4 w-4 mr-1" /> Résoudre</>}
                        </button>
                      </div>
                    </div>
                  </Card>
                );
              })}
            </div>
          );
        }}
      </AsyncContent>
    </>
  );
};

export default Alerts;
