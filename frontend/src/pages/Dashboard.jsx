import React from 'react';
import { Link } from 'react-router-dom';
import {
  Bar, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { Euro, ShoppingBag, Trash2, Percent, CloudRain, Thermometer, CalendarClock, ArrowRight } from 'lucide-react';

import { api } from '../api/client';
import useApi from '../hooks/useApi';
import KpiCard from '../components/dashboard/KpiCard';
import { Card, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { AsyncContent, PageHeader } from '../components/ui/States';
import SeverityBadge from '../components/ui/SeverityBadge';
import Badge from '../components/ui/Badge';
import { COLORS, categoryLabel, fmtDate, fmtDateLong, fmtEur, fmtInt, fmtPct } from '../utils/format';

const Context = ({ ctx }) => (
  <div className="flex flex-wrap gap-3 text-sm text-gray-700">
    <span className="inline-flex items-center"><Thermometer className="h-4 w-4 mr-1 text-orange-500" />{ctx.temperature.toFixed(1)} °C</span>
    <span className="inline-flex items-center"><CloudRain className="h-4 w-4 mr-1 text-blue-500" />{ctx.rain.toFixed(1)} mm</span>
    {ctx.is_holiday === 1 && <Badge variant="warning">Jour férié</Badge>}
    {ctx.events.map((e) => <Badge key={e.name} variant="primary">{e.name} · x{e.impact.toFixed(1)}</Badge>)}
  </div>
);

const Dashboard = () => {
  const state = useApi(() => api.dashboard(), []);

  return (
    <AsyncContent state={state}>
      {(d) => (
        <>
          <PageHeader title="Tableau de bord" subtitle={`Activité du ${fmtDateLong(d.date)}`} />

          <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 mb-6">
            <KpiCard title="Chiffre d'affaires" value={fmtEur(d.kpis.revenue)} change={d.kpis.vs_last_week.revenue}
                     icon={Euro} iconColor="text-green-600" bgColor="bg-green-50" />
            <KpiCard title="Ventes" value={`${fmtInt(d.kpis.sales)} unités`} change={d.kpis.vs_last_week.sales}
                     icon={ShoppingBag} />
            <KpiCard title="Pertes" value={`${fmtInt(d.kpis.waste)} unités`} change={d.kpis.vs_last_week.waste}
                     higherIsBetter={false} icon={Trash2} iconColor="text-red-600" bgColor="bg-red-50" />
            <KpiCard title="Coût des pertes" value={fmtEur(d.kpis.waste_cost)} change={d.kpis.vs_last_week.waste_cost}
                     higherIsBetter={false} hint={`Taux de perte ${fmtPct(d.kpis.waste_rate)}`}
                     icon={Percent} iconColor="text-amber-600" bgColor="bg-amber-50" />
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Ventes et pertes — 30 derniers jours</CardTitle>
              </CardHeader>
              <div className="h-72">
                <ResponsiveContainer>
                  <ComposedChart data={d.trend}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="date" tickFormatter={(v) => fmtDate(v, { day: '2-digit', month: '2-digit' })} fontSize={12} />
                    <YAxis yAxisId="s" fontSize={12} />
                    <YAxis yAxisId="w" orientation="right" fontSize={12} />
                    <Tooltip labelFormatter={(v) => fmtDate(v)} formatter={(v, n) => [fmtInt(v), n]} />
                    <Legend />
                    <Bar yAxisId="s" dataKey="sales" name="Ventes" fill={COLORS.ai} radius={[3, 3, 0, 0]} />
                    <Line yAxisId="w" dataKey="waste" name="Pertes" stroke={COLORS.waste} strokeWidth={2} dot={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Prévision pour demain</CardTitle>
                <CardDescription>{fmtDateLong(d.tomorrow.date)}</CardDescription>
              </CardHeader>
              <dl className="space-y-3">
                <div className="flex justify-between"><dt className="text-gray-600">Ventes prévues</dt><dd className="font-semibold tabular-nums">{fmtInt(d.tomorrow.predicted_sales)}</dd></div>
                <div className="flex justify-between"><dt className="text-gray-600">CA prévu</dt><dd className="font-semibold tabular-nums">{fmtEur(d.tomorrow.predicted_revenue)}</dd></div>
                <div className="flex justify-between"><dt className="text-gray-600">Pertes estimées</dt><dd className="font-semibold tabular-nums text-red-600">{fmtInt(d.tomorrow.predicted_waste)}</dd></div>
              </dl>
              <div className="mt-4 pt-4 border-t border-gray-200">
                <Context ctx={d.tomorrow.context} />
              </div>
              <Link to="/forecasts" className="mt-4 inline-flex items-center text-sm text-primary-600 hover:text-primary-800">
                Détail des prévisions <ArrowRight className="h-4 w-4 ml-1" />
              </Link>
            </Card>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Meilleurs produits du jour</CardTitle>
                <div className="mt-2"><Context ctx={d.context} /></div>
              </CardHeader>
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-left text-gray-500 border-b"><th className="pb-2">Produit</th><th className="pb-2">Catégorie</th><th className="pb-2 text-right">Ventes</th><th className="pb-2 text-right">CA</th></tr>
                </thead>
                <tbody>
                  {d.top_products.map((p) => (
                    <tr key={p.id} className="border-b last:border-0">
                      <td className="py-2 font-medium">{p.name}</td>
                      <td className="py-2 text-gray-600">{categoryLabel(p.category)}</td>
                      <td className="py-2 text-right tabular-nums">{fmtInt(p.sales)}</td>
                      <td className="py-2 text-right tabular-nums">{fmtEur(p.revenue)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>

            <Card>
              <CardHeader className="flex items-start justify-between">
                <div>
                  <CardTitle>Alertes prioritaires</CardTitle>
                  <CardDescription>{d.alerts.active} alertes actives sur les 3 prochains jours</CardDescription>
                </div>
                <CalendarClock className="h-5 w-5 text-gray-400" />
              </CardHeader>
              <ul className="space-y-3">
                {d.alerts.top.map((a) => (
                  <li key={a.id} className="border border-gray-200 rounded-lg p-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className="font-medium text-gray-900">{a.title}</span>
                      <SeverityBadge severity={a.severity} />
                    </div>
                    <p className="text-sm text-gray-600 mt-1">{a.suggested_action}</p>
                    <p className="text-xs text-gray-500 mt-1">
                      {fmtDate(a.date)}{a.estimated_impact != null && ` · impact estimé ${fmtEur(a.estimated_impact)}`}
                    </p>
                  </li>
                ))}
              </ul>
              <Link to="/alerts" className="mt-4 inline-flex items-center text-sm text-primary-600 hover:text-primary-800">
                Toutes les alertes <ArrowRight className="h-4 w-4 ml-1" />
              </Link>
            </Card>
          </div>
        </>
      )}
    </AsyncContent>
  );
};

export default Dashboard;
