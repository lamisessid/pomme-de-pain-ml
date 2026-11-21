import React, { useState } from 'react';
import {
  Bar, BarChart, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from 'recharts';
import { Trash2, Euro, Percent, CalendarDays } from 'lucide-react';

import { api } from '../api/client';
import useApi from '../hooks/useApi';
import KpiCard from '../components/dashboard/KpiCard';
import { Card, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { AsyncContent, PageHeader, PeriodSelect } from '../components/ui/States';
import { COLORS, categoryLabel, fmtDate, fmtEur, fmtInt, fmtPct } from '../utils/format';

const Losses = () => {
  const [days, setDays] = useState(30);
  const state = useApi(() => api.losses(days), [days]);

  return (
    <>
      <PageHeader
        title="Analyse des pertes"
        subtitle="Invendus jetés, valorisés au coût de revient"
        actions={<PeriodSelect value={days} onChange={setDays} options={[7, 30, 90, 180]} />}
      />
      <AsyncContent state={state}>
        {(l) => (
          <>
            <p className="text-sm text-gray-500 -mt-4 mb-6">Du {fmtDate(l.period.start)} au {fmtDate(l.period.end)}</p>
            <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4 mb-6">
              <KpiCard title="Unités perdues" value={fmtInt(l.totals.waste)} icon={Trash2} iconColor="text-red-600" bgColor="bg-red-50" />
              <KpiCard title="Coût des pertes" value={fmtEur(l.totals.waste_cost)} change={l.totals.vs_previous_period_pct}
                       changeLabel="vs période préc." higherIsBetter={false} icon={Euro} iconColor="text-amber-600" bgColor="bg-amber-50" />
              <KpiCard title="Taux de perte" value={fmtPct(l.totals.waste_rate)} hint="pertes / ventes" icon={Percent} />
              <KpiCard title="Coût moyen par jour" value={fmtEur(l.totals.avg_daily_cost)} icon={CalendarDays}
                       iconColor="text-gray-600" bgColor="bg-gray-100" />
            </div>

            <Card className="mb-6">
              <CardHeader><CardTitle>Évolution quotidienne</CardTitle></CardHeader>
              <div className="h-64">
                <ResponsiveContainer>
                  <ComposedChart data={l.daily.map((d) => ({ ...d, rate: d.waste_rate * 100 }))}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="date" tickFormatter={(v) => fmtDate(v, { day: '2-digit', month: '2-digit' })} fontSize={12} />
                    <YAxis yAxisId="c" fontSize={12} tickFormatter={(v) => `${v} €`} />
                    <YAxis yAxisId="r" orientation="right" fontSize={12} tickFormatter={(v) => `${v.toFixed(0)} %`} />
                    <Tooltip labelFormatter={(v) => fmtDate(v)}
                             formatter={(v, n) => [n === 'Taux de perte' ? fmtPct(v) : fmtEur(v, true), n]} />
                    <Legend />
                    <Bar yAxisId="c" dataKey="waste_cost" name="Coût des pertes" fill={COLORS.waste} fillOpacity={0.8} radius={[3, 3, 0, 0]} />
                    <Line yAxisId="r" dataKey="rate" name="Taux de perte" stroke={COLORS.actual} dot={false} strokeWidth={2} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </Card>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <Card>
                <CardHeader><CardTitle>Par catégorie</CardTitle><CardDescription>Coût des pertes</CardDescription></CardHeader>
                <div className="h-72">
                  <ResponsiveContainer>
                    <BarChart data={l.by_category.map((c) => ({ ...c, label: categoryLabel(c.category) }))} layout="vertical" margin={{ left: 30 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis type="number" fontSize={12} tickFormatter={(v) => `${v} €`} />
                      <YAxis type="category" dataKey="label" fontSize={12} width={90} />
                      <Tooltip formatter={(v, n, p) => [`${fmtEur(v)} (taux ${fmtPct(p.payload.waste_rate)})`, 'Coût']} />
                      <Bar dataKey="waste_cost" fill={COLORS.seat} radius={[0, 3, 3, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              <Card>
                <CardHeader><CardTitle>Par jour de la semaine</CardTitle><CardDescription>Taux de perte moyen</CardDescription></CardHeader>
                <div className="h-72">
                  <ResponsiveContainer>
                    <BarChart data={l.by_weekday}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="weekday" fontSize={12} tickFormatter={(v) => v.slice(0, 3)} />
                      <YAxis fontSize={12} tickFormatter={(v) => `${v} %`} />
                      <Tooltip formatter={(v) => [fmtPct(v), 'Taux de perte']} />
                      <Bar dataKey="waste_rate" fill={COLORS.ai} radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <Card className="lg:col-span-2" padding={false}>
                <div className="p-6 pb-0"><CardTitle>Produits les plus coûteux</CardTitle></div>
                <table className="w-full text-sm mt-4">
                  <thead className="bg-gray-50 text-gray-500 text-left">
                    <tr><th className="px-6 py-3">Produit</th><th className="px-3 py-3">Catégorie</th><th className="px-3 py-3 text-right">Pertes</th><th className="px-3 py-3 text-right">Taux</th><th className="px-6 py-3 text-right">Coût</th></tr>
                  </thead>
                  <tbody>
                    {l.top_products.map((p) => (
                      <tr key={p.product_id} className="border-t">
                        <td className="px-6 py-2 font-medium">{p.name}</td>
                        <td className="px-3 py-2 text-gray-600">{categoryLabel(p.category)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{fmtInt(p.waste)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{fmtPct(p.waste_rate)}</td>
                        <td className="px-6 py-2 text-right tabular-nums font-medium">{fmtEur(p.waste_cost)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>

              <Card>
                <CardHeader><CardTitle>Facteurs de contexte</CardTitle><CardDescription>Taux de perte selon le contexte du jour</CardDescription></CardHeader>
                {Object.entries({ rainy: 'Météo', event: 'Événements SQY' }).map(([key, title]) => (
                  <div key={key} className="mb-5">
                    <p className="text-sm font-medium text-gray-700 mb-2">{title}</p>
                    {l.effects[key].map((e) => (
                      <div key={e.label} className="flex justify-between text-sm py-1">
                        <span className="text-gray-600">{e.label} <span className="text-gray-400">({e.days} j)</span></span>
                        <span className="font-semibold tabular-nums">{fmtPct(e.waste_rate)}</span>
                      </div>
                    ))}
                  </div>
                ))}
              </Card>
            </div>
          </>
        )}
      </AsyncContent>
    </>
  );
};

export default Losses;
