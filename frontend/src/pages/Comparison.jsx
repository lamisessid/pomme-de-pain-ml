import React, { useState } from 'react';
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { PiggyBank } from 'lucide-react';

import { api } from '../api/client';
import useApi from '../hooks/useApi';
import { Card, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { AsyncContent, PageHeader, PeriodSelect } from '../components/ui/States';
import Badge from '../components/ui/Badge';
import { COLORS, categoryLabel, fmtDate, fmtEur, fmtInt } from '../utils/format';

/** Ligne du bilan ; la valeur la plus basse est mise en vert sauf si `neutral`. */
const Row = ({ label, ai, seat, fmt = fmtInt, neutral = false }) => {
  const win = (mine, other) => (!neutral && mine < other ? 'text-green-700' : '');
  return (
    <tr className="border-t">
      <td className="py-2 text-gray-600">{label}</td>
      <td className={`py-2 text-right tabular-nums font-medium ${win(ai, seat)}`}>{fmt(ai)}</td>
      <td className={`py-2 text-right tabular-nums font-medium ${win(seat, ai)}`}>{fmt(seat)}</td>
    </tr>
  );
};

const Comparison = () => {
  const [days, setDays] = useState(30);
  const state = useApi(() => api.comparison(days), [days]);

  return (
    <>
      <PageHeader
        title="Siège vs IA"
        subtitle="Simulation : si l'on avait commandé selon chaque règle sur la période, quels invendus et quelles ventes manquées ?"
        actions={<PeriodSelect value={days} onChange={setDays} options={[14, 30, 60, 90]} />}
      />
      <AsyncContent state={state}>
        {(c) => {
          const { ai, seat } = c.totals;
          const chart = [
            { name: 'Invendus', IA: ai.unsold_cost, Siège: seat.unsold_cost },
            { name: 'Ruptures', IA: ai.missed_margin, Siège: seat.missed_margin },
            { name: 'Total', IA: ai.total_cost, Siège: seat.total_cost },
          ];
          return (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
                <Card className="bg-gradient-to-br from-green-50 to-white">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm text-gray-600">Économie simulée sur {c.period.days} jours</p>
                      <p className={`text-3xl font-bold mt-2 tabular-nums ${c.totals.savings >= 0 ? 'text-green-700' : 'text-red-700'}`}>
                        {fmtEur(c.totals.savings)}
                      </p>
                      <p className="text-xs text-gray-500 mt-2">Invendus (au coût) + marge perdue sur ruptures</p>
                    </div>
                    <PiggyBank className="h-10 w-10 text-green-600" />
                  </div>
                  <div className="mt-4 pt-4 border-t border-green-100 text-xs text-gray-600 space-y-1">
                    <p><span className="font-medium" style={{ color: COLORS.ai }}>IA</span> : {c.rules.ai}</p>
                    <p><span className="font-medium" style={{ color: COLORS.seat }}>Siège</span> : {c.rules.seat}</p>
                  </div>
                </Card>

                <Card>
                  <CardTitle>Bilan</CardTitle>
                  <table className="w-full text-sm mt-3">
                    <thead><tr className="text-gray-500"><th className="text-left pb-1" /><th className="text-right pb-1">IA</th><th className="text-right pb-1">Siège</th></tr></thead>
                    <tbody>
                      <Row label="Unités commandées" ai={ai.units_ordered} seat={seat.units_ordered} neutral />
                      <Row label="Invendus" ai={ai.unsold_units} seat={seat.unsold_units} />
                      <Row label="Ventes manquées" ai={ai.missed_units} seat={seat.missed_units} />
                      <Row label="Coût total" ai={ai.total_cost} seat={seat.total_cost} fmt={fmtEur} />
                    </tbody>
                  </table>
                </Card>

                <Card>
                  <CardTitle>Décomposition des coûts</CardTitle>
                  <div className="h-48 mt-3">
                    <ResponsiveContainer>
                      <BarChart data={chart}>
                        <XAxis dataKey="name" fontSize={12} interval={0} />
                        <YAxis fontSize={11} tickFormatter={(v) => `${(v / 1000).toFixed(0)}k`} />
                        <Tooltip formatter={(v, n) => [fmtEur(v), n]} />
                        <Bar dataKey="IA" fill={COLORS.ai} radius={[3, 3, 0, 0]} />
                        <Bar dataKey="Siège" fill={COLORS.seat} radius={[3, 3, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </Card>
              </div>

              <Card className="mb-6">
                <CardHeader><CardTitle>Coût quotidien</CardTitle><CardDescription>Invendus + marge manquée, tous produits</CardDescription></CardHeader>
                <div className="h-64">
                  <ResponsiveContainer>
                    <LineChart data={c.daily}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="date" tickFormatter={(v) => fmtDate(v, { day: '2-digit', month: '2-digit' })} fontSize={12} />
                      <YAxis fontSize={12} tickFormatter={(v) => `${v} €`} />
                      <Tooltip labelFormatter={(v) => fmtDate(v)} formatter={(v, n) => [fmtEur(v), n]} />
                      <Legend />
                      <Line dataKey="ai_cost" name="IA" stroke={COLORS.ai} strokeWidth={2} dot={false} />
                      <Line dataKey="seat_cost" name="Siège" stroke={COLORS.seat} strokeWidth={2} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Card>

              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <Card padding={false}>
                  <div className="p-6 pb-0">
                    <CardTitle>Commandes recommandées pour le {fmtDate(c.tomorrow.date)}</CardTitle>
                    <CardDescription>Produits où l'écart IA / siège est le plus grand</CardDescription>
                  </div>
                  <table className="w-full text-sm mt-4">
                    <thead className="bg-gray-50 text-gray-500 text-left">
                      <tr><th className="px-6 py-3">Produit</th><th className="px-3 py-3 text-right">Prévision</th><th className="px-3 py-3 text-right">IA</th><th className="px-3 py-3 text-right">Siège</th><th className="px-6 py-3">Action</th></tr>
                    </thead>
                    <tbody>
                      {c.tomorrow.recommendations.slice(0, 12).map((r) => (
                        <tr key={r.product_id} className="border-t">
                          <td className="px-6 py-2 font-medium">{r.name}</td>
                          <td className="px-3 py-2 text-right tabular-nums">{fmtInt(r.predicted_sales)}</td>
                          <td className="px-3 py-2 text-right tabular-nums">{r.ai_order}</td>
                          <td className="px-3 py-2 text-right tabular-nums">{r.seat_order}</td>
                          <td className="px-6 py-2">
                            <Badge variant={r.action === 'réduire' ? 'success' : r.action === 'augmenter' ? 'warning' : 'default'}>
                              {r.action} {r.difference !== 0 && `(${r.difference > 0 ? '+' : ''}${r.difference})`}
                            </Badge>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Card>

                <Card padding={false}>
                  <div className="p-6 pb-0">
                    <CardTitle>Par produit</CardTitle>
                    <CardDescription>Économie simulée (coût siège − coût IA)</CardDescription>
                  </div>
                  <div className="max-h-[480px] overflow-y-auto mt-4">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-50 text-gray-500 text-left sticky top-0">
                        <tr><th className="px-6 py-3">Produit</th><th className="px-3 py-3">Catégorie</th><th className="px-3 py-3 text-right">Coût IA</th><th className="px-3 py-3 text-right">Coût siège</th><th className="px-6 py-3 text-right">Économie</th></tr>
                      </thead>
                      <tbody>
                        {c.per_product.map((p) => (
                          <tr key={p.product_id} className="border-t">
                            <td className="px-6 py-2 font-medium">{p.name}</td>
                            <td className="px-3 py-2 text-gray-600">{categoryLabel(p.category)}</td>
                            <td className="px-3 py-2 text-right tabular-nums">{fmtEur(p.ai.total_cost)}</td>
                            <td className="px-3 py-2 text-right tabular-nums">{fmtEur(p.seat.total_cost)}</td>
                            <td className={`px-6 py-2 text-right tabular-nums font-medium ${p.savings >= 0 ? 'text-green-700' : 'text-red-700'}`}>{fmtEur(p.savings)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            </>
          );
        }}
      </AsyncContent>
    </>
  );
};

export default Comparison;
