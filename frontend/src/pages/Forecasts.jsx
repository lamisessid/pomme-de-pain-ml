import React, { useState } from 'react';
import {
  Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';

import { api } from '../api/client';
import useApi from '../hooks/useApi';
import { Card, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { AsyncContent, PageHeader, PeriodSelect } from '../components/ui/States';
import Badge from '../components/ui/Badge';
import { COLORS, categoryLabel, fmtDate, fmtEur, fmtInt, fmtPct } from '../utils/format';

const tickDate = (v) => fmtDate(v, { day: '2-digit', month: '2-digit' });
const num = (v) => (Array.isArray(v) ? `${fmtInt(v[0])} – ${fmtInt(v[1])}` : fmtInt(v));

function chartRows(fc) {
  const hist = fc.history.map((h) => ({ date: h.date, history: h.sales }));
  const future = fc.daily.map((d) => ({
    date: d.date, predicted: d.predicted_sales, band: [d.lower, d.upper], actual: d.actual_sales, seat: d.seat_order,
  }));
  // relie visuellement l'historique à la prévision
  if (hist.length) hist[hist.length - 1].predicted = hist[hist.length - 1].history;
  return [...hist, ...future];
}

const ProductSelect = ({ value, onChange }) => {
  const products = useApi(() => api.products(), []);
  return (
    <select value={value ?? ''} onChange={(e) => onChange(e.target.value ? Number(e.target.value) : null)}
            className="border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white max-w-xs">
      <option value="">Tous les produits</option>
      {products.data?.products.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
    </select>
  );
};

const Backtest = ({ productId }) => {
  const state = useApi(() => api.backtest(30, productId), [productId]);
  return (
    <Card className="mt-6">
      <CardHeader>
        <CardTitle>Fiabilité récente (backtest J+1, 30 jours)</CardTitle>
        <CardDescription>
          Chaque point est la prévision faite la veille, comparée aux ventes réelles.
        </CardDescription>
      </CardHeader>
      <AsyncContent state={state}>
        {(bt) => (
          <>
            <div className="flex gap-6 mb-4 text-sm">
              <div><span className="text-gray-600">Erreur LSTM (WAPE)</span> <span className="font-semibold">{fmtPct(bt.wape.lstm)}</span></div>
              <div><span className="text-gray-600">Moyenne 4 semaines</span> <span className="font-semibold">{fmtPct(bt.wape.moyenne_4_semaines)}</span></div>
            </div>
            <div className="h-64">
              <ResponsiveContainer>
                <ComposedChart data={bt.daily}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <XAxis dataKey="date" tickFormatter={tickDate} fontSize={12} />
                  <YAxis fontSize={12} />
                  <Tooltip labelFormatter={(v) => fmtDate(v)} formatter={(v, n) => [fmtInt(v), n]} />
                  <Legend />
                  <Line dataKey="actual_sales" name="Ventes réelles" stroke={COLORS.actual} strokeWidth={2} dot={false} />
                  <Line dataKey="predicted_sales" name="LSTM (J+1)" stroke={COLORS.ai} strokeWidth={2} dot={false} />
                  <Line dataKey="seat_forecast" name="Moyenne 4 semaines" stroke={COLORS.seat} strokeDasharray="5 4" dot={false} />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </>
        )}
      </AsyncContent>
    </Card>
  );
};

const Forecasts = () => {
  const [days, setDays] = useState(7);
  const [productId, setProductId] = useState(null);
  const state = useApi(() => api.forecasts(days, productId), [days, productId]);

  return (
    <>
      <PageHeader
        title="Prévisions"
        subtitle="Demande prévue par le LSTM à partir du lendemain de la date de référence"
        actions={<><ProductSelect value={productId} onChange={setProductId} /><PeriodSelect value={days} onChange={setDays} options={[7, 14]} /></>}
      />
      <AsyncContent state={state}>
        {(fc) => (
          <>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              {[
                ['Ventes prévues', fmtInt(fc.summary.predicted_sales)],
                ['CA prévu', fmtEur(fc.summary.predicted_revenue)],
                ['Pertes estimées', fmtInt(fc.summary.predicted_waste)],
                ['Coût des pertes estimé', fmtEur(fc.summary.predicted_waste_cost)],
              ].map(([label, value]) => (
                <Card key={label}><p className="text-sm text-gray-600">{label}</p><p className="text-xl font-bold mt-1 tabular-nums">{value}</p></Card>
              ))}
            </div>

            <Card>
              <CardHeader>
                <CardTitle>{productId ? fc.by_product[0]?.name : 'Tous les produits'} — du {fmtDate(fc.start_date)} au {fmtDate(fc.end_date)}</CardTitle>
                <CardDescription>
                  Zone bleue : intervalle de prévision (10e–90e centile des erreurs de validation).
                  Au-delà de J+1, la prévision réutilise ses propres valeurs comme historique.
                </CardDescription>
              </CardHeader>
              <div className="h-80">
                <ResponsiveContainer>
                  <ComposedChart data={chartRows(fc)}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="date" tickFormatter={tickDate} fontSize={12} />
                    <YAxis fontSize={12} />
                    <Tooltip labelFormatter={(v) => fmtDate(v)} formatter={(v, n) => [num(v), n]} />
                    <Legend />
                    <Area dataKey="band" name="Intervalle" fill={COLORS.band} stroke="none" fillOpacity={0.6} />
                    {productId && <Line dataKey="history" name="Historique" stroke={COLORS.muted} dot={false} />}
                    <Line dataKey="predicted" name="Prévision LSTM" stroke={COLORS.ai} strokeWidth={2} />
                    <Line dataKey="seat" name="Commande siège" stroke={COLORS.seat} strokeDasharray="5 4" dot={false} />
                    <Line dataKey="actual" name="Ventes réelles" stroke={COLORS.actual} strokeWidth={2} dot={{ r: 2 }} connectNulls={false} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <div className="mt-4 grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2 text-xs">
                {fc.daily.slice(0, 7).map((d) => (
                  <div key={d.date} className="border border-gray-200 rounded-lg p-2">
                    <p className="font-medium text-gray-900">{fmtDate(d.date)}</p>
                    <p className="text-gray-600">{d.context.temperature.toFixed(0)} °C · {d.context.rain.toFixed(1)} mm</p>
                    {d.context.is_holiday === 1 && <Badge variant="warning" size="small">Férié</Badge>}
                    {d.context.events.map((e) => <Badge key={e.name} variant="primary" size="small">{e.name}</Badge>)}
                  </div>
                ))}
              </div>
            </Card>

            {!productId && (
              <Card className="mt-6" padding={false}>
                <div className="p-6 pb-0">
                  <CardTitle>Détail par produit</CardTitle>
                  <CardDescription>Totaux sur {fc.days} jours. Commande IA = prévision + 5 %.</CardDescription>
                </div>
                <div className="overflow-x-auto mt-4">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-50 text-gray-500 text-left">
                      <tr>
                        <th className="px-6 py-3">Produit</th><th className="px-3 py-3">Catégorie</th>
                        <th className="px-3 py-3 text-right">Ventes prévues</th><th className="px-3 py-3 text-right">Pertes</th>
                        <th className="px-3 py-3 text-right">Commande IA</th><th className="px-3 py-3 text-right">Commande siège</th>
                        <th className="px-6 py-3 text-right">Écart</th>
                      </tr>
                    </thead>
                    <tbody>
                      {fc.by_product.map((p) => {
                        const diff = p.ai_order - p.seat_order;
                        return (
                          <tr key={p.product_id} className="border-t hover:bg-gray-50 cursor-pointer" onClick={() => setProductId(p.product_id)}>
                            <td className="px-6 py-2 font-medium">{p.name}</td>
                            <td className="px-3 py-2 text-gray-600">{categoryLabel(p.category)}</td>
                            <td className="px-3 py-2 text-right tabular-nums">{fmtInt(p.predicted_sales)}</td>
                            <td className="px-3 py-2 text-right tabular-nums">{fmtInt(p.predicted_waste)}</td>
                            <td className="px-3 py-2 text-right tabular-nums">{fmtInt(p.ai_order)}</td>
                            <td className="px-3 py-2 text-right tabular-nums">{fmtInt(p.seat_order)}</td>
                            <td className={`px-6 py-2 text-right tabular-nums font-medium ${diff < 0 ? 'text-green-700' : diff > 0 ? 'text-amber-700' : 'text-gray-500'}`}>
                              {diff > 0 ? '+' : ''}{fmtInt(diff)}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </Card>
            )}
            <Backtest productId={productId} />
          </>
        )}
      </AsyncContent>
    </>
  );
};

export default Forecasts;
