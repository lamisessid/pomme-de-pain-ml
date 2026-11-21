import React, { useState } from 'react';
import {
  Area, CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from 'recharts';
import { Search, X } from 'lucide-react';

import { api } from '../api/client';
import useApi from '../hooks/useApi';
import { Card, CardTitle, CardDescription } from '../components/ui/Card';
import { AsyncContent, PageHeader } from '../components/ui/States';
import Badge from '../components/ui/Badge';
import { COLORS, categoryLabel, fmtDate, fmtEur, fmtInt, fmtPct, fmtSignedPct } from '../utils/format';

const RISK = { low: ['Faible', 'success'], medium: ['Moyen', 'warning'], high: ['Élevé', 'error'] };

const ProductDetail = ({ id, onClose }) => {
  const state = useApi(() => api.product(id), [id]);
  return (
    <Card className="mb-6 border-primary-200">
      <AsyncContent state={state}>
        {(d) => {
          const rows = [
            ...d.history.map((h) => ({ date: h.date, sales: h.sales, waste: h.waste })),
            ...d.forecast.map((f) => ({ date: f.date, predicted: f.predicted_sales, band: [f.lower, f.upper] })),
          ];
          return (
            <>
              <div className="flex items-start justify-between mb-4">
                <div>
                  <CardTitle>{d.product.name}</CardTitle>
                  <CardDescription>
                    {categoryLabel(d.product.category)} · prix {fmtEur(d.product.price, true)} · coût {fmtEur(d.product.cost, true)} ·
                    durée de vie {d.product.shelf_life_days} j
                  </CardDescription>
                </div>
                <button onClick={onClose} className="p-1 hover:bg-gray-100 rounded" aria-label="Fermer"><X className="h-5 w-5" /></button>
              </div>
              <div className="h-64">
                <ResponsiveContainer>
                  <ComposedChart data={rows}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <XAxis dataKey="date" tickFormatter={(v) => fmtDate(v, { day: '2-digit', month: '2-digit' })} fontSize={12} />
                    <YAxis fontSize={12} />
                    <Tooltip labelFormatter={(v) => fmtDate(v)}
                             formatter={(v, n) => [Array.isArray(v) ? `${fmtInt(v[0])} – ${fmtInt(v[1])}` : fmtInt(v), n]} />
                    <Legend />
                    <Area dataKey="band" name="Intervalle" fill={COLORS.band} stroke="none" fillOpacity={0.6} />
                    <Line dataKey="sales" name="Ventes (60 j)" stroke={COLORS.actual} dot={false} />
                    <Line dataKey="waste" name="Pertes" stroke={COLORS.waste} dot={false} />
                    <Line dataKey="predicted" name="Prévision" stroke={COLORS.ai} strokeWidth={2} />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
              <div className="overflow-x-auto mt-4">
                <table className="w-full text-sm">
                  <thead className="text-gray-500 text-left"><tr><th className="py-2">Jour</th><th className="py-2 text-right">Prévision</th><th className="py-2 text-right">Commande IA</th><th className="py-2 text-right">Commande siège</th><th className="py-2 text-right">Risque de perte</th></tr></thead>
                  <tbody>
                    {d.forecast.map((f) => (
                      <tr key={f.date} className="border-t">
                        <td className="py-2">{fmtDate(f.date)}</td>
                        <td className="py-2 text-right tabular-nums">{fmtInt(f.predicted_sales)} <span className="text-gray-400">({fmtInt(f.lower)}–{fmtInt(f.upper)})</span></td>
                        <td className="py-2 text-right tabular-nums">{f.ai_order}</td>
                        <td className="py-2 text-right tabular-nums">{f.seat_order}</td>
                        <td className="py-2 text-right"><Badge variant={RISK[f.risk][1]}>{RISK[f.risk][0]}</Badge></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          );
        }}
      </AsyncContent>
    </Card>
  );
};

const Products = () => {
  const [category, setCategory] = useState('');
  const [q, setQ] = useState('');
  const [selected, setSelected] = useState(null);
  const [sort, setSort] = useState('revenue_30d');
  const state = useApi(() => api.products({ category }), [category]);

  return (
    <>
      <PageHeader title="Produits" subtitle="Catalogue du point de vente et performance sur 30 jours" />
      {selected && <ProductDetail id={selected} onClose={() => setSelected(null)} />}
      <AsyncContent state={state}>
        {(d) => {
          const list = d.products
            .filter((p) => p.name.toLowerCase().includes(q.toLowerCase()))
            .sort((a, b) => b[sort] - a[sort]);
          return (
            <Card padding={false}>
              <div className="p-4 flex flex-wrap gap-3 border-b border-gray-200">
                <div className="relative">
                  <Search className="h-4 w-4 text-gray-400 absolute left-3 top-1/2 -translate-y-1/2" />
                  <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Rechercher un produit"
                         className="pl-9 pr-3 py-2 border border-gray-300 rounded-lg text-sm" />
                </div>
                <select value={category} onChange={(e) => setCategory(e.target.value)} className="border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white">
                  <option value="">Toutes les catégories</option>
                  {d.categories.map((c) => <option key={c} value={c}>{categoryLabel(c)}</option>)}
                </select>
                <select value={sort} onChange={(e) => setSort(e.target.value)} className="border border-gray-300 rounded-lg px-3 py-2 text-sm bg-white">
                  <option value="revenue_30d">Trier par CA</option>
                  <option value="sales_30d">Trier par ventes</option>
                  <option value="waste_rate_30d">Trier par taux de perte</option>
                  <option value="margin_pct">Trier par marge</option>
                </select>
                <span className="ml-auto self-center text-sm text-gray-500">{list.length} produits · 30 j au {fmtDate(d.end)}</span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50 text-gray-500 text-left">
                    <tr>
                      <th className="px-6 py-3">Produit</th><th className="px-3 py-3">Catégorie</th><th className="px-3 py-3 text-right">Prix</th>
                      <th className="px-3 py-3 text-right">Marge</th><th className="px-3 py-3 text-right">Ventes</th><th className="px-3 py-3 text-right">Tendance</th>
                      <th className="px-3 py-3 text-right">CA</th><th className="px-6 py-3 text-right">Taux de perte</th>
                    </tr>
                  </thead>
                  <tbody>
                    {list.map((p) => (
                      <tr key={p.id} onClick={() => { setSelected(p.id); window.scrollTo({ top: 0, behavior: 'smooth' }); }}
                          className={`border-t cursor-pointer hover:bg-gray-50 ${selected === p.id ? 'bg-primary-50' : ''}`}>
                        <td className="px-6 py-2 font-medium">{p.name}</td>
                        <td className="px-3 py-2 text-gray-600">{categoryLabel(p.category)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{fmtEur(p.price, true)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{fmtPct(p.margin_pct, 0)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{fmtInt(p.sales_30d)}</td>
                        <td className={`px-3 py-2 text-right tabular-nums ${p.trend_pct >= 0 ? 'text-green-700' : 'text-red-700'}`}>{fmtSignedPct(p.trend_pct)}</td>
                        <td className="px-3 py-2 text-right tabular-nums">{fmtEur(p.revenue_30d)}</td>
                        <td className={`px-6 py-2 text-right tabular-nums ${p.waste_rate_30d > 12 ? 'text-red-700 font-medium' : ''}`}>{fmtPct(p.waste_rate_30d)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          );
        }}
      </AsyncContent>
    </>
  );
};

export default Products;
