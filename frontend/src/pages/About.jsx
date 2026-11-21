import React from 'react';
import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';

import { api } from '../api/client';
import useApi from '../hooks/useApi';
import { Card, CardHeader, CardTitle, CardDescription } from '../components/ui/Card';
import { AsyncContent, PageHeader } from '../components/ui/States';
import { COLORS, categoryLabel, fmtEur, fmtInt, fmtPct } from '../utils/format';

const MODELS = [
  ['lstm', 'LSTM multi-tâches'],
  ['moyenne_4_semaines', 'Moyenne des 4 mêmes jours'],
  ['naive_j7', 'Naïf (J-7)'],
];

const About = () => {
  const state = useApi(() => api.accuracy(), []);

  return (
    <>
      <PageHeader title="Modèle & méthode" subtitle="Comment les prévisions sont produites et évaluées" />
      <AsyncContent state={state}>
        {(m) => (
          <>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Erreur sur la période de test ({m.split.test})</CardTitle>
                  <CardDescription>
                    Prévision {m.horizon}. WAPE = somme des erreurs absolues / somme des ventes (plus bas = mieux).
                  </CardDescription>
                </CardHeader>
                <table className="w-full text-sm">
                  <thead className="text-gray-500 text-left"><tr><th className="py-2">Méthode</th><th className="py-2 text-right">WAPE</th><th className="py-2 text-right">MAE (unités)</th><th className="py-2 text-right">RMSE</th><th className="py-2 text-right">Biais</th></tr></thead>
                  <tbody>
                    {MODELS.map(([key, label]) => (
                      <tr key={key} className={`border-t ${key === 'lstm' ? 'font-semibold' : ''}`}>
                        <td className="py-2">{label}</td>
                        <td className="py-2 text-right tabular-nums">{fmtPct(m.sales[key].wape)}</td>
                        <td className="py-2 text-right tabular-nums">{m.sales[key].mae.toFixed(2)}</td>
                        <td className="py-2 text-right tabular-nums">{m.sales[key].rmse.toFixed(2)}</td>
                        <td className="py-2 text-right tabular-nums">{m.sales[key].bias.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-6 text-sm">
                  <div className="bg-gray-50 rounded-lg p-3">
                    <p className="text-gray-600">Pertes — WAPE</p>
                    <p className="font-semibold">LSTM {fmtPct(m.waste.lstm.wape)} · J-7 {fmtPct(m.waste.naive_j7.wape)}</p>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-3">
                    <p className="text-gray-600">Classe de risque — F1 macro</p>
                    <p className="font-semibold">{m.risk.lstm_macro_f1.toFixed(2)} (exactitude {fmtPct(m.risk.lstm_accuracy * 100)})</p>
                  </div>
                  <div className="bg-gray-50 rounded-lg p-3">
                    <p className="text-gray-600">Intervalle « 80 % » — couverture réelle</p>
                    <p className="font-semibold">{fmtPct(m.interval_80.coverage_test_pct)}</p>
                  </div>
                </div>
              </Card>

              <Card>
                <CardHeader><CardTitle>Simulation de commandes (test)</CardTitle><CardDescription>{m.order_simulation.description}</CardDescription></CardHeader>
                <table className="w-full text-sm">
                  <thead className="text-gray-500"><tr><th /><th className="text-right py-1">IA</th><th className="text-right py-1">Siège</th></tr></thead>
                  <tbody>
                    {[
                      ['Invendus (unités)', 'unsold_units', fmtInt],
                      ['Coût des invendus', 'unsold_cost_eur', fmtEur],
                      ['Ventes manquées', 'missed_sales_units', fmtInt],
                      ['Marge manquée', 'missed_margin_eur', fmtEur],
                      ['Jours en rupture', 'stockout_days_pct', (v) => fmtPct(v)],
                    ].map(([label, key, fmt]) => (
                      <tr key={key} className="border-t">
                        <td className="py-2 text-gray-600">{label}</td>
                        <td className="py-2 text-right tabular-nums">{fmt(m.order_simulation.ia[key])}</td>
                        <td className="py-2 text-right tabular-nums">{fmt(m.order_simulation.siege[key])}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
              <Card>
                <CardHeader><CardTitle>WAPE par catégorie</CardTitle></CardHeader>
                <div className="h-72">
                  <ResponsiveContainer>
                    <BarChart data={Object.entries(m.sales_wape_by_category).map(([c, v]) => ({ name: categoryLabel(c), LSTM: v.lstm, 'Moyenne 4 sem.': v.moyenne_4_semaines }))}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="name" fontSize={11} interval={0} angle={-30} textAnchor="end" height={60} />
                      <YAxis fontSize={12} tickFormatter={(v) => `${v} %`} />
                      <Tooltip formatter={(v, n) => [fmtPct(v), n]} />
                      <Legend />
                      <Bar dataKey="LSTM" fill={COLORS.ai} radius={[3, 3, 0, 0]} />
                      <Bar dataKey="Moyenne 4 sem." fill={COLORS.seat} radius={[3, 3, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </Card>
              <Card>
                <CardHeader><CardTitle>Courbes d'apprentissage</CardTitle><CardDescription>Entraîné le {new Date(m.trained_at).toLocaleString('fr-FR')} en {m.training_seconds} s</CardDescription></CardHeader>
                <div className="h-72">
                  <ResponsiveContainer>
                    <LineChart data={m.history}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="epoch" fontSize={12} />
                      <YAxis fontSize={12} domain={['auto', 'auto']} />
                      <Tooltip />
                      <Legend />
                      <Line dataKey="train_loss" name="Entraînement" stroke={COLORS.ai} dot={false} />
                      <Line dataKey="val_loss" name="Validation" stroke={COLORS.seat} dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </Card>
            </div>

            <Card>
              <CardTitle>Méthode</CardTitle>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-4 text-sm text-gray-700">
                <div>
                  <p className="font-medium text-gray-900 mb-1">Données</p>
                  <p>Jeu de données simulé (fictif mais réaliste), 50 produits × 730 jours (2022–2023) : ventes, pertes, météo, jours fériés et événements SQY (rugby, examens UVSQ, grèves RER…).
                    Découpage temporel : entraînement {m.split.train}, validation {m.split.validation}, test {m.split.test}.</p>
                </div>
                <div>
                  <p className="font-medium text-gray-900 mb-1">Modèle</p>
                  <p>LSTM 2 couches + attention temporelle + connexion résiduelle, trois têtes : ventes, pertes, classe de risque.
                    Entrée : 28 jours de contexte ; chaque jour contient les ventes de la veille, de J-7, la moyenne glissante, le calendrier,
                    la météo, les événements et les caractéristiques du produit. Ventes normalisées par produit.</p>
                </div>
                <div>
                  <p className="font-medium text-gray-900 mb-1">Limites</p>
                  <p>La météo des jours futurs est remplacée par la climatologie si elle n'est pas connue. L'intervalle calibré sur l'été
                    est trop étroit en fin d'année ({fmtPct(m.interval_80.coverage_test_pct)} de couverture au lieu de 80 %).
                    La simulation de commandes prend les ventes observées comme demande réelle : la demande perdue lors des ruptures passées n'est pas visible.</p>
                </div>
              </div>
            </Card>
          </>
        )}
      </AsyncContent>
    </>
  );
};

export default About;
