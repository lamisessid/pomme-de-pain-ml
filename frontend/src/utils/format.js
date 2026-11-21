const eur = new Intl.NumberFormat('fr-FR', { style: 'currency', currency: 'EUR', maximumFractionDigits: 0 });
const eur2 = new Intl.NumberFormat('fr-FR', { style: 'currency', currency: 'EUR' });
const int = new Intl.NumberFormat('fr-FR', { maximumFractionDigits: 0 });

export const fmtEur = (v, decimals = false) => (v == null ? '—' : (decimals ? eur2 : eur).format(v));
export const fmtInt = (v) => (v == null ? '—' : int.format(v));
export const fmtPct = (v, digits = 1) => (v == null ? '—' : `${v.toFixed(digits).replace('.', ',')} %`);
export const fmtSignedPct = (v) => (v == null ? '—' : `${v > 0 ? '+' : ''}${fmtPct(v)}`);

export const fmtDate = (iso, opts = { weekday: 'short', day: '2-digit', month: '2-digit' }) =>
  iso ? new Date(`${iso}T00:00:00`).toLocaleDateString('fr-FR', opts) : '—';
export const fmtDateLong = (iso) =>
  fmtDate(iso, { weekday: 'long', day: 'numeric', month: 'long', year: 'numeric' });

export const CATEGORY_LABELS = {
  boisson: 'Boissons', chaud: 'Plats chauds', dessert: 'Desserts', divers: 'Divers', menu: 'Formules',
  pain: 'Pains', sale: 'Salés', sandwich: 'Sandwichs', special: 'Spécial', viennoiserie: 'Viennoiseries',
};
export const categoryLabel = (c) => CATEGORY_LABELS[c] || c;

export const ALERT_TYPES = { rupture: 'Rupture', gaspillage: 'Gaspillage', evenement: 'Événement' };

/** Couleurs partagées par tous les graphiques. */
export const COLORS = {
  actual: '#1f2937',
  ai: '#0284c7',
  seat: '#d97706',
  waste: '#dc2626',
  band: '#bae6fd',
  muted: '#9ca3af',
};
