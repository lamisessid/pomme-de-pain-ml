const BASE_URL = (import.meta.env.VITE_API_URL || 'http://localhost:8642').replace(/\/$/, '');

async function request(path, { method = 'GET', params, body } = {}) {
  const url = new URL(`${BASE_URL}${path}`);
  Object.entries(params || {}).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, v);
  });
  const res = await fetch(url, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch {
      /* réponse non JSON */
    }
    throw new Error(`${res.status} — ${detail}`);
  }
  return res.json();
}

export const api = {
  health: () => request('/health'),
  dashboard: () => request('/api/v1/dashboard'),
  forecasts: (days, productId) => request('/api/v1/forecasts', { params: { days, product_id: productId } }),
  accuracy: () => request('/api/v1/forecasts/accuracy'),
  backtest: (days, productId) => request('/api/v1/forecasts/backtest', { params: { days, product_id: productId } }),
  losses: (days) => request('/api/v1/losses', { params: { days } }),
  comparison: (days) => request('/api/v1/comparison', { params: { days } }),
  alerts: (status = 'active') => request('/api/v1/alerts', { params: { status } }),
  alertStats: () => request('/api/v1/alerts/stats'),
  resolveAlert: (id) => request(`/api/v1/alerts/${encodeURIComponent(id)}/resolve`, { method: 'POST', body: {} }),
  reopenAlert: (id) => request(`/api/v1/alerts/${encodeURIComponent(id)}/reopen`, { method: 'POST' }),
  products: (params) => request('/api/v1/products', { params }),
  product: (id) => request(`/api/v1/products/${id}`),
  events: () => request('/api/v1/events'),
  chat: (message) => request('/api/v1/chatbot/message', { method: 'POST', body: { message } }),
  chatSuggestions: () => request('/api/v1/chatbot/suggestions'),
};
