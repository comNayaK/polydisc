// =============================================================
// api.js — API client for Polymer Discovery AI backend
// =============================================================

const API_BASE = 'http://localhost:8000/api';

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  const config = {
    credentials: 'include',
    headers: { 'Content-Type': 'application/json' },
    ...options,
  };

  const resp = await fetch(url, config);
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ error: resp.statusText }));
    throw new Error(err.error || err.detail || `HTTP ${resp.status}`);
  }

  const contentType = resp.headers.get('content-type');
  if (contentType && contentType.includes('application/json')) {
    return resp.json();
  }
  return resp;
}

// ── Config
export const getConfig = () => request('/config');

// ── Discovery
export const runDiscovery = (params) =>
  request('/discovery', { method: 'POST', body: JSON.stringify(params) });

// ── Compare
export const runCompare = (smiles_a, smiles_b) =>
  request('/compare', {
    method: 'POST',
    body: JSON.stringify({ smiles_a, smiles_b }),
  });

// ── Interpolate
export const runInterpolate = (smiles_a, smiles_b, steps) =>
  request('/interpolate', {
    method: 'POST',
    body: JSON.stringify({ smiles_a, smiles_b, steps }),
  });

// ── Favourites
export const addFavourite = (smiles) =>
  request('/favourite', { method: 'POST', body: JSON.stringify({ smiles }) });

export const getFavourites = () => request('/favourites');

export const clearFavourites = () => request('/favourites/clear', { method: 'DELETE' });

export const exportFavouritesUrl = `${API_BASE}/favourites/export`;

// ── History
export const getHistory = () => request('/history');

export const clearHistory = () => request('/history/clear', { method: 'DELETE' });

// ── Downloads
export const downloadPdfUrl = `${API_BASE}/download/pdf`;
export const downloadCsvUrl = `${API_BASE}/download/csv`;

// ── Theme
export const toggleTheme = () => request('/theme/toggle', { method: 'POST' });

// ── AI Chat
export const sendChatMessage = (messages) =>
  request('/chat', { method: 'POST', body: JSON.stringify({ messages }) });
