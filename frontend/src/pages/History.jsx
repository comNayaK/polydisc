// =============================================================
// pages/History.jsx — Session history and Favourites
// =============================================================

import { useState, useEffect } from 'react';
import { getHistory, clearHistory, getFavourites, clearFavourites, exportFavouritesUrl } from '../api';

export default function History() {
  const [history, setHistory] = useState([]);
  const [favourites, setFavourites] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchData = async () => {
    setLoading(true);
    try {
      const [histRes, favRes] = await Promise.all([getHistory(), getFavourites()]);
      setHistory(histRes.entries || []);
      setFavourites(favRes.favourites || []);
    } catch (e) {
      console.error('Failed to fetch history data', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, []);

  const handleClearHistory = async () => {
    await clearHistory();
    setHistory([]);
  };

  const handleClearFavourites = async () => {
    await clearFavourites();
    setFavourites([]);
  };

  if (loading) {
    return <div className="p-6 flex justify-center items-center h-full"><div className="spinner"></div></div>;
  }

  return (
    <div className="p-6 animate-fade-in flex-col gap-6" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      <div>
        <h2 className="text-accent mb-2">Session Data</h2>
        <p className="text-muted text-sm m-0">Your run history and starred molecules for this session.</p>
      </div>

      <div className="grid-2">
        {/* Favourites */}
        <div className="card glass-card flex-col">
          <div className="flex justify-between items-center mb-4">
            <h3 className="m-0">Favourites</h3>
            <div className="flex gap-2">
              <a href={exportFavouritesUrl} target="_blank" rel="noreferrer" className="btn btn-sm btn-secondary" disabled={favourites.length === 0}>
                Export CSV
              </a>
              <button className="btn btn-sm btn-danger" onClick={handleClearFavourites} disabled={favourites.length === 0}>
                Clear
              </button>
            </div>
          </div>
          
          <div className="overflow-auto flex-1" style={{ maxHeight: '500px' }}>
            {favourites.length === 0 ? (
              <div className="text-muted text-center mt-4">No favourites saved yet.</div>
            ) : (
              <table className="data-table">
                <thead>
                  <tr>
                    <th>SMILES</th>
                    <th>GS</th>
                  </tr>
                </thead>
                <tbody>
                  {favourites.map((f, i) => (
                    <tr key={i}>
                      <td title={f.smiles}>{f.smiles.length > 25 ? f.smiles.substring(0, 25) + '…' : f.smiles}</td>
                      <td className="text-accent font-bold">{f.green_score.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* History */}
        <div className="card glass-card flex-col">
          <div className="flex justify-between items-center mb-4">
            <h3 className="m-0">Run History</h3>
            <div className="flex gap-2">
              <button className="btn btn-sm btn-secondary" onClick={fetchData}>
                Refresh
              </button>
              <button className="btn btn-sm btn-danger" onClick={handleClearHistory} disabled={history.length === 0}>
                Clear
              </button>
            </div>
          </div>
          
          <div className="overflow-auto flex-1" style={{ maxHeight: '500px' }}>
            {history.length === 0 ? (
              <div className="text-muted text-center mt-4">No runs yet this session.</div>
            ) : (
              <div className="flex-col gap-2">
                {history.map((h, i) => (
                  <div key={i} className="p-3 bg-panel rounded border" style={{ background: 'var(--bg-panel)', borderRadius: 'var(--radius-md)', borderColor: 'var(--border-color)' }}>
                    <div className="flex justify-between mb-1">
                      <span className="text-sm font-bold text-secondary">{h.time}</span>
                      <span className="badge badge-info">{h.mode}</span>
                    </div>
                    <div className="text-sm mono text-muted mb-1">Top GS: <span className="text-accent">{h.top_gs.toFixed(4)}</span> | N: {h.n_candidates}</div>
                    <div className="text-xs mono" style={{ wordBreak: 'break-all' }}>{h.top_smiles}</div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

    </div>
  );
}
