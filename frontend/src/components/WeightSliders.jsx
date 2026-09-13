// =============================================================
// components/WeightSliders.jsx — Green Score weight controls
// =============================================================

export default function WeightSliders({ weights, onChange }) {
  const sliders = [
    { id: 'w_bio', label: 'Biodegradability' },
    { id: 'w_sa', label: 'Synthesizability' },
    { id: 'w_tox', label: 'Low Toxicity' },
    { id: 'w_rec', label: 'Recyclability' },
    { id: 'w_str', label: 'Mech. Strength' },
    { id: 'w_tg', label: 'Glass Trans. Tg' },
    { id: 'w_sol', label: 'Solubility' },
  ];

  return (
    <div className="weight-sliders">
      <div className="flex justify-between items-center mb-4">
        <h3 className="text-secondary text-sm uppercase tracking-wide m-0" style={{ fontSize: '0.8rem', letterSpacing: '0.05em' }}>Green Score Weights</h3>
        <span className="badge badge-info" style={{ fontSize: '0.65rem' }}>Auto-normalised</span>
      </div>
      
      <div className="flex-col gap-3">
        {sliders.map((s) => (
          <div key={s.id} className="slider-group">
            <div className="flex justify-between mb-1">
              <label className="text-sm font-medium">{s.label}</label>
              <span className="text-muted text-sm mono">
                {Number(weights[s.id] || 0).toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.01"
              value={weights[s.id] || 0}
              onChange={(e) => onChange(s.id, parseFloat(e.target.value))}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
