// =============================================================
// pages/Dashboard.jsx — Main discovery page
// =============================================================

import { useState } from 'react';
import { runDiscovery, downloadPdfUrl, downloadCsvUrl } from '../api';
import WeightSliders from '../components/WeightSliders';
import CandidatesTable from '../components/CandidatesTable';
import ChartGrid from '../components/ChartGrid';
import StatusLog from '../components/StatusLog';

export default function Dashboard() {
  const [loading, setLoading] = useState(false);
  const [params, setParams] = useState({
    smiles_input: '',
    mode: 'Generate new materials',
    w_bio: 0.22,
    w_sa: 0.13,
    w_tox: 0.18,
    w_rec: 0.13,
    w_str: 0.10,
    w_tg: 0.12,
    w_sol: 0.12,
    temperature: 1.0,
    n_candidates: 500,
  });

  const [result, setResult] = useState(null);

  const handleWeightChange = (key, val) => {
    setParams(p => ({ ...p, [key]: val }));
  };

  const handleRun = async () => {
    setLoading(true);
    setResult({ status: 'running', log: ['Starting discovery pipeline...'] });
    
    try {
      const res = await runDiscovery(params);
      setResult(res);
    } catch (e) {
      setResult(prev => ({
        ...prev,
        status: 'error',
        log: [...(prev?.log || []), `❌ Error: ${e.message}`]
      }));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 animate-fade-in flex-col gap-6" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-accent mb-2">Discovery Dashboard</h2>
          <p className="text-muted text-sm m-0">Generate, perturb, and score novel sustainable polymers.</p>
        </div>
      </div>

      <div className="grid" style={{ display: 'grid', gridTemplateColumns: '320px 1fr', gap: '24px' }}>
        
        {/* LEFT PANEL: Controls */}
        <div className="flex-col gap-4">
          <div className="card glass-card">
            <h3 className="mb-3 text-sm uppercase text-secondary">Input</h3>
            <div className="mb-4">
              <label className="label">Seed SMILES (optional)</label>
              <textarea 
                className="textarea" 
                placeholder="Leave blank for fresh generation, or paste SMILES to perturb..."
                value={params.smiles_input}
                onChange={e => setParams({...params, smiles_input: e.target.value})}
              />
            </div>
            
            <div className="mb-4">
              <label className="label">Discovery Mode</label>
              <select 
                className="input"
                value={params.mode}
                onChange={e => setParams({...params, mode: e.target.value})}
              >
                <option value="Generate new materials">Generate new materials</option>
                <option value="Encode & perturb input">Encode & perturb input</option>
                <option value="Both">Both</option>
              </select>
            </div>
          </div>

          <div className="card glass-card">
            <WeightSliders weights={params} onChange={handleWeightChange} />
          </div>

          <div className="card glass-card">
            <h3 className="mb-3 text-sm uppercase text-secondary">Generation Settings</h3>
            
            <div className="mb-4">
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium">Temperature</label>
                <span className="text-muted text-sm mono">{params.temperature.toFixed(2)}</span>
              </div>
              <input 
                type="range" min="0.4" max="2.0" step="0.05"
                value={params.temperature}
                onChange={e => setParams({...params, temperature: parseFloat(e.target.value)})}
              />
              <div className="text-muted" style={{fontSize: '0.7rem', marginTop: '4px'}}>↑ Higher = more diverse, lower validity</div>
            </div>

            <div className="mb-4">
              <div className="flex justify-between mb-1">
                <label className="text-sm font-medium">N Candidates</label>
                <span className="text-muted text-sm mono">{params.n_candidates}</span>
              </div>
              <input 
                type="range" min="100" max="3000" step="100"
                value={params.n_candidates}
                onChange={e => setParams({...params, n_candidates: parseInt(e.target.value)})}
              />
            </div>
            
            <button 
              className="btn btn-primary w-full mt-2" 
              onClick={handleRun}
              disabled={loading}
              style={{ padding: '14px', fontSize: '1.05rem', letterSpacing: '0.05em' }}
            >
              {loading ? <span className="flex items-center gap-2"><div className="spinner" style={{width:'16px',height:'16px'}}></div> Running...</span> : 'RUN DISCOVERY'}
            </button>
          </div>
        </div>

        {/* RIGHT PANEL: Results */}
        <div className="flex-col gap-4">
          <StatusLog log={result?.log} loading={loading} />
          
          {result?.candidates && result.candidates.length > 0 && (
            <div className="card glass-card animate-fade-in" style={{ padding: '16px' }}>
              <h3 className="text-sm uppercase tracking-wide text-secondary mb-2" style={{ letterSpacing: '0.05em' }}>Top Candidates</h3>
              <CandidatesTable candidates={result.candidates} />
            </div>
          )}

          <div className="card glass-card" style={{ minHeight: '400px' }}>
            <div className="flex justify-between items-center mb-2">
              <h3 className="text-sm uppercase tracking-wide text-secondary m-0" style={{ letterSpacing: '0.05em' }}>Visualisations</h3>
              
              <div className="flex gap-2">
                {result?.pdf_available && (
                  <a href={downloadPdfUrl} target="_blank" rel="noreferrer" className="btn btn-sm btn-secondary">
                    PDF Report
                  </a>
                )}
                {result?.csv_available && (
                  <a href={downloadCsvUrl} target="_blank" rel="noreferrer" className="btn btn-sm btn-secondary">
                    CSV Export
                  </a>
                )}
              </div>
            </div>
            
            <ChartGrid charts={result?.charts} />
          </div>
        </div>

      </div>
    </div>
  );
}
