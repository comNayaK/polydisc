// =============================================================
// pages/Analysis.jsx — Comparison and Interpolation
// =============================================================

import { useState } from 'react';
import { runCompare, runInterpolate } from '../api';

export default function Analysis() {
  const [cmp, setCmp] = useState({ a: '', b: '' });
  const [cmpLoading, setCmpLoading] = useState(false);
  const [cmpResult, setCmpResult] = useState(null);

  const [intp, setIntp] = useState({ a: '', b: '', steps: 8 });
  const [intpLoading, setIntpLoading] = useState(false);
  const [intpResult, setIntpResult] = useState(null);

  const handleCompare = async () => {
    if (!cmp.a || !cmp.b) return;
    setCmpLoading(true);
    try {
      const res = await runCompare(cmp.a, cmp.b);
      setCmpResult(res);
    } catch (e) {
      setCmpResult({ text: `❌ Error: ${e.message}`, chart: null });
    } finally {
      setCmpLoading(false);
    }
  };

  const handleInterpolate = async () => {
    if (!intp.a || !intp.b) return;
    setIntpLoading(true);
    try {
      const res = await runInterpolate(intp.a, intp.b, intp.steps);
      setIntpResult(res);
    } catch (e) {
      setIntpResult({ text: `❌ Error: ${e.message}` });
    } finally {
      setIntpLoading(false);
    }
  };

  return (
    <div className="p-6 animate-fade-in flex-col gap-6" style={{ padding: '24px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
      
      <div>
        <h2 className="text-accent mb-2">Detailed Analysis</h2>
        <p className="text-muted text-sm m-0">Compare specific molecules or explore the latent space between them.</p>
      </div>

      {/* Comparison */}
      <div className="card glass-card">
        <h3 className="mb-4">Side-by-Side Comparison</h3>
        
        <div className="grid-2 mb-4">
          <div>
            <label className="label">Molecule A (SMILES)</label>
            <input 
              className="input" 
              placeholder="e.g. CC(=O)Oc1ccc(cc1)OC(C)=O"
              value={cmp.a} onChange={e => setCmp({ ...cmp, a: e.target.value })}
            />
          </div>
          <div>
            <label className="label">Molecule B (SMILES)</label>
            <input 
              className="input" 
              placeholder="e.g. OC(=O)CCCCC(=O)OCCO"
              value={cmp.b} onChange={e => setCmp({ ...cmp, b: e.target.value })}
            />
          </div>
        </div>
        
        <button 
          className="btn btn-primary mb-4" 
          onClick={handleCompare}
          disabled={cmpLoading || !cmp.a || !cmp.b}
        >
          {cmpLoading ? 'Comparing...' : 'Compare Molecules'}
        </button>

        {cmpResult && (
          <div className="mt-4 border-t pt-4" style={{ borderColor: 'var(--border-color)' }}>
            <div className="grid-2">
              <pre className="p-4 bg-input rounded text-sm mono text-muted overflow-auto" style={{background:'var(--bg-input)', borderRadius:'var(--radius-md)'}}>
                {cmpResult.text}
              </pre>
              {cmpResult.chart && (
                <div className="flex items-center justify-center p-2 bg-panel rounded" style={{background:'var(--bg-panel)', borderRadius:'var(--radius-md)'}}>
                  <img src={`data:image/png;base64,${cmpResult.chart}`} alt="Comparison" style={{maxHeight:'320px', maxWidth:'100%', objectFit:'contain'}} />
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Interpolation */}
      <div className="card glass-card">
        <h3 className="mb-2">Latent Space Interpolation</h3>
        <p className="text-muted text-sm mb-4">Generate transitional molecules along the path between two SMILES in the VAE latent space.</p>
        
        <div className="grid-3 mb-4 items-end">
          <div>
            <label className="label">Start SMILES</label>
            <input 
              className="input" 
              value={intp.a} onChange={e => setIntp({ ...intp, a: e.target.value })}
            />
          </div>
          <div>
            <label className="label">End SMILES</label>
            <input 
              className="input" 
              value={intp.b} onChange={e => setIntp({ ...intp, b: e.target.value })}
            />
          </div>
          <div>
            <label className="label">Steps ({intp.steps})</label>
            <input 
              type="range" min="4" max="16" step="1"
              value={intp.steps} onChange={e => setIntp({ ...intp, steps: parseInt(e.target.value) })}
            />
          </div>
        </div>
        
        <button 
          className="btn btn-primary mb-4" 
          onClick={handleInterpolate}
          disabled={intpLoading || !intp.a || !intp.b}
        >
          {intpLoading ? 'Interpolating...' : 'Interpolate Path'}
        </button>

        {intpResult && (
          <div className="mt-4 border-t pt-4" style={{ borderColor: 'var(--border-color)' }}>
            <pre className="p-4 bg-input rounded text-sm mono text-muted mb-4 overflow-auto" style={{background:'var(--bg-input)', borderRadius:'var(--radius-md)', maxHeight:'200px'}}>
              {intpResult.text}
            </pre>
            <div className="grid-2">
              {intpResult.bar_chart && (
                <div className="p-2 bg-panel rounded">
                  <img src={`data:image/png;base64,${intpResult.bar_chart}`} alt="Path Bar" style={{width:'100%', objectFit:'contain'}} />
                </div>
              )}
              {intpResult.radar_chart && (
                <div className="p-2 bg-panel rounded">
                  <img src={`data:image/png;base64,${intpResult.radar_chart}`} alt="Path Radar" style={{width:'100%', objectFit:'contain'}} />
                </div>
              )}
            </div>
          </div>
        )}
      </div>

    </div>
  );
}
