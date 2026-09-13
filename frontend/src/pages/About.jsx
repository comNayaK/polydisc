// =============================================================
// pages/About.jsx — Architecture and Methodology
// =============================================================

export default function About() {
  return (
    <div className="p-6 animate-fade-in overflow-auto" style={{ padding: '24px' }}>
      <div className="max-w-4xl mx-auto flex-col gap-6" style={{ maxWidth: '900px', display: 'flex', flexDirection: 'column', gap: '24px' }}>
        
        <div>
          <h2 className="text-accent mb-2">About PolyLab</h2>
          <p className="text-muted text-sm m-0">Architecture and Methodology of the Polymer Discovery Platform</p>
        </div>

        <div className="card glass-card">
          <h3 className="mb-4">Architecture</h3>
          <table className="data-table">
            <tbody>
              <tr>
                <td style={{ width: '200px' }}>Generator</td>
                <td className="text-muted">Character-level GRU VAE (latent dim=32, hidden=512)</td>
              </tr>
              <tr>
                <td>Property Models</td>
                <td className="text-muted">XGBoost + Random Forest + Graph Attention Network (ensemble)</td>
              </tr>
              <tr>
                <td>Properties</td>
                <td className="text-muted">Biodegradability, Synthesizability, Toxicity, Recyclability, Strength, Tg, Solubility</td>
              </tr>
              <tr>
                <td>Green Score</td>
                <td className="text-muted">Weighted sum of 7 normalised properties [0,1]</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="card glass-card">
          <h3 className="mb-4">Datasets</h3>
          <table className="data-table">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Source</th>
                <th>Used for</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>PI1M</td>
                <td className="text-muted">RUIMINMA1996/PI1M</td>
                <td className="text-muted">VAE training corpus</td>
              </tr>
              <tr>
                <td>tsudalab</td>
                <td className="text-muted">tsudalab/Polymer-degradability</td>
                <td className="text-muted">Biodegradability</td>
              </tr>
              <tr>
                <td>Tox21 & ToxCast</td>
                <td className="text-muted">DeepChem</td>
                <td className="text-muted">Toxicity classification</td>
              </tr>
              <tr>
                <td>ESOL</td>
                <td className="text-muted">DeepChem</td>
                <td className="text-muted">Aqueous Solubility</td>
              </tr>
              <tr>
                <td>polyVERSE</td>
                <td className="text-muted">Ramprasad-Group/polyVERSE</td>
                <td className="text-muted">Mechanical strength, Tg, Recyclability proxy</td>
              </tr>
            </tbody>
          </table>
        </div>

        <div className="card glass-card">
          <h3 className="mb-3">Technical Stack</h3>
          <div className="flex gap-2 flex-wrap">
            <span className="badge badge-info p-2 px-3">FastAPI</span>
            <span className="badge badge-info p-2 px-3">React / Vite</span>
            <span className="badge badge-success p-2 px-3">PyTorch</span>
            <span className="badge badge-success p-2 px-3">XGBoost</span>
            <span className="badge badge-success p-2 px-3">scikit-learn</span>
            <span className="badge badge-success p-2 px-3">RDKit</span>
            <span className="badge badge-success p-2 px-3">PyG (Graph Neural Networks)</span>
          </div>
        </div>

      </div>
    </div>
  );
}
