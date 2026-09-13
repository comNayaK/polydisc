// =============================================================
// components/ChartGrid.jsx — Tabbed chart viewer
// =============================================================

import { useState } from 'react';

const TABS = [
  { id: 'summary', label: 'Summary Charts' },
  { id: 'analysis', label: 'Detailed Analysis' },
  { id: 'distribution', label: 'Score Distribution' },
];

export default function ChartGrid({ charts }) {
  const [activeTab, setActiveTab] = useState('summary');

  if (!charts || Object.keys(charts).length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-muted" style={{ minHeight: '300px', background: 'var(--bg-panel)', borderRadius: 'var(--radius-lg)' }}>
        Run discovery to view results and visualisations
      </div>
    );
  }

  const renderImage = (key, alt, height) => {
    if (!charts[key]) return null;
    return (
      <div className="card p-2 animate-fade-in" style={{ 
        height: height || 'auto', 
        overflow: 'hidden',
        background: '#fff',
        border: '1px solid var(--border-color)',
        boxShadow: 'var(--shadow-sm)'
      }}>
        <img 
          src={`data:image/png;base64,${charts[key]}`} 
          alt={alt} 
          style={{ width: '100%', height: '100%', objectFit: 'contain' }} 
        />
      </div>
    );
  };

  return (
    <div className="chart-grid-container">
      {/* Tabs */}
      <div className="flex gap-4 mb-6 border-b" style={{ borderColor: 'var(--border-color)' }}>
        {TABS.map(t => (
          <button
            key={t.id}
            className={`pb-3 text-sm font-semibold transition-all ${activeTab === t.id ? 'text-accent border-b-2 border-accent' : 'text-muted hover:text-secondary'}`}
            style={{ background: 'none', borderTop: 'none', borderLeft: 'none', borderRight: 'none', cursor: 'pointer' }}
            onClick={() => setActiveTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </div>

      {/* Content */}
      <div className="tab-content" style={{ minHeight: '500px' }}>
        {activeTab === 'summary' && (
          <div className="flex-col gap-6" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
              {renderImage('bar', 'Green Score Ranking', '320px')}
              {renderImage('radar', 'Property Radar', '320px')}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.2fr', gap: '24px' }}>
              {renderImage('heatmap', 'Property Heatmap', '350px')}
              {renderImage('breakdown', 'Component Breakdown', '350px')}
            </div>
          </div>
        )}

        {activeTab === 'analysis' && (
          <div className="flex-col gap-6" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr', gap: '24px' }}>
              {renderImage('corr', 'Correlation Matrix', '400px')}
              {renderImage('struct', 'Chemical Structures', '400px')}
            </div>
            {renderImage('umap', 'UMAP Chemical Space Projection', '450px')}
          </div>
        )}

        {activeTab === 'distribution' && (
          <div className="flex-col gap-6" style={{ display: 'flex', flexDirection: 'column', gap: '24px' }}>
            {renderImage('dist', 'Green Score Distribution', '400px')}
            {renderImage('violin', 'Property Variance (Violin Plots)', '350px')}
          </div>
        )}
      </div>
    </div>
  );
}
