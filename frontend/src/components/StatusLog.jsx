// =============================================================
// components/StatusLog.jsx — Displays backend execution log
// =============================================================

import { useEffect, useRef } from 'react';

export default function StatusLog({ log, loading }) {
  const logEndRef = useRef(null);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [log]);

  return (
    <div className="card flex-col" style={{ height: '220px', padding: '16px' }}>
      <div className="flex justify-between items-center mb-2">
        <h3 className="text-sm uppercase tracking-wide text-secondary m-0" style={{ letterSpacing: '0.05em' }}>
          Status Log
        </h3>
        {loading && <div className="spinner"></div>}
      </div>
      
      <div 
        className="bg-input rounded flex-1 p-3 overflow-auto mono text-sm" 
        style={{ 
          background: 'var(--bg-input)', 
          borderRadius: 'var(--radius-sm)', 
          border: '1px solid var(--border-color)',
          fontSize: '0.8rem',
          color: 'var(--text-muted)'
        }}
      >
        {log && log.length > 0 ? (
          log.map((line, i) => (
            <div key={i} className="mb-1">
              {line.includes('error') ? (
                <span className="text-danger" style={{ color: '#e74c3c' }}>{line.replace('❌ ', '')}</span>
              ) : line.includes('✓') ? (
                <span className="text-accent">{line.replace(' ✓', '')}</span>
              ) : (
                line
              )}
            </div>
          ))
        ) : (
          <div className="opacity-50">Ready to start discovery...</div>
        )}
        <div ref={logEndRef} />
      </div>
    </div>
  );
}
