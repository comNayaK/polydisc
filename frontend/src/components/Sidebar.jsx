// =============================================================
// components/Sidebar.jsx — Persistent vertical navigation
// =============================================================

import { useState } from 'react';
import './Sidebar.css';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'analysis', label: 'Analysis' },
  { id: 'assistant', label: 'AI Assistant' },
  { id: 'history', label: 'History' },
  { id: 'about', label: 'About' },
];

export default function Sidebar({ activePage, onNavigate }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <aside className={`sidebar ${collapsed ? 'collapsed' : ''}`}>
      {/* Logo */}
      <div className="sidebar-logo">
        {!collapsed && (
          <div className="logo-text">
            <span className="logo-title">PolyLab</span>
            <span className="logo-sub">AI Discovery</span>
          </div>
        )}
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <div className="nav-label">{!collapsed && 'Navigation'}</div>
        {NAV_ITEMS.map((item) => (
          <button
            key={item.id}
            className={`nav-item ${activePage === item.id ? 'active' : ''}`}
            onClick={() => onNavigate(item.id)}
            title={collapsed ? item.label : ''}
          >
            {!collapsed && <span className="nav-text">{item.label}</span>}
            {collapsed && <span className="nav-text-collapsed">{item.label.substring(0, 1)}</span>}
            {activePage === item.id && <div className="nav-indicator" />}
          </button>
        ))}
      </nav>

      {/* Footer */}
      <div className="sidebar-footer">
        <button
          className="collapse-btn"
          onClick={() => setCollapsed(!collapsed)}
          title={collapsed ? 'Expand' : 'Collapse'}
        >
          {collapsed ? '→' : '←'}
        </button>
        {!collapsed && (
          <div className="sidebar-version">
            <span className="text-muted">v2.0 · React</span>
          </div>
        )}
      </div>
    </aside>
  );
}
