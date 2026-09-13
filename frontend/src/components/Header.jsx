// =============================================================
// components/Header.jsx — Top header with theme toggle
// =============================================================

import { toggleTheme } from '../api';
import './Header.css';

export default function Header({ theme, onThemeChange }) {
  const handleToggleTheme = async () => {
    try {
      const res = await toggleTheme();
      onThemeChange(res.theme);
    } catch (e) {
      console.error('Theme toggle failed', e);
      // Optimistic update
      onThemeChange(theme === 'dark' ? 'light' : 'dark');
    }
  };

  return (
    <header className="header">
      <div className="header-left">
        <h1 className="header-title">Generative AI for Sustainable Polymer Discovery</h1>
        <div className="header-subtitle">
          VAE generation · XGBoost + RF + GNN Ensemble · 7-Property Green Score
        </div>
      </div>
      
    </header>
  );
}
