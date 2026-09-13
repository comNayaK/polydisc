// =============================================================
// App.jsx — Main application layout and router
// =============================================================

import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import Header from './components/Header';
import Dashboard from './pages/Dashboard';
import Analysis from './pages/Analysis';
import History from './pages/History';
import AIAssistant from './pages/AIAssistant';
import About from './pages/About';
import './App.css';

function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const theme = 'light'; // Forced light theme

  const renderPage = () => {
    switch (activePage) {
      case 'dashboard': return <Dashboard />;
      case 'analysis': return <Analysis />;
      case 'assistant': return <AIAssistant />;
      case 'history': return <History />;
      case 'about': return <About />;
      default: return <Dashboard />;
    }
  };

  return (
    <div className="app-container">
      <Sidebar activePage={activePage} onNavigate={setActivePage} />
      
      <main className="main-content">
        <Header theme={theme} onThemeChange={() => {}} />
        
        <div className="page-wrapper">
          {renderPage()}
        </div>
      </main>
    </div>
  );
}

export default App;
