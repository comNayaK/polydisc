// =============================================================
// pages/AIAssistant.jsx — Groq LLaMA chat interface
// =============================================================

import { useState, useRef, useEffect } from 'react';
import { sendChatMessage } from '../api';

export default function AIAssistant() {
  const [messages, setMessages] = useState([
    { role: 'assistant', content: 'Hello! I am PolyLab AI, powered by Groq LLaMA 3.3. Describe your polymer application, and I can suggest generation configurations and interpret property scores.' }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;
    
    const userMsg = { role: 'user', content: input };
    const newMessages = [...messages, userMsg];
    setMessages(newMessages);
    setInput('');
    setLoading(true);

    try {
      const res = await sendChatMessage(newMessages);
      setMessages([...newMessages, { role: 'assistant', content: res.reply, config: res.config }]);
    } catch (e) {
      setMessages([...newMessages, { role: 'assistant', content: `Error connecting to AI: ${e.message}` }]);
    } finally {
      setLoading(false);
    }
  };

  const handleApplyConfig = (config) => {
    // In a full implementation, this would use a React Context or similar state manager
    // to apply these settings to the Dashboard page.
    alert(`Configuration applied!\n\nWeights: ${JSON.stringify(config, null, 2)}\n\nNavigate to the Dashboard to run discovery.`);
  };

  return (
    <div className="p-6 animate-fade-in flex-col h-full" style={{ padding: '24px', display: 'flex', flexDirection: 'column', height: 'calc(100vh - 64px)' }}>
      
      <div className="mb-4">
        <h2 className="text-accent mb-2">PolyLab AI Assistant</h2>
        <p className="text-muted text-sm m-0">Powered by Groq LLaMA 3.3 70B</p>
      </div>

      <div className="card glass-card flex-1 flex-col relative" style={{ overflow: 'hidden', padding: 0 }}>
        
        {/* Chat history */}
        <div className="flex-1 overflow-auto p-6 flex-col gap-4" style={{ display: 'flex', flexDirection: 'column' }}>
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div 
                className={`p-4 rounded-lg max-w-[80%] ${m.role === 'user' ? 'bg-accent text-white' : 'bg-panel border'}`}
                style={{ 
                  background: m.role === 'user' ? 'var(--accent-dim)' : 'var(--bg-panel)',
                  borderColor: m.role === 'user' ? 'transparent' : 'var(--border-color)',
                  borderRadius: 'var(--radius-lg)',
                  borderBottomRightRadius: m.role === 'user' ? '4px' : 'var(--radius-lg)',
                  borderBottomLeftRadius: m.role !== 'user' ? '4px' : 'var(--radius-lg)',
                  maxWidth: '80%',
                  lineHeight: '1.5'
                }}
              >
                <div style={{ whiteSpace: 'pre-wrap', fontSize: '0.9rem' }}>{m.content}</div>
                
                {m.config && (
                  <div className="mt-3 pt-3 border-t" style={{ borderColor: 'var(--border-color)' }}>
                    <div className="text-xs text-muted mb-2 font-bold uppercase">Suggested Configuration</div>
                    <button 
                      className="btn btn-primary btn-sm w-full"
                      onClick={() => handleApplyConfig(m.config)}
                    >
                      Apply to Dashboard
                    </button>
                  </div>
                )}
              </div>
            </div>
          ))}
          {loading && (
            <div className="flex justify-start">
              <div className="p-4 rounded-lg bg-panel border flex items-center gap-2" style={{ background: 'var(--bg-panel)', borderColor: 'var(--border-color)', borderRadius: 'var(--radius-lg)' }}>
                <div className="spinner" style={{width:'16px',height:'16px'}}></div> <span className="text-sm text-muted">Thinking...</span>
              </div>
            </div>
          )}
          <div ref={endRef} />
        </div>

        {/* Input area */}
        <div className="p-4 border-t bg-card" style={{ borderColor: 'var(--border-color)', background: 'var(--bg-card)' }}>
          <div className="flex gap-2">
            <input 
              className="input flex-1"
              placeholder="E.g., I need a highly biodegradable polymer for packaging..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handleSend()}
              disabled={loading}
              style={{ borderRadius: 'var(--radius-xl)', paddingLeft: '20px' }}
            />
            <button 
              className="btn btn-primary"
              onClick={handleSend}
              disabled={loading || !input.trim()}
              style={{ borderRadius: 'var(--radius-xl)', padding: '10px 24px' }}
            >
              Send
            </button>
          </div>
        </div>

      </div>

    </div>
  );
}
