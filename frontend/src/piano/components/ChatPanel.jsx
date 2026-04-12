import { useState, useRef, useEffect } from 'react';
import { API_BASE } from '../../lib/api.js';

const WELCOME_MESSAGE = {
  role: 'tutor',
  text: "Hi! I'm your AI piano tutor. Once you load a MIDI piece, I can analyse it and help you practise. Ask me anything — fingering tips, tricky passages, theory questions, you name it.",
};

export default function ChatPanel({ isOpen, onToggle, midiUrl = null, notes = [] }) {
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const bottomRef = useRef(null);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    if (isOpen) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen]);

  // When a MIDI is loaded, inject a context message
  useEffect(() => {
    if (midiUrl && notes.length > 0) {
      const noteCount = notes.length;
      const duration = notes.reduce((max, n) => Math.max(max, n.time + n.duration), 0);
      const mins = Math.floor(duration / 60);
      const secs = Math.round(duration % 60);
      const contextMsg = {
        role: 'tutor',
        text: `I can see your piece has been loaded — ${noteCount} notes, ${mins}m ${secs}s long. Play it and I'll give you feedback, or ask me anything about the piece.`,
      };
      setMessages([WELCOME_MESSAGE, contextMsg]);
    }
  }, [midiUrl]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isThinking) return;

    const userMsg = { role: 'user', text };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsThinking(true);

    try {
      // Build context about the loaded piece for the AI
      const midiContext = midiUrl
        ? `The user has loaded a MIDI file. It has ${notes.length} notes and lasts ${Math.round(notes.reduce((m, n) => Math.max(m, n.time + n.duration), 0))} seconds.`
        : 'No MIDI is currently loaded.';

      const conversationHistory = messages.map(m => ({
        role: m.role === 'tutor' ? 'assistant' : 'user',
        content: m.text,
      }));

      const response = await fetch(`${API_BASE}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          context: midiContext,
          history: conversationHistory,
        }),
      });

      if (!response.ok) throw new Error('Chat request failed');
      const data = await response.json();
      setMessages(prev => [...prev, { role: 'tutor', text: data.reply || data.message || 'I had trouble responding. Please try again.' }]);
    } catch {
      // Fallback: simple local heuristics when backend chat isn't available
      const fallbackReply = getFallbackReply(text);
      setMessages(prev => [...prev, { role: 'tutor', text: fallbackReply }]);
    } finally {
      setIsThinking(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <>
      {/* Chat toggle button - always visible */}
      <button
        className="chat-toggle-btn"
        onClick={onToggle}
        title={isOpen ? 'Close chat' : 'Open AI Tutor'}
      >
        <svg width="22" height="22" viewBox="0 0 24 24" fill="currentColor">
          <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z" />
        </svg>
        {!isOpen && <span className="chat-badge">AI</span>}
      </button>

      {/* Sliding chat panel */}
      <div className={`chat-panel ${isOpen ? 'open' : ''}`}>
        <div className="chat-panel-content">
          <div className="chat-header">
            <div className="chat-header-info">
              <div className="chat-avatar">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
                </svg>
              </div>
              <div>
                <h4 className="chat-title">AI Piano Tutor</h4>
                <span className="chat-status">{midiUrl ? '● Piece loaded' : '○ No piece loaded'}</span>
              </div>
            </div>
          </div>

          <div className="chat-messages">
            {messages.map((msg, i) => (
              <div key={i} className={`chat-msg ${msg.role}`}>
                {msg.role === 'tutor' && (
                  <div className="chat-msg-avatar">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                      <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
                    </svg>
                  </div>
                )}
                <div className="chat-bubble">
                  <p>{msg.text}</p>
                </div>
              </div>
            ))}
            {isThinking && (
              <div className="chat-msg tutor">
                <div className="chat-msg-avatar">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
                  </svg>
                </div>
                <div className="chat-bubble chat-thinking">
                  <span className="dot" /><span className="dot" /><span className="dot" />
                </div>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="chat-input-area">
            <input
              type="text"
              className="chat-input"
              placeholder="Ask your tutor..."
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isThinking}
            />
            <button
              className="chat-send-btn"
              onClick={sendMessage}
              disabled={!input.trim() || isThinking}
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

// Simple local fallback replies when the backend /chat endpoint isn't set up yet
function getFallbackReply(text) {
  const t = text.toLowerCase();
  if (t.includes('finger') || t.includes('hand')) return "Fingering depends on the passage. Generally, keep your wrist relaxed and let your thumb cross naturally under longer phrases. For fast runs, practice hands separately first.";
  if (t.includes('slow') || t.includes('speed')) return "Practising slowly is one of the best techniques. Try 50% speed and focus on clean articulation before building up tempo. The speed control at the top of the player will help.";
  if (t.includes('mistake') || t.includes('miss') || t.includes('error')) return "Isolate the exact measure where you're making the mistake. Loop just those 2–4 bars at a slower tempo until muscle memory kicks in, then gradually increase speed.";
  if (t.includes('left hand') || t.includes('right hand')) return "Working hands separately is always a good idea for tricky passages. Master each hand independently, then combine them at a slow tempo.";
  if (t.includes('chord') || t.includes('harmony')) return "When practising chords, make sure all notes land simultaneously. If some notes are uneven, practice 'blocking' the chord (holding it rather than playing) to feel the hand shape.";
  if (t.includes('rhythm') || t.includes('timing')) return "A metronome is your best friend for timing. Start very slow, nail the rhythm, then increase BPM by 5–10 each session.";
  return "Great question! For the best advice, try practising the passage at 50–70% speed with a metronome. Breaking it into small sections and working hands separately usually helps a lot. Let me know if you want more specific guidance.";
}
