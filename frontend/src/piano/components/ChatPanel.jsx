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

  useEffect(() => {
    if (isOpen) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen]);

  useEffect(() => {
    if (midiUrl && notes.length > 0) {
      const noteCount = notes.length;
      const pieceDuration = notes.reduce((max, note) => Math.max(max, note.time + note.duration), 0);
      const mins = Math.floor(pieceDuration / 60);
      const secs = Math.round(pieceDuration % 60);

      setMessages([
        WELCOME_MESSAGE,
        {
          role: 'tutor',
          text: `I can see your piece has been loaded — ${noteCount} notes, ${mins}m ${secs}s long. Play it and I'll give you feedback, or ask me anything about the piece.`,
        },
      ]);
    }
  }, [midiUrl, notes]);

  const sendMessage = async () => {
    const text = input.trim();
    if (!text || isThinking) {
      return;
    }

    setMessages((prev) => [...prev, { role: 'user', text }]);
    setInput('');
    setIsThinking(true);

    try {
      const midiContext = midiUrl
        ? `The user has loaded a MIDI file. It has ${notes.length} notes and lasts ${Math.round(notes.reduce((max, note) => Math.max(max, note.time + note.duration), 0))} seconds.`
        : 'No MIDI is currently loaded.';

      const conversationHistory = messages.map((message) => ({
        role: message.role === 'tutor' ? 'assistant' : 'user',
        content: message.text,
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

      if (!response.ok) {
        throw new Error('Chat request failed');
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: 'tutor', text: data.reply || data.message || 'I had trouble responding. Please try again.' },
      ]);
    } catch {
      setMessages((prev) => [...prev, { role: 'tutor', text: getFallbackReply(text) }]);
    } finally {
      setIsThinking(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className={`chat-panel ${isOpen ? 'open' : ''}`}>
      <div className="chat-panel-content">
        <div className="chat-header">
          <div className="chat-header-info">
            <div className="chat-avatar">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
              </svg>
            </div>
            <div>
              <h4 className="chat-title">AI Piano Tutor</h4>
              <span className="chat-status">{midiUrl ? 'Piece loaded' : 'No piece loaded'}</span>
            </div>
          </div>

          <button
            className="panel-close-btn"
            onClick={onToggle}
            title="Close tutor"
            type="button"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" />
            </svg>
          </button>
        </div>

        <div className="chat-messages">
          {messages.map((message, index) => (
            <div key={index} className={`chat-msg ${message.role}`}>
              {message.role === 'tutor' && (
                <div className="chat-msg-avatar">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
                  </svg>
                </div>
              )}
              <div className="chat-bubble">
                <p>{message.text}</p>
              </div>
            </div>
          ))}

          {isThinking && (
            <div className="chat-msg tutor">
              <div className="chat-msg-avatar">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-2 15l-5-5 1.41-1.41L10 14.17l7.59-7.59L19 8l-9 9z" />
                </svg>
              </div>
              <div className="chat-bubble chat-thinking">
                <span className="dot" />
                <span className="dot" />
                <span className="dot" />
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
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={handleKeyDown}
            disabled={isThinking}
          />
          <button
            className="chat-send-btn"
            onClick={sendMessage}
            disabled={!input.trim() || isThinking}
            type="button"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}

function getFallbackReply(text) {
  const normalized = text.toLowerCase();

  if (normalized.includes('finger') || normalized.includes('hand')) {
    return "Fingering depends on the passage. Generally, keep your wrist relaxed and let your thumb cross naturally under longer phrases. For fast runs, practice hands separately first.";
  }

  if (normalized.includes('slow') || normalized.includes('speed')) {
    return "Practising slowly is one of the best techniques. Try 50% speed and focus on clean articulation before building up tempo. The speed control at the top of the player will help.";
  }

  if (normalized.includes('mistake') || normalized.includes('miss') || normalized.includes('error')) {
    return "Isolate the exact measure where you're making the mistake. Loop just those 2–4 bars at a slower tempo until muscle memory kicks in, then gradually increase speed.";
  }

  if (normalized.includes('left hand') || normalized.includes('right hand')) {
    return "Working hands separately is always a good idea for tricky passages. Master each hand independently, then combine them at a slow tempo.";
  }

  if (normalized.includes('chord') || normalized.includes('harmony')) {
    return "When practising chords, make sure all notes land simultaneously. If some notes are uneven, practice 'blocking' the chord to feel the hand shape before playing it in rhythm.";
  }

  if (normalized.includes('rhythm') || normalized.includes('timing')) {
    return "A metronome is your best friend for timing. Start very slow, nail the rhythm, then increase BPM by 5–10 each session.";
  }

  return "Great question. Try practicing the passage at 50–70% speed with a metronome, and break it into small sections before bringing the hands back together.";
}
