const SAMPLE_MESSAGES = [
  {
    role: 'tutor',
    text: "Welcome! I can see you're working on this piece. Let me help you with the tricky parts.",
  },
  {
    role: 'user',
    text: "I keep missing the F# in measure 12.",
  },
  {
    role: 'tutor',
    text: "That's a common spot! Try isolating measures 11-13 at 50% speed. Focus on the finger crossing from thumb to index finger.",
  },
  {
    role: 'user',
    text: "Thanks! Should I practice hands separately first?",
  },
  {
    role: 'tutor',
    text: "Absolutely. Start with the right hand alone at 50% speed, then add the left hand once you're comfortable. I noticed your timing drifts in the left hand around beat 3 \u2014 a metronome at 60 BPM would help lock that in.",
  },
  {
    role: 'user',
    text: "Got it, I'll try that now.",
  },
  {
    role: 'tutor',
    text: "Great! I'll be here watching. Hit play whenever you're ready and I'll give you real-time feedback on your accuracy and timing.",
  },
];

export default function ChatPanel({ isOpen, onToggle }) {
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
                <span className="chat-status">Online</span>
              </div>
            </div>
          </div>

          <div className="chat-messages">
            {SAMPLE_MESSAGES.map((msg, i) => (
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
          </div>

          <div className="chat-input-area">
            <input
              type="text"
              className="chat-input"
              placeholder="Ask your tutor..."
              disabled
            />
            <button className="chat-send-btn" disabled>
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
