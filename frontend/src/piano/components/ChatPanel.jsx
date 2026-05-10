import { useState, useRef, useEffect } from 'react';
import { API_BASE } from '../../shared/api.js';

const CHAT_PANEL_WIDTH_STORAGE_KEY = 'pianoTutorChatPanelWidth';
const DEFAULT_CHAT_PANEL_WIDTH = 340;
const MIN_CHAT_PANEL_WIDTH = 300;
const MAX_CHAT_PANEL_WIDTH = 760;

const WELCOME_MESSAGE = {
  role: 'tutor',
  text: "Hi! I'm your AI piano tutor. Once you load a MIDI piece, I can analyse it and help you practise. Ask me anything - fingering tips, tricky passages, theory questions, you name it.",
};

function clampChatPanelWidth(width) {
  const viewportMax = typeof window === 'undefined'
    ? MAX_CHAT_PANEL_WIDTH
    : Math.max(MIN_CHAT_PANEL_WIDTH, Math.floor(window.innerWidth * 0.82));
  return Math.min(MAX_CHAT_PANEL_WIDTH, viewportMax, Math.max(MIN_CHAT_PANEL_WIDTH, Math.round(width)));
}

function getInitialChatPanelWidth() {
  if (typeof window === 'undefined') {
    return DEFAULT_CHAT_PANEL_WIDTH;
  }

  const storedWidth = Number(window.localStorage.getItem(CHAT_PANEL_WIDTH_STORAGE_KEY));
  return Number.isFinite(storedWidth)
    ? clampChatPanelWidth(storedWidth)
    : DEFAULT_CHAT_PANEL_WIDTH;
}

function saveChatPanelWidth(width) {
  if (typeof window !== 'undefined') {
    window.localStorage.setItem(CHAT_PANEL_WIDTH_STORAGE_KEY, String(width));
  }
}

export default function ChatPanel({
  isOpen,
  onToggle,
  midiUrl = null,
  notes = [],
  analysisData = null,
  analysisLoading = false,
  preparedTutor = null,
  projectName = null,
}) {
  const [messages, setMessages] = useState([WELCOME_MESSAGE]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const [panelWidth, setPanelWidth] = useState(getInitialChatPanelWidth);
  const bottomRef = useRef(null);

  const summaryCards = preparedTutor?.summaryCards || null;
  const suggestedQuestions = Array.isArray(preparedTutor?.suggestedQuestions)
    ? preparedTutor.suggestedQuestions
    : [];
  const preparedSessionId = preparedTutor?.sessionId || null;

  useEffect(() => {
    if (isOpen) {
      bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isOpen]);

  useEffect(() => {
    if (summaryCards || preparedTutor?.openingMessage || preparedSessionId) {
      setMessages([
        {
          role: 'tutor',
          text: preparedTutor?.openingMessage?.trim()
            || 'Your tutor session is ready. Ask about this performance or use one of the starter prompts below.',
        },
      ]);
      return;
    }

    if (!midiUrl || notes.length === 0) {
      setMessages([WELCOME_MESSAGE]);
      return;
    }

    const noteCount = notes.length;
    const pieceDuration = notes.reduce((max, note) => Math.max(max, note.time + note.duration), 0);
    const mins = Math.floor(pieceDuration / 60);
    const secs = Math.round(pieceDuration % 60);
    const intro = `I can see your piece has been loaded - ${noteCount} notes, ${mins}m ${secs}s long.`;

    let analysisMessage = "Play it and I'll give you feedback, or ask me anything about the piece.";
    if (analysisLoading) {
      analysisMessage = "I'm also analyzing the MIDI now so I can give you more specific practice feedback.";
    } else if (analysisData?.analysis_overview) {
      analysisMessage = `${analysisData.analysis_overview} Ask me about timing, phrasing, difficulty, or how to practise it.`;
    }

    setMessages([
      WELCOME_MESSAGE,
      {
        role: 'tutor',
        text: `${intro} ${analysisMessage}`,
      },
    ]);
  }, [midiUrl, notes, analysisData, analysisLoading, preparedSessionId, preparedTutor?.openingMessage, summaryCards]);

  const sendMessage = async (prefilledText = null) => {
    const text = (prefilledText ?? input).trim();
    if (!text || isThinking) {
      return;
    }

    setMessages((prev) => [...prev, { role: 'user', text }]);
    setInput('');
    setIsThinking(true);

    try {
      if (preparedSessionId) {
        const response = await fetch(`${API_BASE}/tutor/message`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            session_id: preparedSessionId,
            message: text,
          }),
        });

        if (!response.ok) {
          throw new Error('Tutor request failed');
        }

        const data = await response.json();
        setMessages((prev) => [
          ...prev,
          { role: 'tutor', text: data.reply || 'I had trouble responding. Please try again.' },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: 'tutor', text: getFallbackReply(text, analysisData, summaryCards) },
        ]);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: 'tutor', text: getFallbackReply(text, analysisData, summaryCards) },
      ]);
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

  const setAndSavePanelWidth = (width) => {
    const nextWidth = clampChatPanelWidth(width);
    setPanelWidth(nextWidth);
    saveChatPanelWidth(nextWidth);
  };

  const handleResizePointerDown = (event) => {
    if (!isOpen) {
      return;
    }

    event.preventDefault();
    const startX = event.clientX;
    const startWidth = panelWidth;
    let latestWidth = startWidth;

    const handlePointerMove = (moveEvent) => {
      latestWidth = clampChatPanelWidth(startWidth + startX - moveEvent.clientX);
      setPanelWidth(latestWidth);
    };

    const endResize = () => {
      document.body.classList.remove('chat-panel-resizing');
      window.removeEventListener('pointermove', handlePointerMove);
      window.removeEventListener('pointerup', endResize);
      window.removeEventListener('pointercancel', endResize);
      saveChatPanelWidth(latestWidth);
    };

    document.body.classList.add('chat-panel-resizing');
    window.addEventListener('pointermove', handlePointerMove);
    window.addEventListener('pointerup', endResize);
    window.addEventListener('pointercancel', endResize);
  };

  const handleResizeKeyDown = (event) => {
    if (!isOpen) {
      return;
    }

    if (event.key === 'ArrowLeft') {
      event.preventDefault();
      setAndSavePanelWidth(panelWidth + 32);
    } else if (event.key === 'ArrowRight') {
      event.preventDefault();
      setAndSavePanelWidth(panelWidth - 32);
    } else if (event.key === 'Home') {
      event.preventDefault();
      setAndSavePanelWidth(MIN_CHAT_PANEL_WIDTH);
    } else if (event.key === 'End') {
      event.preventDefault();
      setAndSavePanelWidth(MAX_CHAT_PANEL_WIDTH);
    }
  };

  const baseChatStatus = !midiUrl
    ? 'No piece loaded'
    : preparedTutor?.mode === 'compare'
      ? 'Comparison ready'
      : preparedSessionId
        ? 'Tutor ready'
        : analysisLoading
          ? 'Analyzing MIDI'
          : analysisData
            ? 'MIDI analyzed'
            : 'Piece loaded';
  const chatStatus = projectName && midiUrl
    ? `${baseChatStatus} · ${projectName}`
    : baseChatStatus;

  return (
    <div
      className={`chat-panel ${isOpen ? 'open' : ''}`}
      style={{ '--chat-panel-width': `${panelWidth}px` }}
    >
      <div
        className="chat-resize-handle"
        onPointerDown={handleResizePointerDown}
        onKeyDown={handleResizeKeyDown}
        role="separator"
        aria-label="Resize tutor panel"
        aria-orientation="vertical"
        aria-valuemin={MIN_CHAT_PANEL_WIDTH}
        aria-valuemax={MAX_CHAT_PANEL_WIDTH}
        aria-valuenow={panelWidth}
        tabIndex={isOpen ? 0 : -1}
        title="Drag to resize tutor"
      />
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
              <span className="chat-status">{chatStatus}</span>
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
          {summaryCards && (
            <div className="chat-summary">
              <div className="chat-summary-card">
                <span className="chat-summary-kicker">Overall Assessment</span>
                <h5>{summaryCards.overall_assessment?.headline || 'Tutor feedback is ready'}</h5>
                {summaryCards.overall_assessment?.summary && <p>{summaryCards.overall_assessment.summary}</p>}
                {summaryCards.overall_assessment?.stats?.length > 0 && (
                  <div className="chat-summary-stats">
                    {summaryCards.overall_assessment.stats.map((stat) => (
                      <div key={`${stat.label}-${stat.value}`} className="chat-summary-stat">
                        <span>{stat.label}</span>
                        <strong>{stat.value}</strong>
                      </div>
                    ))}
                  </div>
                )}
                {summaryCards.strengths?.length > 0 && (
                  <div className="chat-summary-tags">
                    {summaryCards.strengths.map((strength) => (
                      <span key={strength}>{strength}</span>
                    ))}
                  </div>
                )}
              </div>

              <div className="chat-summary-card">
                <span className="chat-summary-kicker">What To Fix First</span>
                <ul className="chat-summary-list">
                  {(summaryCards.immediate_focus || []).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>

              <div className="chat-summary-card">
                <span className="chat-summary-kicker">Practice Plan</span>
                <ul className="chat-summary-list">
                  {(summaryCards.practice_plan || []).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}

          {messages.map((message, index) => (
            <div key={`${message.role}-${index}`}>
              <div className={`chat-msg ${message.role}`}>
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

              {message.role === 'tutor' && index === 0 && suggestedQuestions.length > 0 && (
                <div className="chat-suggested-prompts">
                  {suggestedQuestions.map((question) => (
                    <button
                      key={question}
                      className="chat-suggested-btn"
                      onClick={() => sendMessage(question)}
                      disabled={isThinking}
                      type="button"
                    >
                      {question}
                    </button>
                  ))}
                </div>
              )}
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
            onClick={() => sendMessage()}
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

function getFallbackReply(text, analysisData, summaryCards = null) {
  const normalized = text.toLowerCase();
  const metrics = analysisData?.metrics || {};
  const recommendations = analysisData?.practice_recommendations || [];
  const dynamicRange = Math.round(metrics?.velocity_stats?.dynamic_range || 0);
  const noteDensity = Number(metrics?.notes_per_second || 0).toFixed(2);
  const avgDuration = Number(metrics?.duration_stats?.mean || 0).toFixed(2);
  const preparedFocus = summaryCards?.immediate_focus || [];
  const preparedPlan = summaryCards?.practice_plan || [];
  const preparedStrengths = summaryCards?.strengths || [];

  if (normalized.includes('practice first') || normalized.includes('start with')) {
    if (preparedFocus.length > 0) {
      return `Start with this first: ${preparedFocus[0]}. Once that feels more reliable, move to ${preparedFocus[1] || 'another short section at the same slow tempo'}.`;
    }
  }

  if (normalized.includes('15-minute') || normalized.includes('practice plan')) {
    if (preparedPlan.length > 0) {
      return `Try this short block: 5 minutes on ${preparedPlan[0].toLowerCase()}, 5 minutes on ${preparedPlan[1]?.toLowerCase() || 'your hardest short section'}, then finish with ${preparedPlan[2]?.toLowerCase() || 'one controlled full run-through'}.`;
    }
  }

  if (normalized.includes('finger') || normalized.includes('hand')) {
    return "Fingering depends on the passage. Generally, keep your wrist relaxed and let your thumb cross naturally under longer phrases. For fast runs, practice hands separately first.";
  }

  if (normalized.includes('slow') || normalized.includes('speed')) {
    if (metrics?.notes_per_second) {
      return `This MIDI sits around ${noteDensity} notes per second, so start below performance speed and only increase tempo when the passage stays even. A good target is 50-70% speed with clean attacks before building back up.`;
    }
    return "Practising slowly is one of the best techniques. Try 50% speed and focus on clean articulation before building up tempo. The speed control at the top of the player will help.";
  }

  if (normalized.includes('mistake') || normalized.includes('miss') || normalized.includes('error')) {
    if (preparedFocus.length > 0) {
      return `The clearest first correction is this: ${preparedFocus[0]}. Isolate the exact bar that breaks down, loop 2-4 bars slowly, and rebuild the passage with a steady pulse.`;
    }
    if (recommendations.length > 0) {
      return `The analysis already points us toward this: ${recommendations[0]}. Isolate the exact bar that breaks down, loop 2-4 bars slowly, and rebuild the passage with a steady pulse.`;
    }
    return "Isolate the exact measure where you're making the mistake. Loop just those 2-4 bars at a slower tempo until muscle memory kicks in, then gradually increase speed.";
  }

  if (normalized.includes('left hand') || normalized.includes('right hand')) {
    return "Working hands separately is always a good idea for tricky passages. Master each hand independently, then combine them at a slow tempo.";
  }

  if (normalized.includes('chord') || normalized.includes('harmony')) {
    return "When practising chords, make sure all notes land simultaneously. If some notes are uneven, practice 'blocking' the chord to feel the hand shape before playing it in rhythm.";
  }

  if (normalized.includes('rhythm') || normalized.includes('timing')) {
    return "A metronome is your best friend for timing. Start very slow, nail the rhythm, then increase BPM by 5-10 each session.";
  }

  if ((normalized.includes('strength') || normalized.includes('good')) && preparedStrengths.length > 0) {
    return `One thing worth keeping is this: ${preparedStrengths[0]}. Keep that intact while you tighten the weaker spots one passage at a time.`;
  }

  if (summaryCards?.overall_assessment?.summary) {
    return `${summaryCards.overall_assessment.summary} ${preparedFocus[0] ? `The next practical step is ${preparedFocus[0]}.` : 'Ask me for a shorter drill if you want a more focused practice step.'}`;
  }

  if (analysisData?.analysis_overview) {
    return `From the MIDI analysis, I'm seeing ${dynamicRange} dynamic-range points and an average note length of ${avgDuration}s. ${recommendations[0] || 'Try practicing in short sections with a metronome and listen for even tone and timing.'}`;
  }

  return "Great question. Try practicing the passage at 50-70% speed with a metronome, and break it into small sections before bringing the hands back together.";
}
