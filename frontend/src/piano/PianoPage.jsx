import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import './PianoPage.css';
import TopControls from './components/TopControls.jsx';
import FallingNotesCanvas from './components/FallingNotesCanvas.jsx';
import PianoKeyboard from './components/PianoKeyboard.jsx';
import LeftMenu from './components/LeftMenu.jsx';
import ChatPanel from './components/ChatPanel.jsx';
import RecordButton from './components/RecordButton.jsx';
import { useMidi } from './hooks/useMidi.js';
import { usePianoPlayer } from './hooks/usePianoPlayer.js';
import { useRecorder } from './hooks/useRecorder.js';

export default function PianoPage() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const mainRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 1200, height: 500 });

  // Load MIDI
  const { notes, duration, tempo, isLoading, error } = useMidi();

  // Player
  const player = usePianoPlayer(notes, duration, tempo);

  // Recorder
  const recorder = useRecorder();

  // Track container dimensions
  useEffect(() => {
    const el = mainRef.current;
    if (!el) return;

    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({ width, height: Math.max(height - 140, 200) }); // subtract keyboard height
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Compute active notes (currently sounding)
  const activeNotes = useMemo(() => {
    const t = player.currentTime;
    return notes.filter(n => t >= n.time && t < n.time + n.duration);
  }, [notes, player.currentTime]);

  // Keyboard shortcuts
  useEffect(() => {
    function handleKey(e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;

      switch (e.code) {
        case 'Space':
          e.preventDefault();
          player.isPlaying ? player.pause() : player.play();
          break;
        case 'ArrowLeft':
          e.preventDefault();
          player.seek(Math.max(0, player.currentTime - 5));
          break;
        case 'ArrowRight':
          e.preventDefault();
          player.seek(Math.min(player.duration, player.currentTime + 5));
          break;
        case 'Escape':
          setMenuOpen(false);
          setChatOpen(false);
          break;
      }
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [player]);

  return (
    <div className="piano-page">
      {/* Left Menu */}
      <LeftMenu isOpen={menuOpen} onToggle={() => setMenuOpen(v => !v)} />

      {/* Chat Panel */}
      <ChatPanel isOpen={chatOpen} onToggle={() => setChatOpen(v => !v)} />

      {/* Top Controls */}
      <div className="pp-top">
        <TopControls
          isPlaying={player.isPlaying}
          currentTime={player.currentTime}
          duration={player.duration || duration}
          speed={player.speed}
          volume={player.volume}
          isLoaded={player.isLoaded}
          onPlay={player.play}
          onPause={player.pause}
          onStop={player.stop}
          onSeek={player.seek}
          onSpeedChange={player.setSpeed}
          onVolumeChange={player.setVolume}
        />
        <RecordButton
          isRecording={recorder.isRecording}
          onStart={recorder.startRecording}
          onStop={recorder.stopRecording}
          audioURL={recorder.audioURL}
          error={recorder.error}
        />
      </div>

      {/* Main content area */}
      <div className="pp-main" ref={mainRef}>
        {isLoading && (
          <div className="pp-overlay">
            <div className="pp-loader">
              <div className="spinner" />
              <p>Loading MIDI file...</p>
            </div>
          </div>
        )}

        {error && (
          <div className="pp-overlay">
            <div className="pp-error">
              <p>Failed to load MIDI</p>
              <p className="pp-error-detail">{error}</p>
              <p className="pp-error-hint">
                Make sure the backend is running and <code>Backend/uploads/test.mid</code> exists.
              </p>
            </div>
          </div>
        )}

        {!isLoading && !error && (
          <FallingNotesCanvas
            notes={notes}
            currentTime={player.currentTime}
            containerWidth={dimensions.width}
            containerHeight={dimensions.height}
            isPlaying={player.isPlaying}
          />
        )}
      </div>

      {/* Piano keyboard at bottom */}
      <div className="pp-keyboard">
        <PianoKeyboard
          activeNotes={activeNotes}
          containerWidth={dimensions.width}
        />
      </div>

      {/* Progress indicator bar */}
      <div className="pp-progress-bar">
        <div
          className="pp-progress-fill"
          style={{ width: duration ? `${(player.currentTime / duration) * 100}%` : '0%' }}
        />
      </div>
    </div>
  );
}
