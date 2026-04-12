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
import { API_BASE, resolveApiUrl } from '../lib/api.js';

export default function PianoPage({ midiUrl }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const mainRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 1200, height: 500 });
  const [activeMidiUrl, setActiveMidiUrl] = useState(midiUrl || null);
  const [selectedAudioFile, setSelectedAudioFile] = useState(null);
  const [isTranscribingUpload, setIsTranscribingUpload] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [uploadStatus, setUploadStatus] = useState('Upload a song to transcribe it and start learning.');

  // Load MIDI
  const { notes, duration, tempo, isLoading, error } = useMidi(activeMidiUrl);

  // Player
  const player = usePianoPlayer(notes, duration, tempo);

  // Recorder
  const recorder = useRecorder();
  const hasMidi = Boolean(activeMidiUrl);
  const hasNotes = notes.length > 0 && duration > 0;
  const controlsReady = player.isLoaded && hasNotes;

  // Speed-synced visual time
  const visualTime = player.currentTime;

  useEffect(() => {
    setActiveMidiUrl(midiUrl || null);
  }, [midiUrl]);

  // Track container dimensions
  useEffect(() => {
    const el = mainRef.current;
    if (!el) return;
    const observer = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        setDimensions({ width, height: Math.max(height - 140, 200) });
      }
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  // Compute active notes using visual time
  const activeNotes = useMemo(() => {
    const t = visualTime;
    return notes.filter(n => t >= n.time && t < n.time + n.duration);
  }, [notes, visualTime]);

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

  const handleAudioSelection = useCallback((event) => {
    const file = event.target.files?.[0] || null;
    setSelectedAudioFile(file);
    setUploadError('');
    if (file) {
      setUploadStatus(`Ready to transcribe: ${file.name}`);
    } else {
      setUploadStatus('Upload a song to transcribe it and start learning.');
    }
  }, []);

  const handleUploadAndTranscribe = useCallback(async () => {
    if (!selectedAudioFile) {
      setUploadError('Choose an audio file first.');
      return;
    }
    setUploadError('');
    setIsTranscribingUpload(true);
    setUploadStatus('Uploading audio and transcribing with the seq2seq model...');
    try {
      const formData = new FormData();
      formData.append('file', selectedAudioFile);
      const response = await fetch(`${API_BASE}/transcribe-upload`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        let message = 'Failed to transcribe audio.';
        try {
          const payload = await response.json();
          if (payload?.detail) message = payload.detail;
        } catch { }
        throw new Error(message);
      }
      const data = await response.json();
      const nextMidiUrl = resolveApiUrl(data.midi_url);
      setUploadStatus('Opening your transcription...');
      setSelectedAudioFile(null);
      setUploadError('');
      setActiveMidiUrl(nextMidiUrl);
      window.location.hash = `#/piano?midi=${encodeURIComponent(nextMidiUrl)}`;
    } catch (err) {
      setUploadError(err.message || 'Something went wrong.');
      setUploadStatus('Upload a song to transcribe it and start learning.');
    } finally {
      setIsTranscribingUpload(false);
    }
  }, [selectedAudioFile]);

  return (
    <div className="piano-page">
      <LeftMenu isOpen={menuOpen} onToggle={() => setMenuOpen(v => !v)} />

      <ChatPanel
        isOpen={chatOpen}
        onToggle={() => setChatOpen(v => !v)}
        midiUrl={activeMidiUrl}
        notes={notes}
      />

      <div className="pp-top">
        <button
          className="pp-play-fab"
          onClick={player.isPlaying ? player.pause : player.play}
          disabled={!controlsReady}
          title={player.isPlaying ? 'Pause (Space)' : 'Play (Space)'}
        >
          {player.isPlaying ? (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="6,4 20,12 6,20" />
            </svg>
          )}
        </button>

        <TopControls
          isPlaying={player.isPlaying}
          currentTime={visualTime}
          duration={player.duration || duration}
          speed={player.speed}
          volume={player.volume}
          isLoaded={controlsReady}
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

      <div className="pp-main" ref={mainRef}>
        {!hasMidi && !isTranscribingUpload && (
          <div className="pp-overlay">
            <div className="pp-upload-card">
              <h2>Upload A Song To Start Learning</h2>
              <p>We&apos;ll transcribe it with the efficient seq2seq transformer and open it here automatically.</p>
              <label className="pp-upload-picker">
                <span>Choose audio file</span>
                <input type="file" accept="audio/*" onChange={handleAudioSelection} />
              </label>
              <button
                className="pp-upload-action"
                onClick={handleUploadAndTranscribe}
                disabled={!selectedAudioFile}
              >
                Transcribe Song
              </button>
              <p className="pp-upload-status">{uploadStatus}</p>
              {selectedAudioFile && <p className="pp-upload-file">{selectedAudioFile.name}</p>}
              {uploadError && <p className="pp-upload-error">{uploadError}</p>}
            </div>
          </div>
        )}

        {isTranscribingUpload && (
          <div className="pp-overlay">
            <div className="pp-loader">
              <div className="spinner" />
              <p>{uploadStatus}</p>
            </div>
          </div>
        )}

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
                {midiUrl
                  ? "Make sure the backend is running and the generated MIDI file is still available."
                  : "Upload a song to start a new transcription."}
              </p>
            </div>
          </div>
        )}

        {!isLoading && !error && hasMidi && (
          <FallingNotesCanvas
            notes={notes}
            currentTime={visualTime}
            containerWidth={dimensions.width}
            containerHeight={dimensions.height}
            isPlaying={player.isPlaying}
          />
        )}
      </div>

      <div className="pp-keyboard">
        <PianoKeyboard
          activeNotes={activeNotes}
          containerWidth={dimensions.width}
        />
      </div>

      <div className="pp-progress-bar">
        <div
          className="pp-progress-fill"
          style={{ width: duration ? `${(visualTime / duration) * 100}%` : '0%' }}
        />
      </div>
    </div>
  );
}