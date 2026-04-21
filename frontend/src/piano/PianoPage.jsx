import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import './PianoPage.css';
import TopControls from './components/TopControls.jsx';
import FallingNotesCanvas from './components/FallingNotesCanvas.jsx';
import PianoKeyboard from './components/PianoKeyboard.jsx';
import LeftMenu from './components/LeftMenu.jsx';
import ChatPanel from './components/ChatPanel.jsx';
import { useMidi } from './hooks/useMidi.js';
import { usePianoPlayer } from './hooks/usePianoPlayer.js';
import { useRecorder } from './hooks/useRecorder.js';
import { API_BASE, resolveApiUrl } from '../lib/api.js';
import grandPianoTheater from '../assets/grand-piano-indoors-theater-place-generative-ai.jpg';

export default function PianoPage({ midiUrl }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const mainRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 1200, height: 500 });
  const [activeMidiUrl, setActiveMidiUrl] = useState(midiUrl || null);
  const [midiAnalysis, setMidiAnalysis] = useState(null);
  const [isAnalyzingMidi, setIsAnalyzingMidi] = useState(false);
  const [selectedAudioFile, setSelectedAudioFile] = useState(null);
  const [isTranscribingUpload, setIsTranscribingUpload] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [uploadStatus, setUploadStatus] = useState('Upload a song to transcribe it and start learning.');

  // Load MIDI
  const {
    notes,
    duration,
    tempo,
    isLoading,
    error,
    sustainEvents,
    sustainSpans,
    playbackDuration,
  } = useMidi(activeMidiUrl);

  // Player
  const player = usePianoPlayer(notes, duration, tempo, sustainSpans, playbackDuration);

  // Recorder
  const recorder = useRecorder();
  const hasMidi = Boolean(activeMidiUrl);
  const hasNotes = notes.length > 0 && duration > 0;
  const controlsReady = player.isLoaded && hasNotes;
  const timelineDuration = player.duration || playbackDuration || duration;

  // Speed-synced visual time
  const visualTime = player.currentTime;

  useEffect(() => {
    setActiveMidiUrl(midiUrl || null);
  }, [midiUrl]);

  useEffect(() => {
    let cancelled = false;

    if (!activeMidiUrl) {
      setMidiAnalysis(null);
      setIsAnalyzingMidi(false);
      return () => {
        cancelled = true;
      };
    }

    async function analyzeMidi() {
      setIsAnalyzingMidi(true);
      try {
        const response = await fetch(`${API_BASE}/midi/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ midi_url: activeMidiUrl }),
        });

        if (!response.ok) {
          throw new Error('Failed to analyze MIDI.');
        }

        const data = await response.json();
        if (!cancelled) {
          setMidiAnalysis(data);
        }
      } catch {
        if (!cancelled) {
          setMidiAnalysis(null);
        }
      } finally {
        if (!cancelled) {
          setIsAnalyzingMidi(false);
        }
      }
    }

    analyzeMidi();
    return () => {
      cancelled = true;
    };
  }, [activeMidiUrl]);

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
          player.seek(Math.min(timelineDuration, player.currentTime + 5));
          break;
        case 'Escape':
          setMenuOpen(false);
          setChatOpen(false);
          break;
      }
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [player, timelineDuration]);

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
        } catch {
          // Ignore malformed error payloads and keep the default message.
        }
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
        analysisData={midiAnalysis}
        analysisLoading={isAnalyzingMidi}
      />

      <div className="pp-top">
        <TopControls
          isPlaying={player.isPlaying}
          currentTime={visualTime}
          duration={timelineDuration}
          speed={player.speed}
          volume={player.volume}
          isLoaded={controlsReady}
          isMenuOpen={menuOpen}
          isTutorOpen={chatOpen}
          recordProps={{
            isRecording: recorder.isRecording,
            onStart: recorder.startRecording,
            onStop: recorder.stopRecording,
            audioURL: recorder.audioURL,
            error: recorder.error,
          }}
          onPlay={player.play}
          onPause={player.pause}
          onStop={player.stop}
          onSeek={player.seek}
          onSpeedChange={player.setSpeed}
          onVolumeChange={player.setVolume}
          onMenuToggle={() => setMenuOpen(v => !v)}
          onTutorToggle={() => setChatOpen(v => !v)}
        />
      </div>

      <div className="pp-main" ref={mainRef}>
        <div className={`pp-scene-badge${hasMidi ? ' active' : ''}`}>
          <span className="pp-scene-dot" />
          {hasMidi ? 'Tutor stage active' : 'Awaiting transcription'}
        </div>

        {!hasMidi && !isTranscribingUpload && (
          <div className="pp-overlay">
            <div className="pp-upload-card">
              <div className="pp-upload-photo">
                <img
                  src={grandPianoTheater}
                  alt="Grand piano in a warmly lit theater"
                  className="pp-upload-photo-image"
                />
                <div className="pp-upload-photo-overlay">
                  <span>Stage Reference</span>
                  <strong>Grand piano, amber lighting, and a concert-hall mood</strong>
                </div>
              </div>
              <div className="pp-upload-kicker">Tutor Workspace</div>
              <h2>Bring a recording into the piano stage</h2>
              <p>Transcribe the performance, open the falling-note view, and keep the practice controls in the same workspace.</p>
              <div className="pp-upload-highlights">
                <span>Audio input</span>
                <span>MIDI output</span>
                <span>Guided practice</span>
              </div>
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
            sustainEvents={sustainEvents}
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
          style={{ width: timelineDuration ? `${Math.min((visualTime / timelineDuration) * 100, 100)}%` : '0%' }}
        />
      </div>
    </div>
  );
}
