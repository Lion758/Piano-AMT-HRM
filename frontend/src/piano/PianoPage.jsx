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

const PREPARATION_STEP_LABELS = {
  uploading: 'Uploading',
  transcribing: 'Transcribing',
  comparing: 'Comparing',
  ready: 'Tutor ready',
};

function getPreparationSteps(includeComparison) {
  return includeComparison
    ? ['uploading', 'transcribing', 'comparing', 'ready']
    : ['uploading', 'transcribing', 'ready'];
}

export default function PianoPage({ midiUrl }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const mainRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 1200, height: 500 });
  const [activeMidiUrl, setActiveMidiUrl] = useState(midiUrl || null);
  const [midiAnalysis, setMidiAnalysis] = useState(null);
  const [isAnalyzingMidi, setIsAnalyzingMidi] = useState(false);
  const [selectedAudioFile, setSelectedAudioFile] = useState(null);
  const [referenceMidiFile, setReferenceMidiFile] = useState(null);
  const [compareToOriginal, setCompareToOriginal] = useState(false);
  const [isPreparingTutor, setIsPreparingTutor] = useState(false);
  const [preparePhase, setPreparePhase] = useState('idle');
  const [uploadError, setUploadError] = useState('');
  const [uploadStatus, setUploadStatus] = useState('Add your performance audio to open the tutor workspace.');
  const [preparedTutor, setPreparedTutor] = useState(null);

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
    setPreparedTutor(null);
    if (file) {
      setUploadStatus(`Performance ready: ${file.name}`);
    } else {
      setUploadStatus('Add your performance audio to open the tutor workspace.');
    }
  }, []);

  const handleReferenceSelection = useCallback((event) => {
    const file = event.target.files?.[0] || null;
    setReferenceMidiFile(file);
    setUploadError('');
    setPreparedTutor(null);
    if (file) {
      setUploadStatus(`Reference ready: ${file.name}`);
    } else if (compareToOriginal) {
      setUploadStatus('Add the original MIDI to prepare a comparison tutor.');
    }
  }, [compareToOriginal]);

  const handleCompareToggle = useCallback((event) => {
    const checked = event.target.checked;
    setCompareToOriginal(checked);
    setUploadError('');
    setPreparedTutor(null);
    if (!checked) {
      setReferenceMidiFile(null);
      setUploadStatus(
        selectedAudioFile
          ? `Performance ready: ${selectedAudioFile.name}`
          : 'Add your performance audio to open the tutor workspace.',
      );
      return;
    }

    setUploadStatus('Comparison mode enabled. Add the original MIDI to compare against.');
  }, [selectedAudioFile]);

  const handlePrepareTutor = useCallback(async () => {
    if (!selectedAudioFile) {
      setUploadError('Choose your performance audio first.');
      return;
    }
    if (compareToOriginal && !referenceMidiFile) {
      setUploadError('Choose the original MIDI before starting comparison mode.');
      return;
    }

    const phaseSteps = getPreparationSteps(compareToOriginal);
    let phaseIndex = 0;
    let phaseTimer = null;

    setUploadError('');
    setPreparedTutor(null);
    setIsPreparingTutor(true);
    setPreparePhase(phaseSteps[0]);
    setUploadStatus(compareToOriginal
      ? 'Uploading your files and preparing the comparison tutor...'
      : 'Uploading your performance and preparing the tutor...');

    phaseTimer = window.setInterval(() => {
      phaseIndex = Math.min(phaseIndex + 1, phaseSteps.length - 2);
      setPreparePhase(phaseSteps[phaseIndex]);
    }, 1800);

    try {
      const formData = new FormData();
      formData.append('performance_audio', selectedAudioFile);
      if (compareToOriginal && referenceMidiFile) {
        formData.append('reference_midi', referenceMidiFile);
      }
      const response = await fetch(`${API_BASE}/tutor/prepare`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        let message = 'Failed to prepare the tutor session.';
        try {
          const payload = await response.json();
          if (payload?.detail) message = payload.detail;
        } catch {
          // Ignore malformed error payloads and keep the default message.
        }
        throw new Error(message);
      }
      const data = await response.json();
      const nextMidiUrl = resolveApiUrl(data.performance_midi_url);
      setPreparePhase('ready');
      setUploadStatus('Tutor ready. Opening your practice workspace...');
      setSelectedAudioFile(null);
      setReferenceMidiFile(null);
      setCompareToOriginal(false);
      setUploadError('');
      setPreparedTutor({
        mode: data.mode || 'solo',
        summaryCards: data.summary_cards || null,
        suggestedQuestions: Array.isArray(data.suggested_questions) ? data.suggested_questions : [],
        sessionId: data.tutor?.session_id || null,
        openingMessage: data.tutor?.opening_message || '',
      });
      setActiveMidiUrl(nextMidiUrl);
      setChatOpen(true);
      window.location.hash = `#/piano?midi=${encodeURIComponent(nextMidiUrl)}`;
    } catch (err) {
      setUploadError(err.message || 'Something went wrong.');
      setPreparePhase('idle');
      setUploadStatus('Add your files and try again.');
    } finally {
      if (phaseTimer) window.clearInterval(phaseTimer);
      setIsPreparingTutor(false);
    }
  }, [compareToOriginal, referenceMidiFile, selectedAudioFile]);

  const preparationSteps = getPreparationSteps(compareToOriginal);

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
        preparedTutor={preparedTutor}
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

        {!hasMidi && !isPreparingTutor && (
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
                  <strong>Performance audio in, guided coaching out, with an optional original MIDI for comparison.</strong>
                </div>
              </div>
              <div className="pp-upload-kicker">Comparison Tutor</div>
              <h2>Bring your performance into the tutor studio</h2>
              <p>Upload your performance audio, optionally add the original MIDI, and open straight into a coach-ready practice session.</p>
              <div className="pp-upload-highlights">
                <span>Performance audio</span>
                <span>Optional reference MIDI</span>
                <span>Guided coaching</span>
              </div>
              <label className="pp-upload-picker">
                <span>Your performance audio</span>
                <input type="file" accept="audio/*" onChange={handleAudioSelection} />
              </label>
              <label className="pp-compare-toggle">
                <input
                  type="checkbox"
                  checked={compareToOriginal}
                  onChange={handleCompareToggle}
                />
                <span>Compare this performance to the original MIDI</span>
              </label>
              {compareToOriginal && (
                <label className="pp-upload-picker pp-upload-picker-secondary">
                  <span>Original / reference MIDI</span>
                  <input type="file" accept=".mid,.midi,audio/midi,audio/x-midi" onChange={handleReferenceSelection} />
                </label>
              )}
              <div className="pp-upload-steps">
                {preparationSteps.map((step) => (
                  <div key={step} className="pp-upload-step pending">
                    <span>{PREPARATION_STEP_LABELS[step]}</span>
                  </div>
                ))}
              </div>
              <button
                className="pp-upload-action"
                onClick={handlePrepareTutor}
                disabled={!selectedAudioFile || (compareToOriginal && !referenceMidiFile)}
              >
                {compareToOriginal ? 'Prepare comparison tutor' : 'Open solo tutor'}
              </button>
              <p className="pp-upload-status">{uploadStatus}</p>
              {selectedAudioFile && <p className="pp-upload-file">Performance: {selectedAudioFile.name}</p>}
              {compareToOriginal && referenceMidiFile && <p className="pp-upload-file">Reference: {referenceMidiFile.name}</p>}
              {uploadError && <p className="pp-upload-error">{uploadError}</p>}
            </div>
          </div>
        )}

        {isPreparingTutor && (
          <div className="pp-overlay">
            <div className="pp-loader">
              <div className="spinner" />
              <p>{uploadStatus}</p>
              <div className="pp-upload-steps pp-upload-steps-live">
                {preparationSteps.map((step) => {
                  const stepIndex = preparationSteps.indexOf(step);
                  const activeIndex = preparationSteps.indexOf(preparePhase);
                  const state = activeIndex > stepIndex ? 'done' : activeIndex === stepIndex ? 'active' : 'pending';
                  return (
                    <div key={step} className={`pp-upload-step ${state}`}>
                      <span>{PREPARATION_STEP_LABELS[step]}</span>
                    </div>
                  );
                })}
              </div>
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
