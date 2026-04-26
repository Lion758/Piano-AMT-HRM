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
  const [libraryOpen, setLibraryOpen] = useState(false);
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
  const [midiLibraryItems, setMidiLibraryItems] = useState([]);
  const [selectedLibraryMidiId, setSelectedLibraryMidiId] = useState('');
  const [libraryLoading, setLibraryLoading] = useState(false);
  const [libraryError, setLibraryError] = useState('');
  const [libraryStatus, setLibraryStatus] = useState('');
  const [libraryUploadTitle, setLibraryUploadTitle] = useState('');
  const [libraryUploadProject, setLibraryUploadProject] = useState('General');
  const [isSavingLibraryMidi, setIsSavingLibraryMidi] = useState(false);

  // Load MIDI
  const {
    notes,
    duration,
    tempo,
    isLoading,
    error,
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

  const loadMidiLibrary = useCallback(async () => {
    setLibraryLoading(true);
    setLibraryError('');
    try {
      const response = await fetch(`${API_BASE}/library/midis`);
      if (!response.ok) {
        throw new Error('Failed to load MIDI library.');
      }

      const data = await response.json();
      setMidiLibraryItems(Array.isArray(data.items) ? data.items : []);
    } catch (err) {
      setLibraryError(err.message || 'Could not load the MIDI library.');
    } finally {
      setLibraryLoading(false);
    }
  }, []);

  useEffect(() => {
    loadMidiLibrary();
  }, [loadMidiLibrary]);

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

  const selectedLibraryMidi = useMemo(
    () => midiLibraryItems.find(item => item.id === selectedLibraryMidiId) || null,
    [midiLibraryItems, selectedLibraryMidiId],
  );

  const referenceReady = !compareToOriginal || Boolean(referenceMidiFile || selectedLibraryMidiId);

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
          setLibraryOpen(false);
          break;
      }
    }
    window.addEventListener('keydown', handleKey);
    return () => window.removeEventListener('keydown', handleKey);
  }, [player, timelineDuration]);

  const handleLibraryOpen = useCallback(() => {
    setLibraryOpen(true);
    setMenuOpen(false);
    loadMidiLibrary();
  }, [loadMidiLibrary]);

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
    if (file) {
      setSelectedLibraryMidiId('');
    }
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
      setSelectedLibraryMidiId('');
      setUploadStatus(
        selectedAudioFile
          ? `Performance ready: ${selectedAudioFile.name}`
          : 'Add your performance audio to open the tutor workspace.',
      );
      return;
    }

    setUploadStatus('Comparison mode enabled. Add the original MIDI to compare against.');
    loadMidiLibrary();
  }, [loadMidiLibrary, selectedAudioFile]);

  const handleLibraryReferenceSelection = useCallback((event) => {
    const libraryMidiId = event.target.value;
    setSelectedLibraryMidiId(libraryMidiId);
    setReferenceMidiFile(null);
    setUploadError('');
    setPreparedTutor(null);

    const item = midiLibraryItems.find(entry => entry.id === libraryMidiId);
    if (item) {
      setUploadStatus(`Reference ready from library: ${item.title || item.original_filename}`);
    } else if (compareToOriginal) {
      setUploadStatus('Choose a saved MIDI or upload a reference MIDI.');
    }
  }, [compareToOriginal, midiLibraryItems]);

  const handleUseLibraryMidiAsReference = useCallback((item) => {
    if (!item?.id) return;
    setCompareToOriginal(true);
    setSelectedLibraryMidiId(item.id);
    setReferenceMidiFile(null);
    setUploadError('');
    setPreparedTutor(null);
    setUploadStatus(`Reference ready from library: ${item.title || item.original_filename}`);
    setLibraryOpen(false);
  }, []);

  const handleLibraryMidiUpload = useCallback(async (event) => {
    const input = event.currentTarget;
    const file = input.files?.[0] || null;
    if (!file) return;

    setIsSavingLibraryMidi(true);
    setLibraryError('');
    setLibraryStatus(`Saving ${file.name} to the MIDI library...`);

    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('title', libraryUploadTitle || file.name.replace(/\.[^.]+$/, ''));
      formData.append('project', libraryUploadProject || 'General');

      const response = await fetch(`${API_BASE}/library/midis`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        let message = 'Failed to save MIDI to library.';
        try {
          const payload = await response.json();
          if (payload?.detail) message = payload.detail;
        } catch {
          // Keep the default message for malformed error payloads.
        }
        throw new Error(message);
      }

      const data = await response.json();
      const item = data.item;
      if (!item?.id) {
        throw new Error('The library did not return a saved MIDI item.');
      }

      setMidiLibraryItems(prev => [item, ...prev.filter(existing => existing.id !== item.id)]);
      setSelectedLibraryMidiId(item.id);
      setReferenceMidiFile(null);
      setCompareToOriginal(true);
      setLibraryUploadTitle('');
      setLibraryStatus(`Saved to ${item.project || 'General'}: ${item.title || item.original_filename}`);
      setUploadStatus(`Reference ready from library: ${item.title || item.original_filename}`);
    } catch (err) {
      setLibraryError(err.message || 'Could not save this MIDI.');
      setLibraryStatus('');
    } finally {
      input.value = '';
      setIsSavingLibraryMidi(false);
    }
  }, [libraryUploadProject, libraryUploadTitle]);

  const handlePrepareTutor = useCallback(async () => {
    if (!selectedAudioFile) {
      setUploadError('Choose your performance audio first.');
      return;
    }
    if (compareToOriginal && !referenceReady) {
      setUploadError('Choose a saved MIDI or upload the original MIDI before starting comparison mode.');
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
      } else if (compareToOriginal && selectedLibraryMidiId) {
        formData.append('reference_midi_library_id', selectedLibraryMidiId);
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
      setSelectedLibraryMidiId('');
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
  }, [compareToOriginal, referenceMidiFile, referenceReady, selectedAudioFile, selectedLibraryMidiId]);

  const preparationSteps = getPreparationSteps(compareToOriginal);

  return (
    <div className="piano-page">
      <LeftMenu
        isOpen={menuOpen}
        onToggle={() => setMenuOpen(v => !v)}
        onLibraryOpen={handleLibraryOpen}
        libraryCount={midiLibraryItems.length}
      />

      <ChatPanel
        isOpen={chatOpen}
        onToggle={() => setChatOpen(v => !v)}
        midiUrl={activeMidiUrl}
        notes={notes}
        analysisData={midiAnalysis}
        analysisLoading={isAnalyzingMidi}
        preparedTutor={preparedTutor}
      />

      <aside className={`midi-library-drawer${libraryOpen ? ' open' : ''}`}>
        <div className="midi-library-drawer-content">
          <div className="midi-library-drawer-header">
            <div>
              <span>MIDI Library</span>
              <strong>Saved reference files and generated transcriptions</strong>
            </div>
            <button
              className="panel-close-btn"
              onClick={() => setLibraryOpen(false)}
              title="Close library"
              type="button"
            >
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" />
              </svg>
            </button>
          </div>

          <div className="midi-library-drawer-actions">
            <button type="button" onClick={loadMidiLibrary} disabled={libraryLoading}>
              {libraryLoading ? 'Refreshing...' : 'Refresh library'}
            </button>
          </div>

          <div className="midi-library-drawer-list">
            {libraryLoading && <p className="midi-library-empty">Loading saved MIDIs...</p>}
            {!libraryLoading && midiLibraryItems.length === 0 && (
              <p className="midi-library-empty">No saved MIDIs yet. Add an original MIDI to start a project library.</p>
            )}
            {!libraryLoading && midiLibraryItems.map(item => (
              <div className="midi-library-item" key={item.id}>
                <div>
                  <strong>{item.title || item.original_filename}</strong>
                  <span>{item.project || 'General'}</span>
                </div>
                <div className="midi-library-item-actions">
                  <button type="button" onClick={() => handleUseLibraryMidiAsReference(item)}>
                    Use as reference
                  </button>
                  <a href={resolveApiUrl(item.download_url)}>Download</a>
                </div>
              </div>
            ))}
          </div>

          <div className="midi-library-drawer-upload">
            <span>Add MIDI to a project</span>
            <div className="pp-midi-library-fields">
              <label>
                <span>Title</span>
                <input
                  type="text"
                  value={libraryUploadTitle}
                  onChange={(event) => setLibraryUploadTitle(event.target.value)}
                  placeholder="Reference title"
                />
              </label>
              <label>
                <span>Project</span>
                <input
                  type="text"
                  value={libraryUploadProject}
                  onChange={(event) => setLibraryUploadProject(event.target.value)}
                  placeholder="Project name"
                />
              </label>
            </div>
            <label className="pp-midi-library-file">
              <span>{isSavingLibraryMidi ? 'Saving MIDI...' : 'Choose MIDI file'}</span>
              <input
                type="file"
                accept=".mid,.midi,audio/midi,audio/x-midi"
                onChange={handleLibraryMidiUpload}
                disabled={isSavingLibraryMidi}
              />
            </label>
            {(libraryStatus || libraryError) && (
              <p className={`pp-midi-library-status${libraryError ? ' error' : ''}`}>
                {libraryError || libraryStatus}
              </p>
            )}
          </div>
        </div>
      </aside>

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
                <>
                  <div className="pp-midi-library">
                    <div className="pp-midi-library-head">
                      <div>
                        <span className="pp-midi-library-kicker">Reference MIDI Library</span>
                        <p>Save original MIDIs by project, choose one for comparison, or download it for later.</p>
                      </div>
                      <button type="button" onClick={loadMidiLibrary} disabled={libraryLoading}>
                        {libraryLoading ? 'Loading' : 'Refresh'}
                      </button>
                    </div>

                    <label className="pp-midi-library-select">
                      <span>Choose saved reference</span>
                      <select
                        value={selectedLibraryMidiId}
                        onChange={handleLibraryReferenceSelection}
                        disabled={libraryLoading || midiLibraryItems.length === 0}
                      >
                        <option value="">
                          {midiLibraryItems.length === 0 ? 'No saved MIDIs yet' : 'No library MIDI selected'}
                        </option>
                        {midiLibraryItems.map(item => (
                          <option key={item.id} value={item.id}>
                            [{item.project || 'General'}] {item.title || item.original_filename}
                          </option>
                        ))}
                      </select>
                    </label>

                    {selectedLibraryMidi && (
                      <div className="pp-midi-library-selected">
                        <div>
                          <strong>{selectedLibraryMidi.title || selectedLibraryMidi.original_filename}</strong>
                          <span>{selectedLibraryMidi.project || 'General'}</span>
                        </div>
                        <a href={resolveApiUrl(selectedLibraryMidi.download_url)}>
                          Download
                        </a>
                      </div>
                    )}

                    <div className="pp-midi-library-add">
                      <div className="pp-midi-library-fields">
                        <label>
                          <span>Title</span>
                          <input
                            type="text"
                            value={libraryUploadTitle}
                            onChange={(event) => setLibraryUploadTitle(event.target.value)}
                            placeholder="Moonlight Sonata reference"
                          />
                        </label>
                        <label>
                          <span>Project</span>
                          <input
                            type="text"
                            value={libraryUploadProject}
                            onChange={(event) => setLibraryUploadProject(event.target.value)}
                            placeholder="Beginner recital"
                          />
                        </label>
                      </div>
                      <label className="pp-midi-library-file">
                        <span>{isSavingLibraryMidi ? 'Saving MIDI...' : 'Add MIDI to library'}</span>
                        <input
                          type="file"
                          accept=".mid,.midi,audio/midi,audio/x-midi"
                          onChange={handleLibraryMidiUpload}
                          disabled={isSavingLibraryMidi}
                        />
                      </label>
                    </div>

                    {(libraryStatus || libraryError) && (
                      <p className={`pp-midi-library-status${libraryError ? ' error' : ''}`}>
                        {libraryError || libraryStatus}
                      </p>
                    )}
                  </div>

                  <label className="pp-upload-picker pp-upload-picker-secondary">
                    <span>Or upload a one-time reference MIDI</span>
                    <input type="file" accept=".mid,.midi,audio/midi,audio/x-midi" onChange={handleReferenceSelection} />
                  </label>
                </>
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
                disabled={!selectedAudioFile || !referenceReady}
              >
                {compareToOriginal ? 'Prepare comparison tutor' : 'Open solo tutor'}
              </button>
              <p className="pp-upload-status">{uploadStatus}</p>
              {selectedAudioFile && <p className="pp-upload-file">Performance: {selectedAudioFile.name}</p>}
              {compareToOriginal && referenceMidiFile && <p className="pp-upload-file">Reference: {referenceMidiFile.name}</p>}
              {compareToOriginal && selectedLibraryMidi && <p className="pp-upload-file">Library reference: {selectedLibraryMidi.title || selectedLibraryMidi.original_filename}</p>}
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
            sustainSpans={sustainSpans}
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
