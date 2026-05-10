import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import './styles/PianoPage.css';
import TopControls from './components/TopControls.jsx';
import FallingNotesCanvas from './components/FallingNotesCanvas.jsx';
import PianoKeyboard from './components/PianoKeyboard.jsx';
import LeftMenu from './components/LeftMenu.jsx';
import ChatPanel from './components/ChatPanel.jsx';
import { useMidi } from './hooks/useMidi.js';
import { usePianoPlayer } from './hooks/usePianoPlayer.js';
import { API_BASE, resolveApiUrl } from '../shared/api.js';
import grandPianoTheater from '../assets/grand-piano-indoors-theater-place-generative-ai.jpg';

const PREPARATION_STEP_LABELS = {
  uploading: 'Uploading',
  transcribing: 'Transcribing',
  comparing: 'Comparing',
  ready: 'Ready',
};

function getPreparationSteps(includeComparison, skipTranscription = false) {
  const steps = ['uploading'];
  if (!skipTranscription) steps.push('transcribing');
  if (includeComparison) steps.push('comparing');
  steps.push('ready');
  return steps;
}

function isMidiFile(file) {
  if (!file) return false;
  const name = file.name || '';
  const type = file.type || '';
  return /\.(mid|midi)$/i.test(name) || type === 'audio/midi' || type === 'audio/x-midi';
}

function titleFromFilename(filename, fallback = 'Untitled Song') {
  const stem = String(filename || '').replace(/\.[^.]+$/, '').trim();
  return stem.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim() || fallback;
}

function getProjectFromPrepareResponse(data) {
  const project = data?.project || {};
  const referenceLibraryItem = project.reference_library_item || data?.reference_library_item || null;
  const performanceLibraryItem = project.performance_library_item || data?.performance_library_item || null;
  const name = project.name || referenceLibraryItem?.project || performanceLibraryItem?.project || null;

  if (!name && !referenceLibraryItem && !performanceLibraryItem) {
    return null;
  }

  return {
    name,
    referenceLibraryItem,
    performanceLibraryItem,
  };
}

function getLibraryRoleLabel(role) {
  if (role === 'reference') return 'Reference';
  if (role === 'performance') return 'Performance';
  return 'MIDI';
}

function buildPianoHash(midiUrl, project = null) {
  const params = new URLSearchParams({ midi: midiUrl });
  const referenceItem = project?.referenceLibraryItem;
  if (referenceItem?.id) params.set('reference', referenceItem.id);
  if (project?.name || referenceItem?.project) params.set('project', project?.name || referenceItem.project);
  if (referenceItem?.title) params.set('referenceTitle', referenceItem.title);
  return `#/piano?${params.toString()}`;
}

function mergeLibraryItems(existingItems, ...itemsToMerge) {
  const cleanItems = itemsToMerge.filter(item => item?.id);
  if (cleanItems.length === 0) return existingItems;
  const incomingIds = new Set(cleanItems.map(item => item.id));
  return [...cleanItems, ...existingItems.filter(item => !incomingIds.has(item.id))];
}

export default function PianoPage({ midiUrl, projectName = null, referenceLibraryId = null, referenceTitle = null }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const [chatOpen, setChatOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const mainRef = useRef(null);
  const [dimensions, setDimensions] = useState({ width: 1200, height: 500 });
  const [activeMidiUrl, setActiveMidiUrl] = useState(midiUrl || null);
  const [midiAnalysis, setMidiAnalysis] = useState(null);
  const [isAnalyzingMidi, setIsAnalyzingMidi] = useState(false);
  const [selectedAudioFile, setSelectedAudioFile] = useState(null);
  const [isPreparingTutor, setIsPreparingTutor] = useState(false);
  const [preparePhase, setPreparePhase] = useState('idle');
  const [uploadError, setUploadError] = useState('');
  const [uploadStatus, setUploadStatus] = useState('Add a song audio file or MIDI to open the player.');
  const [preparedTutor, setPreparedTutor] = useState(null);
  const [comparisonPanelOpen, setComparisonPanelOpen] = useState(false);
  const [comparisonPerformanceFile, setComparisonPerformanceFile] = useState(null);
  const [comparisonReferenceFile, setComparisonReferenceFile] = useState(null);
  const [comparisonStatus, setComparisonStatus] = useState('Choose your performance file and the original MIDI.');
  const [comparisonError, setComparisonError] = useState('');
  const [midiLibraryItems, setMidiLibraryItems] = useState([]);
  const [libraryLoading, setLibraryLoading] = useState(false);
  const [libraryError, setLibraryError] = useState('');
  const [libraryStatus, setLibraryStatus] = useState('');
  const [libraryUploadTitle, setLibraryUploadTitle] = useState('');
  const [libraryUploadProject, setLibraryUploadProject] = useState('General');
  const [isSavingLibraryMidi, setIsSavingLibraryMidi] = useState(false);
  const [currentProject, setCurrentProject] = useState(() => {
    if (!referenceLibraryId && !projectName) return null;
    const referenceLibraryItem = referenceLibraryId
      ? {
          id: referenceLibraryId,
          title: referenceTitle || projectName || 'Current reference',
          project: projectName || 'Current Project',
          role: 'reference',
        }
      : null;
    return {
      name: projectName || referenceLibraryItem?.project || null,
      referenceLibraryItem,
      performanceLibraryItem: null,
    };
  });

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

  const hasMidi = Boolean(activeMidiUrl);
  const hasNotes = notes.length > 0 && duration > 0;
  const controlsReady = player.isLoaded && hasNotes;
  const shouldShowMidiStage = hasMidi || isLoading || Boolean(error);
  const timelineDuration = player.duration || playbackDuration || duration;
  const performanceIsMidi = useMemo(() => isMidiFile(selectedAudioFile), [selectedAudioFile]);
  const comparisonPerformanceIsMidi = useMemo(
    () => isMidiFile(comparisonPerformanceFile),
    [comparisonPerformanceFile],
  );
  const projectReferenceItem = currentProject?.referenceLibraryItem || null;
  const midiLibraryProjects = useMemo(() => {
    const groups = new Map();
    midiLibraryItems.forEach((item) => {
      const name = item.project || 'General';
      if (!groups.has(name)) groups.set(name, []);
      groups.get(name).push(item);
    });
    return Array.from(groups, ([name, items]) => ({ name, items }));
  }, [midiLibraryItems]);

  // Speed-synced visual time
  const visualTime = player.currentTime;

  useEffect(() => {
    setActiveMidiUrl(midiUrl || null);
    if (referenceLibraryId || projectName) {
      const referenceLibraryItem = referenceLibraryId
        ? {
            id: referenceLibraryId,
            title: referenceTitle || projectName || 'Current reference',
            project: projectName || 'Current Project',
            role: 'reference',
          }
        : null;
      setCurrentProject(prev => {
        if (prev?.referenceLibraryItem?.id && prev.referenceLibraryItem.id === referenceLibraryId) {
          return {
            name: projectName || prev.name || referenceLibraryItem?.project || null,
            referenceLibraryItem: {
              ...referenceLibraryItem,
              ...prev.referenceLibraryItem,
              title: prev.referenceLibraryItem.title || referenceLibraryItem?.title,
              project: prev.referenceLibraryItem.project || referenceLibraryItem?.project,
            },
            performanceLibraryItem: prev.performanceLibraryItem || null,
          };
        }
        return {
          name: projectName || referenceLibraryItem?.project || null,
          referenceLibraryItem,
          performanceLibraryItem: null,
        };
      });
    } else {
      setCurrentProject(null);
    }
  }, [midiUrl, projectName, referenceLibraryId, referenceTitle]);

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
    if (!currentProject?.referenceLibraryItem?.id || midiLibraryItems.length === 0) return;
    const freshReference = midiLibraryItems.find(item => item.id === currentProject.referenceLibraryItem.id);
    if (!freshReference) return;
    setCurrentProject(prev => ({
      name: prev?.name || freshReference.project,
      referenceLibraryItem: freshReference,
      performanceLibraryItem: prev?.performanceLibraryItem || null,
    }));
  }, [currentProject?.referenceLibraryItem?.id, midiLibraryItems]);

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
          setLibraryOpen(false);
          setComparisonPanelOpen(false);
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

  const handleUseLibraryReference = useCallback((item) => {
    if (!item?.id || !item?.download_url) return;
    const nextMidiUrl = resolveApiUrl(item.download_url);
    const nextProject = {
      name: item.project || item.title || 'Library Project',
      referenceLibraryItem: item,
      performanceLibraryItem: null,
    };
    setCurrentProject(nextProject);
    setPreparedTutor(null);
    setActiveMidiUrl(nextMidiUrl);
    setLibraryOpen(false);
    window.location.hash = buildPianoHash(nextMidiUrl, nextProject);
  }, []);

  const handleAudioSelection = useCallback((event) => {
    const file = event.target.files?.[0] || null;
    setSelectedAudioFile(file);
    setUploadError('');
    setPreparedTutor(null);
    if (file) {
      setUploadStatus(`Song ready: ${file.name}`);
    } else {
      setUploadStatus('Add a song audio file or MIDI to open the player.');
    }
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
      setLibraryUploadTitle('');
      setLibraryStatus(`Saved to ${item.project || 'General'}: ${item.title || item.original_filename}`);
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
      setUploadError('Choose a song audio file or MIDI first.');
      return;
    }

    const phaseSteps = getPreparationSteps(false, performanceIsMidi);
    let phaseIndex = 0;
    let phaseTimer = null;

    setUploadError('');
    setPreparedTutor(null);
    setIsPreparingTutor(true);
    setPreparePhase(phaseSteps[0]);
    setUploadStatus('Uploading your song and preparing the player...');

    phaseTimer = window.setInterval(() => {
      phaseIndex = Math.min(phaseIndex + 1, phaseSteps.length - 2);
      setPreparePhase(phaseSteps[phaseIndex]);
    }, 1800);

    try {
      const formData = new FormData();
      formData.append(performanceIsMidi ? 'performance_midi' : 'performance_audio', selectedAudioFile);
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
      const nextProject = getProjectFromPrepareResponse(data) || {
        name: titleFromFilename(selectedAudioFile.name),
        referenceLibraryItem: null,
        performanceLibraryItem: null,
      };
      setPreparePhase('ready');
      setUploadStatus('Player ready. Opening your practice workspace...');
      setSelectedAudioFile(null);
      setUploadError('');
      setCurrentProject(nextProject);
      setMidiLibraryItems(prev => mergeLibraryItems(
        prev,
        nextProject?.referenceLibraryItem,
        nextProject?.performanceLibraryItem,
      ));
      setPreparedTutor({
        mode: data.mode || 'solo',
        performanceSourceKind: data.performance_source_kind || null,
        projectName: nextProject?.name || null,
        summaryCards: data.summary_cards || null,
        suggestedQuestions: Array.isArray(data.suggested_questions) ? data.suggested_questions : [],
        sessionId: data.tutor?.session_id || null,
        openingMessage: data.tutor?.opening_message || '',
      });
      setActiveMidiUrl(nextMidiUrl);
      setChatOpen(false);
      window.location.hash = buildPianoHash(nextMidiUrl, nextProject);
    } catch (err) {
      setUploadError(err.message || 'Something went wrong.');
      setPreparePhase('idle');
      setUploadStatus('Add your files and try again.');
    } finally {
      if (phaseTimer) window.clearInterval(phaseTimer);
      setIsPreparingTutor(false);
    }
  }, [performanceIsMidi, selectedAudioFile]);

  const handleOpenComparisonPanel = useCallback(() => {
    setComparisonPanelOpen(true);
    setComparisonError('');
    setComparisonReferenceFile(null);
    setComparisonStatus(
      projectReferenceItem?.id
        ? `Reference ready from ${projectReferenceItem.project || currentProject?.name || 'this project'}: ${projectReferenceItem.title || 'saved MIDI'}`
        : 'Choose your performance file and the original MIDI.',
    );
    setMenuOpen(false);
    setLibraryOpen(false);
    setChatOpen(false);
  }, [currentProject?.name, projectReferenceItem]);

  const handleCloseComparisonPanel = useCallback(() => {
    if (isPreparingTutor) return;
    setComparisonPanelOpen(false);
    setComparisonError('');
  }, [isPreparingTutor]);

  const handleComparisonPerformanceSelection = useCallback((event) => {
    const file = event.target.files?.[0] || null;
    setComparisonPerformanceFile(file);
    setComparisonError('');
    if (!file) {
      setComparisonStatus(
        projectReferenceItem?.id
          ? `Reference ready from ${projectReferenceItem.project || currentProject?.name || 'this project'}: ${projectReferenceItem.title || 'saved MIDI'}`
          : 'Choose your performance file and the original MIDI.',
      );
      return;
    }
    setComparisonStatus(
      projectReferenceItem?.id
        ? `Performance ready: ${file.name}. Using saved project reference automatically.`
        : `Performance ready: ${file.name}`,
    );
  }, [currentProject?.name, projectReferenceItem]);

  const handleComparisonReferenceSelection = useCallback((event) => {
    const file = event.target.files?.[0] || null;
    setComparisonReferenceFile(file);
    setComparisonError('');
    setComparisonStatus(file ? `Reference ready: ${file.name}` : 'Choose the original MIDI reference.');
  }, []);

  const handlePrepareComparison = useCallback(async () => {
    if (!comparisonPerformanceFile) {
      setComparisonError('Choose the performance audio or MIDI first.');
      return;
    }
    if (!projectReferenceItem?.id && !comparisonReferenceFile) {
      setComparisonError('Choose the original reference MIDI.');
      return;
    }

    const comparisonIsMidi = isMidiFile(comparisonPerformanceFile);
    const phaseSteps = getPreparationSteps(true, comparisonIsMidi);
    let phaseIndex = 0;
    let phaseTimer = null;

    setComparisonError('');
    setPreparedTutor(null);
    setIsPreparingTutor(true);
    setPreparePhase(phaseSteps[0]);
    const referenceStatus = projectReferenceItem?.id
      ? 'using the saved project reference'
      : 'using the selected reference MIDI';
    const nextStatus = comparisonIsMidi
      ? `Uploading performance MIDI and ${referenceStatus}...`
      : `Uploading performance audio and ${referenceStatus}...`;
    setComparisonStatus(nextStatus);
    setUploadStatus(nextStatus);

    phaseTimer = window.setInterval(() => {
      phaseIndex = Math.min(phaseIndex + 1, phaseSteps.length - 2);
      setPreparePhase(phaseSteps[phaseIndex]);
    }, 1800);

    try {
      const formData = new FormData();
      formData.append(comparisonIsMidi ? 'performance_midi' : 'performance_audio', comparisonPerformanceFile);
      if (projectReferenceItem?.id) {
        formData.append('reference_midi_library_id', projectReferenceItem.id);
      } else {
        formData.append('reference_midi', comparisonReferenceFile);
      }

      const response = await fetch(`${API_BASE}/tutor/prepare`, {
        method: 'POST',
        body: formData,
      });
      if (!response.ok) {
        let message = 'Failed to prepare the comparison.';
        try {
          const payload = await response.json();
          if (payload?.detail) message = payload.detail;
        } catch {
          // Keep the default message for malformed error payloads.
        }
        throw new Error(message);
      }

      const data = await response.json();
      const nextMidiUrl = resolveApiUrl(data.performance_midi_url);
      const nextProject = getProjectFromPrepareResponse(data) || currentProject;
      setPreparePhase('ready');
      setComparisonStatus('Comparison ready. Opening the tutor feedback...');
      setComparisonPanelOpen(false);
      setComparisonPerformanceFile(null);
      setComparisonReferenceFile(null);
      setUploadStatus('Comparison tutor ready.');
      setUploadError('');
      setCurrentProject(nextProject);
      setMidiLibraryItems(prev => mergeLibraryItems(
        prev,
        nextProject?.performanceLibraryItem,
        nextProject?.referenceLibraryItem,
      ));
      setPreparedTutor({
        mode: data.mode || 'compare',
        performanceSourceKind: data.performance_source_kind || null,
        projectName: nextProject?.name || null,
        summaryCards: data.summary_cards || null,
        suggestedQuestions: Array.isArray(data.suggested_questions) ? data.suggested_questions : [],
        sessionId: data.tutor?.session_id || null,
        openingMessage: data.tutor?.opening_message || '',
      });
      setActiveMidiUrl(nextMidiUrl);
      setChatOpen(true);
      window.location.hash = buildPianoHash(nextMidiUrl, nextProject);
    } catch (err) {
      setComparisonError(err.message || 'Something went wrong.');
      setPreparePhase('idle');
      setComparisonStatus('Check the files and try again.');
    } finally {
      if (phaseTimer) window.clearInterval(phaseTimer);
      setIsPreparingTutor(false);
    }
  }, [comparisonPerformanceFile, comparisonReferenceFile, currentProject, projectReferenceItem]);

  const preparationSteps = getPreparationSteps(false, performanceIsMidi);
  const comparisonPreparationSteps = getPreparationSteps(true, comparisonPerformanceIsMidi);

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
        projectName={currentProject?.name || preparedTutor?.projectName || null}
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
            {!libraryLoading && midiLibraryProjects.map(project => (
              <section className="midi-library-project" key={project.name}>
                <div className="midi-library-project-header">
                  <strong>{project.name}</strong>
                  <span>{project.items.length} MIDI{project.items.length === 1 ? '' : 's'}</span>
                </div>
                {project.items.map(item => (
                  <div className="midi-library-item" key={item.id}>
                    <div className="midi-library-item-main">
                      <span className={`midi-library-role ${item.role || 'midi'}`}>
                        {getLibraryRoleLabel(item.role)}
                      </span>
                      <strong>{item.title || item.original_filename}</strong>
                      <span>
                        {item.tutor_session_id
                          ? `${item.tutor_mode === 'compare' ? 'Comparison' : 'Solo'} tutor saved`
                          : item.original_filename}
                      </span>
                    </div>
                    <div className="midi-library-item-actions">
                      {(item.role === 'reference' || !item.role || item.role === 'midi') && (
                        <button type="button" onClick={() => handleUseLibraryReference(item)}>
                          Use reference
                        </button>
                      )}
                      <a href={resolveApiUrl(item.download_url)}>Download</a>
                    </div>
                  </div>
                ))}
              </section>
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

      {comparisonPanelOpen && (
        <div
          className="pp-compare-panel-backdrop"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) handleCloseComparisonPanel();
          }}
        >
          <section
            className="pp-compare-panel"
            role="dialog"
            aria-modal="true"
            aria-labelledby="pp-compare-title"
          >
            <div className="pp-compare-panel-header">
              <div>
                <span>Performance Comparison</span>
                <strong id="pp-compare-title">Compare with project reference</strong>
              </div>
              <button
                className="panel-close-btn"
                onClick={handleCloseComparisonPanel}
                disabled={isPreparingTutor}
                title="Close comparison"
                type="button"
              >
                <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <path d="M19 6.41L17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12z" />
                </svg>
              </button>
            </div>

            <p className="pp-compare-panel-copy">
              {projectReferenceItem?.id
                ? 'Add the student performance. The original MIDI from this project is used as the reference automatically.'
                : 'Add the student performance and choose the original MIDI once for this comparison.'}
            </p>

            {projectReferenceItem?.id && (
              <div className="pp-compare-reference-card">
                <span>Reference MIDI</span>
                <strong>{projectReferenceItem.title || 'Saved project reference'}</strong>
                <p>{projectReferenceItem.project || currentProject?.name || 'Current project'}</p>
              </div>
            )}

            <div className={`pp-compare-file-grid${projectReferenceItem?.id ? ' single' : ''}`}>
              <label className="pp-compare-file-picker">
                <span>Performance audio or MIDI</span>
                <input
                  type="file"
                  accept="audio/*,.mid,.midi,audio/midi,audio/x-midi"
                  onChange={handleComparisonPerformanceSelection}
                  disabled={isPreparingTutor}
                />
              </label>
              {!projectReferenceItem?.id && (
                <label className="pp-compare-file-picker">
                  <span>Original reference MIDI</span>
                  <input
                    type="file"
                    accept=".mid,.midi,audio/midi,audio/x-midi"
                    onChange={handleComparisonReferenceSelection}
                    disabled={isPreparingTutor}
                  />
                </label>
              )}
            </div>

            {(comparisonPerformanceFile || comparisonReferenceFile || projectReferenceItem?.id) && (
              <div className="pp-compare-file-list">
                {comparisonPerformanceFile && (
                  <p>Performance {comparisonPerformanceIsMidi ? 'MIDI' : 'audio'}: {comparisonPerformanceFile.name}</p>
                )}
                {projectReferenceItem?.id
                  ? <p>Reference MIDI: {projectReferenceItem.title || 'saved project reference'}</p>
                  : comparisonReferenceFile && <p>Reference MIDI: {comparisonReferenceFile.name}</p>}
              </div>
            )}

            <div className="pp-upload-steps pp-compare-steps">
              {comparisonPreparationSteps.map((step) => {
                const stepIndex = comparisonPreparationSteps.indexOf(step);
                const activeIndex = comparisonPreparationSteps.indexOf(preparePhase);
                const state = isPreparingTutor
                  ? activeIndex > stepIndex ? 'done' : activeIndex === stepIndex ? 'active' : 'pending'
                  : 'pending';
                return (
                  <div key={step} className={`pp-upload-step ${state}`}>
                    <span>{PREPARATION_STEP_LABELS[step]}</span>
                  </div>
                );
              })}
            </div>

            <div className="pp-compare-panel-actions">
              <button
                className="pp-compare-secondary-action"
                onClick={handleCloseComparisonPanel}
                disabled={isPreparingTutor}
                type="button"
              >
                Cancel
              </button>
              <button
                className="pp-upload-action pp-compare-submit"
                onClick={handlePrepareComparison}
                disabled={isPreparingTutor || !comparisonPerformanceFile || (!projectReferenceItem?.id && !comparisonReferenceFile)}
                type="button"
              >
                {comparisonPerformanceIsMidi ? 'Compare MIDI files' : 'Compare performance'}
              </button>
            </div>

            <p className="pp-compare-status">{comparisonStatus}</p>
            {comparisonError && <p className="pp-upload-error">{comparisonError}</p>}
          </section>
        </div>
      )}

      {shouldShowMidiStage && (
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
            isCompareOpen={comparisonPanelOpen}
            onPlay={player.play}
            onPause={player.pause}
            onStop={player.stop}
            onSeek={player.seek}
            onSpeedChange={player.setSpeed}
            onVolumeChange={player.setVolume}
            onMenuToggle={() => setMenuOpen(v => !v)}
            onTutorToggle={() => setChatOpen(v => !v)}
            onCompareOpen={handleOpenComparisonPanel}
          />
        </div>
      )}

      <div className={`pp-main${shouldShowMidiStage ? '' : ' pp-main-setup'}`} ref={mainRef}>
        {shouldShowMidiStage && (
          <div className={`pp-scene-badge${hasMidi ? ' active' : ''}`}>
            <span className="pp-scene-dot" />
            {hasMidi ? 'Tutor stage active' : 'Awaiting transcription'}
          </div>
        )}

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
                  <strong>Song audio or MIDI in, falling-note practice and piano guidance out.</strong>
                </div>
              </div>
              <div className="pp-upload-kicker">Learning Player</div>
              <h2>Open a song in the piano player</h2>
              <p>Upload MP3/WAV audio for transcription, or open a MIDI file directly. Compare performances later from the player toolbar.</p>
              <div className="pp-upload-highlights">
                <span>Falling notes</span>
                <span>Piano keys</span>
                <span>Tutor optional</span>
              </div>
              <label className="pp-upload-picker">
                <span>Song audio or MIDI</span>
                <input
                  type="file"
                  accept="audio/*,.mid,.midi,audio/midi,audio/x-midi"
                  onChange={handleAudioSelection}
                />
              </label>
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
                disabled={!selectedAudioFile}
              >
                Open player
              </button>
              <p className="pp-upload-status">{uploadStatus}</p>
              {selectedAudioFile && <p className="pp-upload-file">Song {performanceIsMidi ? 'MIDI' : 'audio'}: {selectedAudioFile.name}</p>}
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

      {shouldShowMidiStage && (
        <div className="pp-keyboard">
          <PianoKeyboard
            activeNotes={activeNotes}
            containerWidth={dimensions.width}
          />
        </div>
      )}

      {shouldShowMidiStage && (
        <div className="pp-progress-bar">
          <div
            className="pp-progress-fill"
            style={{ width: timelineDuration ? `${Math.min((visualTime / timelineDuration) * 100, 100)}%` : '0%' }}
          />
        </div>
      )}
    </div>
  );
}
