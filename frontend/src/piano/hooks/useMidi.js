import { useState, useEffect } from 'react';
import { Midi } from '@tonejs/midi';
import { assignHands } from '../utils/noteHelpers.js';

const API_BASE = 'http://134.208.3.192:8000';

export function useMidi(url = `${API_BASE}/uploads/test.mid`) {
  const [notes, setNotes] = useState([]);
  const [duration, setDuration] = useState(0);
  const [tempo, setTempo] = useState(120);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [midiData, setMidiData] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function loadMidi() {
      setIsLoading(true);
      setError(null);
      try {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`Failed to load MIDI: ${response.status}`);
        const arrayBuffer = await response.arrayBuffer();
        const midi = new Midi(arrayBuffer);

        if (cancelled) return;

        // Extract tempo from the first tempo event
        const tempoVal = midi.header.tempos.length > 0
          ? Math.round(midi.header.tempos[0].bpm)
          : 120;

        // Get tracks that have notes
        const tracksWithNotes = midi.tracks.filter(t => t.notes.length > 0);
        const allNotes = assignHands(tracksWithNotes);

        // Calculate total duration
        const maxEnd = allNotes.reduce(
          (max, n) => Math.max(max, n.time + n.duration), 0
        );

        setMidiData(midi);
        setNotes(allNotes);
        setDuration(maxEnd);
        setTempo(tempoVal);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    loadMidi();
    return () => { cancelled = true; };
  }, [url]);

  return { notes, duration, tempo, isLoading, error, midiData };
}
