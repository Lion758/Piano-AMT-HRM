import { useState, useEffect } from 'react';
import { Midi } from '@tonejs/midi';
import { assignHands } from '../utils/noteHelpers.js';
import { resolveApiUrl } from '../../lib/api.js';

const MIDI_FETCH_RETRIES = 8;
const MIDI_FETCH_RETRY_DELAY_MS = 750;

export function useMidi(url = null) {
  const [notes, setNotes] = useState([]);
  const [duration, setDuration] = useState(0);
  const [tempo, setTempo] = useState(120);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [midiData, setMidiData] = useState(null);

  useEffect(() => {
    let cancelled = false;
    const midiUrl = typeof url === 'string' && url.trim() ? resolveApiUrl(url.trim()) : null;

    if (!midiUrl) {
      setNotes([]);
      setDuration(0);
      setTempo(120);
      setMidiData(null);
      setError(null);
      setIsLoading(false);
      return () => { cancelled = true; };
    }

    async function loadMidi() {
      setIsLoading(true);
      setError(null);
      try {
        let arrayBuffer = null;
        let lastError = null;

        for (let attempt = 0; attempt < MIDI_FETCH_RETRIES; attempt += 1) {
          try {
            const response = await fetch(midiUrl);
            if (!response.ok) {
              throw new Error(`Failed to load MIDI: ${response.status}`);
            }

            arrayBuffer = await response.arrayBuffer();
            break;
          } catch (err) {
            lastError = err;

            if (attempt === MIDI_FETCH_RETRIES - 1) {
              throw err;
            }

            await new Promise(resolve => window.setTimeout(resolve, MIDI_FETCH_RETRY_DELAY_MS));
            if (cancelled) return;
          }
        }

        if (!arrayBuffer) {
          throw lastError || new Error('Failed to load MIDI.');
        }

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
