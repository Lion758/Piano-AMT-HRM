import {
  MIDI_MIN, MIDI_MAX, NOTE_NAMES, BLACK_KEY_INDICES,
  KEY_MAP, TOTAL_WHITE_KEYS, BLACK_KEY_WIDTH_RATIO,
  getBlackKeyOffset, MIDDLE_C
} from './constants.js';

/**
 * Convert MIDI note number to note name (e.g., 60 -> "C4")
 */
export function midiToNoteName(midi) {
  const noteIndex = midi % 12;
  const octave = Math.floor(midi / 12) - 1;
  return `${NOTE_NAMES[noteIndex]}${octave}`;
}

/**
 * Get short label for display on falling notes (e.g., "C#" without octave)
 */
export function midiToShortName(midi) {
  return NOTE_NAMES[midi % 12];
}

/**
 * Check if a MIDI note is a black key
 */
export function isBlackKey(midi) {
  return BLACK_KEY_INDICES.has(midi % 12);
}

/**
 * Get the x position and width of a MIDI note relative to the keyboard container.
 * Returns { x, width } as fractions of the total container width.
 */
export function getNotePosition(midi) {
  if (midi < MIDI_MIN || midi > MIDI_MAX) return null;

  const whiteKeyWidth = 1 / TOTAL_WHITE_KEYS;
  const info = KEY_MAP[midi];

  if (!info.isBlack) {
    return {
      x: info.whiteKeyIndex * whiteKeyWidth,
      width: whiteKeyWidth,
    };
  }

  // Black key: find the white key to the left
  const noteInOctave = midi % 12;
  // Find the nearest white key below this black key
  let leftWhiteMidi = midi - 1;
  while (leftWhiteMidi >= MIDI_MIN && KEY_MAP[leftWhiteMidi]?.isBlack) {
    leftWhiteMidi--;
  }

  if (leftWhiteMidi < MIDI_MIN) return null;

  const leftWhiteIndex = KEY_MAP[leftWhiteMidi].whiteKeyIndex;
  const offset = getBlackKeyOffset(noteInOctave);
  const blackWidth = whiteKeyWidth * BLACK_KEY_WIDTH_RATIO;

  return {
    x: leftWhiteIndex * whiteKeyWidth + (offset * whiteKeyWidth) - (blackWidth / 2),
    width: blackWidth,
  };
}

/**
 * Assign hand (left/right) to notes based on track or pitch split.
 * Returns the notes array with a `hand` property added.
 */
export function assignHands(tracks) {
  const allNotes = [];

  if (tracks.length >= 2) {
    // Multi-track: track 0 = right hand, track 1 = left hand
    tracks.forEach((track, trackIndex) => {
      track.notes.forEach(note => {
        allNotes.push({
          midi: note.midi,
          name: note.name,
          time: note.time,
          duration: note.duration,
          velocity: note.velocity,
          hand: trackIndex === 0 ? 'right' : 'left',
        });
      });
    });
  } else if (tracks.length === 1) {
    // Single track: split at middle C
    tracks[0].notes.forEach(note => {
      allNotes.push({
        midi: note.midi,
        name: note.name,
        time: note.time,
        duration: note.duration,
        velocity: note.velocity,
        hand: note.midi >= MIDDLE_C ? 'right' : 'left',
      });
    });
  }

  // Sort by time for efficient rendering
  allNotes.sort((a, b) => a.time - b.time);
  return allNotes;
}

/**
 * Binary search to find the index of the first note at or after `time`.
 */
export function findFirstNoteIndex(notes, time) {
  let lo = 0;
  let hi = notes.length;
  while (lo < hi) {
    const mid = (lo + hi) >>> 1;
    if (notes[mid].time + notes[mid].duration < time) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  return lo;
}

/**
 * Format seconds to MM:SS
 */
export function formatTime(seconds) {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}
