// Piano layout constants

// MIDI range: A0 (21) to C8 (108)
export const MIDI_MIN = 21;
export const MIDI_MAX = 108;
export const TOTAL_KEYS = MIDI_MAX - MIDI_MIN + 1; // 88

// Note names in an octave
export const NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

// Which notes in an octave are black keys (semitone indices)
export const BLACK_KEY_INDICES = new Set([1, 3, 6, 8, 10]); // C#, D#, F#, G#, A#

// White keys per octave
export const WHITE_KEYS_PER_OCTAVE = 7;

// Total white keys on 88-key piano (A0 to C8)
export const TOTAL_WHITE_KEYS = 52;

// Black key width ratio relative to white key
export const BLACK_KEY_WIDTH_RATIO = 0.6;
export const BLACK_KEY_HEIGHT_RATIO = 0.62;

// Colors
export const COLORS = {
  leftHand: '#9f7a50',
  rightHand: '#dcc099',
  activeGlow: 'rgba(236, 211, 164, 0.32)',
  whiteKey: '#f3ede3',
  whiteKeyShadow: '#d9c8b1',
  blackKey: '#1a1410',
  blackKeyShadow: '#4f3926',
  keyLabel: '#6d5744',
  keyLabelBlack: '#d9c8b4',
  canvasBg: '#100c09',
  canvasGrid: 'rgba(243, 232, 213, 0.05)',
  canvasBlackLane: 'rgba(16, 12, 9, 0.2)',
  canvasGuide: 'rgba(214, 178, 123, 0.64)',
  sustainGlow: 'rgba(223, 194, 140, 0.24)',
  noteStroke: 'rgba(255, 246, 231, 0.16)',
  noteLabelBg: 'rgba(28, 20, 14, 0.6)',
};

// Speed options
export const SPEED_OPTIONS = [0.25, 0.5, 0.75, 1, 1.25, 1.5, 2];

// Pixels per second for falling notes
export const DEFAULT_PPS = 200;

// Middle C for hand splitting
export const MIDDLE_C = 60;

// Pre-compute white key index for each MIDI note
// Returns -1 for black keys
function buildWhiteKeyMap() {
  const map = {};
  let whiteIndex = 0;
  for (let midi = MIDI_MIN; midi <= MIDI_MAX; midi++) {
    const noteInOctave = midi % 12;
    if (BLACK_KEY_INDICES.has(noteInOctave)) {
      map[midi] = { isBlack: true, whiteKeyIndex: -1 };
    } else {
      map[midi] = { isBlack: false, whiteKeyIndex: whiteIndex };
      whiteIndex++;
    }
  }
  return map;
}

export const KEY_MAP = buildWhiteKeyMap();

// Black key offsets relative to the left edge of their neighboring white key
// These are fractional positions within the white key width
const BLACK_KEY_OFFSETS = {
  1: 0.6,   // C# — right side of C
  3: 0.8,   // D# — right side of D
  6: 0.6,   // F# — right side of F
  8: 0.7,   // G# — right side of G
  10: 0.8,  // A# — right side of A
};

export function getBlackKeyOffset(noteInOctave) {
  return BLACK_KEY_OFFSETS[noteInOctave] || 0.65;
}
