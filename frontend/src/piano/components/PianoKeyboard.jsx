import { useMemo } from 'react';
import {
  MIDI_MIN, MIDI_MAX, TOTAL_WHITE_KEYS,
  BLACK_KEY_WIDTH_RATIO, BLACK_KEY_HEIGHT_RATIO, COLORS
} from '../utils/constants.js';
import { midiToNoteName, isBlackKey, getNotePosition } from '../utils/noteHelpers.js';

export default function PianoKeyboard({ activeNotes = [], containerWidth = 1200 }) {
  const whiteKeyWidth = containerWidth / TOTAL_WHITE_KEYS;

  const keys = useMemo(() => {
    const result = [];
    for (let midi = MIDI_MIN; midi <= MIDI_MAX; midi++) {
      const pos = getNotePosition(midi);
      if (!pos) continue;
      result.push({
        midi,
        name: midiToNoteName(midi),
        black: isBlackKey(midi),
        x: pos.x * containerWidth,
        width: pos.width * containerWidth,
      });
    }
    return result;
  }, [containerWidth]);

  // Build a set of active midi notes with their hand assignment
  const activeMap = useMemo(() => {
    const map = {};
    activeNotes.forEach(n => { map[n.midi] = n.hand; });
    return map;
  }, [activeNotes]);

  const whiteKeys = keys.filter(k => !k.black);
  const blackKeys = keys.filter(k => k.black);

  const keyboardHeight = 140;
  const blackKeyHeight = keyboardHeight * BLACK_KEY_HEIGHT_RATIO;

  return (
    <div className="piano-keyboard" style={{ position: 'relative', width: containerWidth, height: keyboardHeight }}>
      {/* White keys */}
      {whiteKeys.map(key => {
        const hand = activeMap[key.midi];
        const isActive = !!hand;
        const activeColor = hand === 'left' ? COLORS.leftHand : COLORS.rightHand;

        return (
          <div
            key={key.midi}
            className="piano-key white-key"
            style={{
              position: 'absolute',
              left: key.x,
              top: 0,
              width: key.width - 1,
              height: keyboardHeight,
              background: isActive
                ? activeColor
                : 'linear-gradient(to bottom, #f8f8f8, #e8e8e8)',
              border: '1px solid #999',
              borderTop: '1px solid #bbb',
              borderRadius: '0 0 4px 4px',
              boxSizing: 'border-box',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: 'flex-end',
              alignItems: 'center',
              paddingBottom: 6,
              cursor: 'pointer',
              transition: 'background 0.08s',
              zIndex: 1,
            }}
          >
            {key.name.startsWith('C') && !key.name.includes('#') && (
              <span style={{
                fontSize: 10,
                color: isActive ? '#fff' : COLORS.keyLabel,
                fontWeight: 600,
                userSelect: 'none',
              }}>
                {key.name}
              </span>
            )}
          </div>
        );
      })}

      {/* Black keys */}
      {blackKeys.map(key => {
        const hand = activeMap[key.midi];
        const isActive = !!hand;
        const activeColor = hand === 'left' ? COLORS.leftHand : COLORS.rightHand;

        return (
          <div
            key={key.midi}
            className="piano-key black-key"
            style={{
              position: 'absolute',
              left: key.x,
              top: 0,
              width: key.width,
              height: blackKeyHeight,
              background: isActive
                ? activeColor
                : 'linear-gradient(to bottom, #2a2a3e, #111122)',
              border: '1px solid #000',
              borderRadius: '0 0 3px 3px',
              boxSizing: 'border-box',
              cursor: 'pointer',
              transition: 'background 0.08s',
              zIndex: 2,
            }}
          />
        );
      })}

      {/* Red line at top of keyboard */}
      <div style={{
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        height: 2,
        background: '#e74c3c',
        zIndex: 3,
      }} />
    </div>
  );
}
