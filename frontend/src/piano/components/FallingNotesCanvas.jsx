import { useRef, useEffect, useCallback } from 'react';
import { COLORS, DEFAULT_PPS, TOTAL_WHITE_KEYS } from '../utils/constants.js';
import { getNotePosition, midiToShortName, findFirstNoteIndex, isBlackKey } from '../utils/noteHelpers.js';

export default function FallingNotesCanvas({
  notes = [],
  currentTime = 0,
  containerWidth = 1200,
  containerHeight = 500,
  pixelsPerSecond = DEFAULT_PPS,
  isPlaying = false,
}) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const timeRef = useRef(currentTime);

  // Keep timeRef in sync
  useEffect(() => {
    timeRef.current = currentTime;
  }, [currentTime]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    const time = timeRef.current;

    // Clear
    ctx.clearRect(0, 0, w, h);

    // Dark background with subtle grid lines at each octave
    ctx.fillStyle = '#0a0a14';
    ctx.fillRect(0, 0, w, h);

    // Draw vertical key separator lines
    const whiteKeyWidth = w / TOTAL_WHITE_KEYS;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= TOTAL_WHITE_KEYS; i++) {
      const x = i * whiteKeyWidth;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    // Shade black key columns slightly darker
    ctx.fillStyle = 'rgba(0, 0, 0, 0.15)';
    for (let midi = 21; midi <= 108; midi++) {
      if (!isBlackKey(midi)) continue;
      const pos = getNotePosition(midi);
      if (!pos) continue;
      ctx.fillRect(pos.x * w, 0, pos.width * w, h);
    }

    // Look-ahead window in seconds
    const lookAheadSec = h / pixelsPerSecond;

    // Find first relevant note
    const startIdx = findFirstNoteIndex(notes, time);

    // Draw notes
    for (let i = startIdx; i < notes.length; i++) {
      const note = notes[i];

      // If note starts beyond the visible window, stop
      if (note.time > time + lookAheadSec) break;

      const noteEnd = note.time + note.duration;

      // Skip notes that have already passed
      if (noteEnd < time) continue;

      const pos = getNotePosition(note.midi);
      if (!pos) continue;

      // Y position: bottom of canvas = currentTime, top = currentTime + lookAheadSec
      // A note at currentTime should be at the bottom (y = h)
      // A note at currentTime + lookAheadSec should be at the top (y = 0)
      const noteY = h - (note.time - time) * pixelsPerSecond;
      const noteHeight = note.duration * pixelsPerSecond;
      const noteX = pos.x * w;
      const noteW = pos.width * w;

      // Top of the rectangle (notes fall down, so top is further in the future)
      const rectTop = noteY - noteHeight;
      const rectBottom = noteY;

      // Determine if this note is currently being played
      const isActive = time >= note.time && time < noteEnd;

      // Colors
      const color = note.hand === 'left' ? COLORS.leftHand : COLORS.rightHand;

      // Draw the note block
      ctx.save();

      if (isActive) {
        ctx.shadowColor = color;
        ctx.shadowBlur = 15;
      }

      // Rounded rectangle
      const radius = 3;
      const rx = noteX + 1;
      const ry = rectTop;
      const rw = noteW - 2;
      const rh = Math.max(rectBottom - rectTop, 4);

      ctx.fillStyle = color;
      ctx.globalAlpha = isActive ? 1 : 0.85;
      ctx.beginPath();
      ctx.moveTo(rx + radius, ry);
      ctx.lineTo(rx + rw - radius, ry);
      ctx.quadraticCurveTo(rx + rw, ry, rx + rw, ry + radius);
      ctx.lineTo(rx + rw, ry + rh - radius);
      ctx.quadraticCurveTo(rx + rw, ry + rh, rx + rw - radius, ry + rh);
      ctx.lineTo(rx + radius, ry + rh);
      ctx.quadraticCurveTo(rx, ry + rh, rx, ry + rh - radius);
      ctx.lineTo(rx, ry + radius);
      ctx.quadraticCurveTo(rx, ry, rx + radius, ry);
      ctx.closePath();
      ctx.fill();

      // Subtle border
      ctx.strokeStyle = 'rgba(255,255,255,0.15)';
      ctx.lineWidth = 0.5;
      ctx.stroke();

      ctx.restore();

      // Note label (only if block is tall enough)
      if (rh > 18 && rw > 14) {
        ctx.save();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 10px Inter, Arial, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.globalAlpha = 0.9;
        const label = midiToShortName(note.midi);
        // Place label near the bottom of the note block
        const labelY = Math.min(rectBottom - 12, (rectTop + rectBottom) / 2);
        ctx.fillText(label, rx + rw / 2, labelY);
        ctx.restore();
      }
    }

    // Draw a horizontal "now" line at the bottom
    ctx.strokeStyle = 'rgba(231, 76, 60, 0.6)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h);
    ctx.lineTo(w, h);
    ctx.stroke();
  }, [notes, pixelsPerSecond]);

  // Animation loop
  useEffect(() => {
    let running = true;

    function loop() {
      if (!running) return;
      draw();
      animRef.current = requestAnimationFrame(loop);
    }

    loop();
    return () => {
      running = false;
      if (animRef.current) cancelAnimationFrame(animRef.current);
    };
  }, [draw]);

  return (
    <canvas
      ref={canvasRef}
      width={containerWidth}
      height={containerHeight}
      style={{
        display: 'block',
        width: '100%',
        height: '100%',
        background: '#0a0a14',
      }}
    />
  );
}
