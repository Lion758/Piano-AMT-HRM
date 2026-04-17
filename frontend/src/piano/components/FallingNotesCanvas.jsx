import { useRef, useEffect, useCallback } from 'react';
import { COLORS, DEFAULT_PPS, TOTAL_WHITE_KEYS } from '../utils/constants.js';
import { getNotePosition, midiToShortName, findFirstNoteIndex, isBlackKey } from '../utils/noteHelpers.js';

function drawRoundedRect(ctx, x, y, w, h, r = 4) {
  const rr = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.lineTo(x + w - rr, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + rr);
  ctx.lineTo(x + w, y + h - rr);
  ctx.quadraticCurveTo(x + w, y + h, x + w - rr, y + h);
  ctx.lineTo(x + rr, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - rr);
  ctx.lineTo(x, y + rr);
  ctx.quadraticCurveTo(x, y, x + rr, y);
  ctx.closePath();
}

function drawNoteLabel(ctx, label, x, y, w, h) {
  const fontSize = Math.max(8, Math.min(12, Math.floor(Math.min(h * 0.42, w * 0.55))));
  const padX = Math.min(6, w * 0.12);
  const badgeH = Math.max(12, fontSize + 4);
  const badgeW = Math.max(16, Math.min(w - 2, label.length * (fontSize * 0.62) + padX * 2));
  const badgeX = x + (w - badgeW) / 2;
  const badgeY = h >= badgeH + 6 ? y + 3 : y + Math.max(1, (h - badgeH) / 2);

  ctx.save();
  ctx.globalAlpha = 0.92;
  ctx.fillStyle = COLORS.noteLabelBg;
  drawRoundedRect(ctx, badgeX, badgeY, badgeW, Math.min(badgeH, h - 1), 4);
  ctx.fill();

  ctx.fillStyle = '#ffffff';
  ctx.font = `bold ${fontSize}px "Avenir Next", "Segoe UI", sans-serif`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, x + w / 2, badgeY + Math.min(badgeH, h - 1) / 2);
  ctx.restore();
}

export default function FallingNotesCanvas({
  notes = [],
  currentTime = 0,
  containerWidth = 1200,
  containerHeight = 500,
  pixelsPerSecond = DEFAULT_PPS,
  sustainEvents = [],
}) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const timeRef = useRef(currentTime);

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

    ctx.clearRect(0, 0, w, h);

    const bgGradient = ctx.createLinearGradient(0, 0, 0, h);
    bgGradient.addColorStop(0, COLORS.canvasBg);
    bgGradient.addColorStop(0.58, '#1d1611');
    bgGradient.addColorStop(1, '#322419');
    ctx.fillStyle = bgGradient;
    ctx.fillRect(0, 0, w, h);

    const spotlight = ctx.createRadialGradient(w * 0.55, h * 0.16, 0, w * 0.55, h * 0.16, h * 0.72);
    spotlight.addColorStop(0, 'rgba(236, 211, 164, 0.08)');
    spotlight.addColorStop(1, 'rgba(236, 211, 164, 0)');
    ctx.fillStyle = spotlight;
    ctx.fillRect(0, 0, w, h);

    const whiteKeyWidth = w / TOTAL_WHITE_KEYS;
    ctx.strokeStyle = COLORS.canvasGrid;
    ctx.lineWidth = 1;
    for (let i = 0; i <= TOTAL_WHITE_KEYS; i++) {
      const x = i * whiteKeyWidth;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, h);
      ctx.stroke();
    }

    ctx.fillStyle = COLORS.canvasBlackLane;
    for (let midi = 21; midi <= 108; midi++) {
      if (!isBlackKey(midi)) continue;
      const pos = getNotePosition(midi);
      if (!pos) continue;
      ctx.fillRect(pos.x * w, 0, pos.width * w, h);
    }

    let sustainActive = false;

    if (sustainEvents.length > 0) {
      let lastSustain = null;
      for (const ev of sustainEvents) {
        if (ev.time <= time) lastSustain = ev;
        else break;
      }
      sustainActive = lastSustain ? lastSustain.value >= 64 : false;
    } else {
      const simultaneousActive = notes.filter(
        (n) => time >= n.time && time < n.time + n.duration
      );
      sustainActive = simultaneousActive.length >= 3;
    }

    if (sustainActive) {
      const bandH = 20;
      const grad = ctx.createLinearGradient(0, h - bandH, 0, h);
      grad.addColorStop(0, 'rgba(223, 194, 140, 0)');
      grad.addColorStop(1, COLORS.sustainGlow);
      ctx.fillStyle = grad;
      ctx.fillRect(0, h - bandH, w, bandH);

      ctx.strokeStyle = 'rgba(223, 194, 140, 0.55)';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(0, h - 1);
      ctx.lineTo(w, h - 1);
      ctx.stroke();

      ctx.save();
      ctx.fillStyle = 'rgba(246, 230, 198, 0.92)';
      ctx.font = 'bold 10px "Avenir Next", "Segoe UI", sans-serif';
      ctx.textAlign = 'left';
      ctx.textBaseline = 'bottom';
      ctx.fillText('SUS', 6, h - 3);
      ctx.restore();
    }

    const lookAheadSec = h / pixelsPerSecond;
    const startIdx = findFirstNoteIndex(notes, time);

    for (let i = startIdx; i < notes.length; i++) {
      const note = notes[i];
      if (note.time > time + lookAheadSec) break;

      const noteEnd = note.time + note.duration;
      if (noteEnd < time) continue;

      const pos = getNotePosition(note.midi);
      if (!pos) continue;

      const noteY = h - (note.time - time) * pixelsPerSecond;
      const noteHeight = Math.max(note.duration * pixelsPerSecond, 6);
      const noteX = pos.x * w;
      const noteW = Math.max(pos.width * w, 8);

      const rectTop = noteY - noteHeight;
      const rectBottom = noteY;
      const isActive = time >= note.time && time < noteEnd;
      const color = note.hand === 'left' ? COLORS.leftHand : COLORS.rightHand;

      ctx.save();

      if (isActive) {
        ctx.shadowColor = COLORS.activeGlow;
        ctx.shadowBlur = 15;
      }

      const rx = noteX + 1;
      const ry = rectTop;
      const rw = Math.max(noteW - 2, 6);
      const rh = Math.max(rectBottom - rectTop, 6);

      ctx.fillStyle = color;
      ctx.globalAlpha = isActive ? 1 : 0.88;
      drawRoundedRect(ctx, rx, ry, rw, rh, 3);
      ctx.fill();

      ctx.strokeStyle = COLORS.noteStroke;
      ctx.lineWidth = 0.5;
      ctx.stroke();

      ctx.restore();

      // Render labels much more often than before.
      const label = midiToShortName(note.midi);

      if (rh >= 11 && rw >= 10) {
        drawNoteLabel(ctx, label, rx, ry, rw, rh);
      } else if (isActive && rw >= 8) {
        // Fallback: tiny active note gets a micro label above center.
        ctx.save();
        ctx.fillStyle = '#fff';
        ctx.font = 'bold 8px "Avenir Next", "Segoe UI", sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.globalAlpha = 0.95;
        ctx.fillText(label, rx + rw / 2, ry + rh / 2);
        ctx.restore();
      }
    }

    ctx.strokeStyle = COLORS.canvasGuide;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h);
    ctx.lineTo(w, h);
    ctx.stroke();
  }, [notes, pixelsPerSecond, sustainEvents]);

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
        background: COLORS.canvasBg,
      }}
    />
  );
}
