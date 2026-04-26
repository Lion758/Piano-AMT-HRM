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
  ctx.save();
  let fontSize = Math.max(6, Math.min(12, Math.floor(Math.min(w / (label.length * 0.58), h >= 14 ? h * 0.38 : 8))));
  ctx.font = `bold ${fontSize}px "Avenir Next", "Segoe UI", sans-serif`;
  while (fontSize > 5 && ctx.measureText(label).width > w - 1) {
    fontSize -= 1;
    ctx.font = `bold ${fontSize}px "Avenir Next", "Segoe UI", sans-serif`;
  }

  const textW = ctx.measureText(label).width;
  const padX = Math.min(6, Math.max(2, w * 0.12));
  const badgeH = fontSize + (h >= 14 ? 5 : 3);
  const badgeW = Math.min(w - 2, Math.max(12, textW + padX * 2));
  const canDrawBadge = badgeW >= textW + 2;
  const badgeX = x + (w - badgeW) / 2;
  const centerY = h >= badgeH + 4
    ? y + h - Math.min(8, Math.max(1, h * 0.08)) - badgeH / 2
    : y + h * 0.64;
  const badgeY = centerY - badgeH / 2;

  ctx.globalAlpha = 0.92;
  if (canDrawBadge) {
    ctx.fillStyle = COLORS.noteLabelBg;
    drawRoundedRect(ctx, badgeX, badgeY, badgeW, badgeH, 4);
    ctx.fill();
  }

  ctx.fillStyle = '#ffffff';
  ctx.shadowColor = 'rgba(0, 0, 0, 0.45)';
  ctx.shadowBlur = canDrawBadge ? 0 : 3;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, x + w / 2, centerY);
  ctx.restore();
}

function drawSustainSpan(ctx, x, y, w, h) {
  const inset = 10;
  const rx = x + inset;
  const rw = Math.max(0, w - inset * 2);

  if (rw <= 0 || h <= 2) return;

  ctx.save();
  ctx.globalAlpha = 0.72;
  ctx.fillStyle = COLORS.sustainRegionFill;
  drawRoundedRect(ctx, rx, y, rw, h, 6);
  ctx.fill();

  ctx.strokeStyle = COLORS.sustainRegionStroke;
  ctx.lineWidth = 1.5;
  ctx.setLineDash([2, 6]);
  ctx.lineDashOffset = -y * 0.25;
  ctx.lineCap = 'round';
  drawRoundedRect(ctx, rx, y, rw, h, 6);
  ctx.stroke();
  ctx.restore();
}

export default function FallingNotesCanvas({
  notes = [],
  currentTime = 0,
  containerWidth = 1200,
  containerHeight = 500,
  pixelsPerSecond = DEFAULT_PPS,
  sustainSpans = [],
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

    for (const span of sustainSpans) {
      const onset = Number(span?.onset);
      const offset = Number(span?.offset);
      if (!Number.isFinite(onset) || !Number.isFinite(offset) || offset <= time || onset > time + (h / pixelsPerSecond)) {
        continue;
      }

      const top = h - (offset - time) * pixelsPerSecond;
      const bottom = h - (onset - time) * pixelsPerSecond;
      const rectTop = Math.max(0, top);
      const rectBottom = Math.min(h, bottom);
      const rectHeight = rectBottom - rectTop;

      if (rectHeight > 2) {
        drawSustainSpan(ctx, 0, rectTop, w, rectHeight);
      }
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

      if (rw >= 8) {
        drawNoteLabel(ctx, label, rx, ry, rw, rh);
      }
    }

    ctx.strokeStyle = COLORS.canvasGuide;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, h);
    ctx.lineTo(w, h);
    ctx.stroke();
  }, [notes, pixelsPerSecond, sustainSpans]);

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
