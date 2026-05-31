import { SPEED_OPTIONS } from '../utils/constants.js';
import { formatTime } from '../utils/noteHelpers.js';

export default function TopControls({
  isPlaying,
  currentTime,
  duration,
  speed,
  volume,
  isLoaded,
  isMenuOpen,
  isTutorOpen,
  isCompareOpen,
  isRecording = false,
  isCountingDown = false,
  recordDisabled = false,
  onPlay,
  onPause,
  onStop,
  onRecord,
  onStopRecording,
  onSeek,
  onSpeedChange,
  onVolumeChange,
  loopStart = 0,
  loopEnd = 0,
  isLooping = false,
  hasLoopRange = false,
  onLoopStart,
  onLoopEnd,
  onLoopToggle,
  onLoopClear,
  onMenuToggle,
  onTutorToggle,
  onCompareOpen,
}) {
  const safeDuration = duration || 0;
  const loopStartPercent = safeDuration > 0
    ? Math.max(0, Math.min(100, (loopStart / safeDuration) * 100))
    : 0;
  const loopEndPercent = safeDuration > 0
    ? Math.max(0, Math.min(100, (loopEnd / safeDuration) * 100))
    : 0;
  const loopRangeLeft = Math.min(loopStartPercent, loopEndPercent);
  const loopRangeWidth = Math.max(0, loopEndPercent - loopStartPercent);
  const recordingActive = isRecording || isCountingDown;

  return (
    <div className="top-controls">
      <div className="tc-row tc-row-primary">
        <div className="tc-group tc-nav">
          <button
            className={`tc-rail-btn${isMenuOpen ? ' active' : ''}`}
            onClick={onMenuToggle}
            type="button"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <rect x="3" y="5" width="18" height="2" rx="1" />
              <rect x="3" y="11" width="18" height="2" rx="1" />
              <rect x="3" y="17" width="18" height="2" rx="1" />
            </svg>
            <span>Menu</span>
          </button>
          <button
            className={`tc-rail-btn${isTutorOpen ? ' active' : ''}`}
            onClick={onTutorToggle}
            type="button"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2zm0 14H6l-2 2V4h16v12z" />
            </svg>
            <span>Tutor</span>
          </button>
          <button
            className={`tc-rail-btn${isCompareOpen ? ' active' : ''}`}
            onClick={onCompareOpen}
            type="button"
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M7 7h10l-3.2-3.2L15 2.6 20.4 8 15 13.4l-1.2-1.2L17 9H7V7zm10 10H7l3.2 3.2L9 21.4 3.6 16 9 10.6l1.2 1.2L7 15h10v2z" />
            </svg>
            <span>Compare</span>
          </button>
        </div>

        <div className="tc-group tc-core">
          <button
            className="tc-play-btn"
            onClick={isPlaying ? onPause : onPlay}
            disabled={!isLoaded}
            title={isPlaying ? 'Pause (Space)' : 'Play (Space)'}
            type="button"
          >
            {isPlaying ? (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <rect x="6" y="4" width="4" height="16" rx="1" />
                <rect x="14" y="4" width="4" height="16" rx="1" />
              </svg>
            ) : (
              <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                <polygon points="6,4 20,12 6,20" />
              </svg>
            )}
            <span>{isPlaying ? 'Pause' : 'Play'}</span>
          </button>

          <button
            className={`tc-btn tc-record-btn${recordingActive ? ' active' : ''}`}
            onClick={onRecord}
            disabled={recordDisabled || recordingActive}
            title={isRecording ? 'Recording in progress' : isCountingDown ? 'Countdown in progress' : 'Record your take'}
            type="button"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <circle cx="12" cy="12" r="6" />
            </svg>
          </button>

          <button
            className={`tc-btn tc-stop-btn${recordingActive ? ' recording-active' : ''}`}
            onClick={recordingActive ? onStopRecording : onStop}
            disabled={!recordingActive && !isLoaded}
            title={isCountingDown ? 'Cancel recording countdown' : isRecording ? 'Stop recording' : 'Return to start'}
            type="button"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <rect x="6" y="6" width="12" height="12" rx="2" />
            </svg>
          </button>
        </div>

        <div className="tc-group tc-settings">
          <label className="tc-control tc-control-speed">
            <span className="tc-label">Speed</span>
            <select
              className="tc-select"
              value={speed}
              onChange={(e) => onSpeedChange(parseFloat(e.target.value))}
            >
              {SPEED_OPTIONS.map((option) => (
                <option key={option} value={option}>
                  {option}x
                </option>
              ))}
            </select>
          </label>

          <label className="tc-control tc-control-volume">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" className="tc-icon" aria-hidden="true">
              <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z" />
            </svg>
            <input
              type="range"
              className="tc-slider volume-slider"
              min={0}
              max={1}
              step={0.01}
              value={volume}
              onChange={(e) => onVolumeChange(parseFloat(e.target.value))}
            />
          </label>
        </div>
      </div>

      <div className="tc-row tc-row-secondary">
        <span className="tc-time">{formatTime(currentTime)}</span>
        <div className="tc-progress-shell">
          <span className="tc-progress-label">Timeline</span>
          <div className="tc-progress-track">
            {hasLoopRange && (
              <span
                className="tc-loop-range"
                style={{ left: `${loopRangeLeft}%`, width: `${loopRangeWidth}%` }}
                aria-hidden="true"
              />
            )}
            {(loopStart > 0 || hasLoopRange) && (
              <span
                className="tc-loop-marker tc-loop-marker-start"
                style={{ left: `${loopStartPercent}%` }}
                aria-hidden="true"
              >
                A
              </span>
            )}
            {hasLoopRange && (
              <span
                className="tc-loop-marker tc-loop-marker-end"
                style={{ left: `${loopEndPercent}%` }}
                aria-hidden="true"
              >
                B
              </span>
            )}
            <input
              type="range"
              className="tc-slider progress-slider"
              min={0}
              max={duration || 1}
              step={0.1}
              value={currentTime}
              onChange={(e) => onSeek(parseFloat(e.target.value))}
              disabled={!isLoaded}
            />
          </div>
        </div>
        <span className="tc-time">{formatTime(duration)}</span>
      </div>

      <div className="tc-row tc-row-loop">
        <div className="tc-loop-readout">
          <span>Loop A-B</span>
          <strong>
            {hasLoopRange
              ? `${formatTime(loopStart)} - ${formatTime(loopEnd)}`
              : 'Set markers'}
          </strong>
        </div>
        <div className="tc-loop-controls" role="group" aria-label="Loop controls">
          <button
            className="tc-marker-btn"
            onClick={() => onLoopStart?.(currentTime)}
            disabled={!isLoaded}
            type="button"
          >
            Set A
          </button>
          <button
            className="tc-marker-btn"
            onClick={() => onLoopEnd?.(currentTime)}
            disabled={!isLoaded}
            type="button"
          >
            Set B
          </button>
          <button
            className={`tc-marker-btn tc-loop-toggle${isLooping ? ' active' : ''}`}
            onClick={onLoopToggle}
            disabled={!isLoaded || !hasLoopRange}
            type="button"
          >
            {isLooping ? 'Loop On' : 'Loop Off'}
          </button>
          <button
            className="tc-marker-btn tc-loop-clear"
            onClick={onLoopClear}
            disabled={!isLoaded || (!hasLoopRange && loopStart === 0 && loopEnd === 0)}
            type="button"
          >
            Clear
          </button>
        </div>
      </div>

    </div>
  );
}
