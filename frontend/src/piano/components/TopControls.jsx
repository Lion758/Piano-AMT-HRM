import RecordButton from './RecordButton.jsx';
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
  recordProps,
  onPlay,
  onPause,
  onStop,
  onSeek,
  onSpeedChange,
  onVolumeChange,
  onMenuToggle,
  onTutorToggle,
}) {
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
            className="tc-btn tc-stop-btn"
            onClick={onStop}
            disabled={!isLoaded}
            title="Return to start"
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

        <RecordButton {...recordProps} />
      </div>

      <div className="tc-row tc-row-secondary">
        <span className="tc-time">{formatTime(currentTime)}</span>
        <div className="tc-progress-shell">
          <span className="tc-progress-label">Timeline</span>
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
        <span className="tc-time">{formatTime(duration)}</span>
      </div>

      {!isLoaded && (
        <div className="tc-row tc-row-status">
          <span className="tc-loading">Loading piano...</span>
        </div>
      )}
    </div>
  );
}
