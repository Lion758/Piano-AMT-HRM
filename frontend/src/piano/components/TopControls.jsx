import { SPEED_OPTIONS } from '../utils/constants.js';
import { formatTime } from '../utils/noteHelpers.js';

export default function TopControls({
  isPlaying, currentTime, duration,
  speed, volume, isLoaded,
  onPlay, onPause, onStop, onSeek,
  onSpeedChange, onVolumeChange,
}) {
  return (
    <div className="top-controls">
      {/* Left: Transport buttons */}
      <div className="tc-group tc-transport">
        <button
          className="tc-btn"
          onClick={isPlaying ? onPause : onPlay}
          disabled={!isLoaded}
          title={isPlaying ? 'Pause' : 'Play'}
        >
          {isPlaying ? (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <rect x="6" y="4" width="4" height="16" rx="1" />
              <rect x="14" y="4" width="4" height="16" rx="1" />
            </svg>
          ) : (
            <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
              <polygon points="6,4 20,12 6,20" />
            </svg>
          )}
        </button>
        <button className="tc-btn" onClick={onStop} disabled={!isLoaded} title="Stop">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
            <rect x="6" y="6" width="12" height="12" rx="2" />
          </svg>
        </button>
      </div>

      {/* Center: Progress bar + time */}
      <div className="tc-group tc-progress">
        <span className="tc-time">{formatTime(currentTime)}</span>
        <input
          type="range"
          className="tc-slider progress-slider"
          min={0}
          max={duration || 1}
          step={0.1}
          value={currentTime}
          onChange={e => onSeek(parseFloat(e.target.value))}
          disabled={!isLoaded}
        />
        <span className="tc-time">{formatTime(duration)}</span>
      </div>

      {/* Right: Speed + Volume */}
      <div className="tc-group tc-settings">
        {/* Speed */}
        <div className="tc-control">
          <label className="tc-label">Speed</label>
          <select
            className="tc-select"
            value={speed}
            onChange={e => onSpeedChange(parseFloat(e.target.value))}
          >
            {SPEED_OPTIONS.map(s => (
              <option key={s} value={s}>{s}x</option>
            ))}
          </select>
        </div>

        {/* Volume */}
        <div className="tc-control">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor" className="tc-icon">
            <path d="M3 9v6h4l5 5V4L7 9H3zm13.5 3c0-1.77-1.02-3.29-2.5-4.03v8.05c1.48-.73 2.5-2.25 2.5-4.02z" />
          </svg>
          <input
            type="range"
            className="tc-slider volume-slider"
            min={0}
            max={1}
            step={0.01}
            value={volume}
            onChange={e => onVolumeChange(parseFloat(e.target.value))}
          />
        </div>

        {/* Loading indicator */}
        {!isLoaded && (
          <span className="tc-loading">Loading piano...</span>
        )}
      </div>
    </div>
  );
}
