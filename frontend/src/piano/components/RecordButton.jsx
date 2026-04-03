export default function RecordButton({ isRecording, onStart, onStop, audioURL, error }) {
  return (
    <div className="record-section">
      <button
        className={`tc-btn record-btn ${isRecording ? 'recording' : ''}`}
        onClick={isRecording ? onStop : onStart}
        title={isRecording ? 'Stop recording' : 'Record from microphone'}
      >
        <span className={`record-dot ${isRecording ? 'pulse' : ''}`} />
        <span className="record-label">{isRecording ? 'Stop' : 'Rec'}</span>
      </button>

      {audioURL && !isRecording && (
        <audio controls src={audioURL} className="record-playback" />
      )}

      {error && <span className="record-error">{error}</span>}
    </div>
  );
}
