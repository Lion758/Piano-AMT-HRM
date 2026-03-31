import { useState } from "react";
import "./index.css";
import pianoBanner from "./assets/piano-banner.png";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [status, setStatus] = useState("No file selected.");
  const [stems, setStems] = useState(null);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
    setStems(null);
    setError("");

    if (file) {
      setStatus(`Selected file: ${file.name}`);
    } else {
      setStatus("No file selected.");
    }
  };

  const handleSeparateStems = async () => {
    if (!selectedFile) {
      setError("Please choose an audio file first.");
      return;
    }

    setError("");
    setStems(null);
    setStatus("Uploading audio and separating stems...");

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("http://134.208.3.192:8000/transcribe", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Failed to separate stems.");
      }

      const data = await response.json();
      setStems(data.stems);
      setStatus("Stem separation completed successfully.");
    } catch (err) {
      setError(err.message);
      setStatus("Something went wrong.");
    }
  };

  return (
    <div className="app-shell">
      <div className="hero-section">
        <img
          src={pianoBanner}
          alt="Piano keys with sheet music"
          className="hero-image"
        />

        <h1 className="main-title">Piano Automatic Transcription</h1>

        <p className="subtitle">
          Separate stems, transcribe piano audio into MIDI, and access tutoring
          feedback through one interface.
        </p>

        <div className="upload-area">
          <label className="file-btn">
            Choose Audio
            <input type="file" accept="audio/*" onChange={handleFileChange} />
          </label>
        </div>

        <p className="status-text">{status}</p>
        {selectedFile && (
          <p className="file-name">Current file: {selectedFile.name}</p>
        )}
        {error && <p className="error-text">{error}</p>}

        <div className="button-group">
          <button className="main-btn" onClick={handleSeparateStems}>
            Separate Stems
          </button>
          <button className="main-btn">Transcribe</button>
          <button className="main-btn">Tutor</button>
        </div>
      </div>

      {stems && (
        <div className="stems-section">
          <h2>Separated Stems</h2>
          <div className="stems-grid">
            {Object.entries(stems).map(([name, path]) => {
              const audioUrl = `http://134.208.3.192:8000/${path}`;
              return (
                <div key={name} className="stem-card">
                  <h3>{name}</h3>
                  <audio controls src={audioUrl}></audio>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;