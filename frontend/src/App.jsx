import { useState } from "react";
import "./index.css";

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

  const handleTranscribe = async () => {
    if (!selectedFile) {
      setError("Please choose an audio file first.");
      return;
    }

    setStatus("Uploading and separating audio...");
    setError("");
    setStems(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch("http://134.208.3.192:8000/transcribe", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Backend request failed.");
      }

      const data = await response.json();
      setStems(data.stems);
      setStatus("Separation completed successfully.");
    } catch (err) {
      setError(err.message);
      setStatus("Something went wrong.");
    }
  };

  return (
    <div className="page">
      <div className="card">
        <h1>Piano AMT</h1>
        <p>Upload audio and separate it into stems.</p>

        <input type="file" accept="audio/*" onChange={handleFileChange} />
        <br />
        <br />

        <button className="transcribe-btn" onClick={handleTranscribe}>
          Transcribe
        </button>

        <p style={{ marginTop: "1rem" }}>{status}</p>

        {error && <p style={{ color: "red" }}>{error}</p>}

        {stems && (
          <div style={{ marginTop: "1.5rem", textAlign: "left" }}>
            <h2>Separated Stems</h2>

            {Object.entries(stems).map(([name, path]) => {
              const audioUrl = `http://134.208.3.192:8000/${path}`;
              return (
                <div key={name} style={{ marginBottom: "1rem" }}>
                  <strong>{name}</strong>
                  <br />
                  <audio controls src={audioUrl}></audio>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;