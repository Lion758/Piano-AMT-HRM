import { useState } from "react";
import "./index.css";
import pianoBanner from "./assets/piano-banner.png";

const API_BASE = "http://134.208.3.192:8000";

const STEPS = [
  {
    icon: "🎵",
    num: 1,
    title: "Upload Your Audio",
    desc: "Upload any piano recording — solo or mixed. Our system accepts MP3, WAV, M4A and more.",
  },
  {
    icon: "🎛️",
    num: 2,
    title: "Stem Separation",
    desc: "We isolate the piano track from the rest of the audio, giving you a clean signal to work with.",
  },
  {
    icon: "🎹",
    num: 3,
    title: "MIDI Transcription",
    desc: "The clean piano stem is automatically converted into a MIDI file — every note, every timing.",
  },
  {
    icon: "🧑‍🏫",
    num: 4,
    title: "Interactive Tutor",
    desc: "Our AI tutor analyses your MIDI and guides you through the piece — note by note, at your pace.",
  },
];

const FEATURES = [
  {
    icon: "🎚️",
    title: "Stem Separation",
    desc: "Separate a full mix into individual tracks. We extract the piano stem so transcription is clean and accurate.",
  },
  {
    icon: "🎼",
    title: "Automatic MIDI Generation",
    desc: "The isolated piano audio is converted to MIDI automatically — no manual input, no music theory required.",
  },
  {
    icon: "🧑‍🏫",
    title: "Interactive AI Tutor",
    desc: "Your personal practice guide. The tutor listens, compares, and gives real-time feedback on your playing.",
  },
  {
    icon: "📈",
    title: "Progress Tracking",
    desc: "See how you improve over time. Track accuracy, timing, and consistency across sessions.",
  },
  {
    icon: "🔁",
    title: "Loop & Slow Down",
    desc: "Struggling with a passage? Loop any section and slow it down without changing the pitch.",
  },
  {
    icon: "📱",
    title: "Works in Your Browser",
    desc: "No installation needed. Everything runs in the browser — upload, transcribe, and practise immediately.",
  },
];

const PIPELINE_STATS = [
  { label: "Pipeline Step 1", value: "Stem Split" },
  { label: "Pipeline Step 2", value: "MIDI Output" },
  { label: "Pipeline Step 3", value: "AI Tutor" },
  { label: "Focus", value: "Piano" },
];

const FOOTER_COLS = [
  { heading: "Tool", links: ["How It Works", "Features", "Try It Now"] },
  { heading: "Resources", links: ["Documentation", "API Reference", "Changelog"] },
  { heading: "Project", links: ["About", "Contact", "GitHub"] },
];

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [status, setStatus] = useState("Record, upload, or choose audio to begin.");
  const [stems, setStems] = useState(null);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    setSelectedFile(file);
    setStems(null);
    setError("");
    setStatus(file ? `Selected: ${file.name}` : "Record, upload, or choose audio to begin.");
  };

  const handleSeparateStems = async () => {
    if (!selectedFile) { setError("Please choose an audio file first."); return; }
    setError(""); setStems(null); setStatus("Separating stems…");
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const response = await fetch(`${API_BASE}/transcribe`, { method: "POST", body: formData });
      if (!response.ok) throw new Error("Failed to separate stems.");
      const data = await response.json();
      setStems(data.stems || {});
      setStatus("Stem separation completed.");
    } catch (err) {
      setError(err.message || "Something went wrong.");
      setStatus("Request failed.");
    }
  };

  const scrollToDemo = () =>
    document.getElementById("demo")?.scrollIntoView({ behavior: "smooth" });

  return (
    <>
      {/* Topbar */}
      <div className="topbar">
        🎹 Upload a piano recording and let AI transcribe it — then practise with your personal tutor.
      </div>

      <div className="page">

        {/* ── Navbar ── */}
        <header className="navbar">
          <div className="logo">
            <span className="logo-icon">♪</span>
            Piano Automatic Transcription
          </div>
          <nav className="navlinks">
            <a href="#workflow">How it works</a>
            <a href="#features">Features</a>
            <a href="#demo">Try it</a>
            <a href="#results">Results</a>
          </nav>
          <div className="navactions">
            <button className="nav-btn">Login</button>
            <button className="nav-btn solid">Get Started</button>
          </div>
        </header>

        {/* ── Hero ── */}
        <section className="hero">
          <div className="hero-copy">
            <div className="hero-badge">
              <span className="badge-dot" />
              Stem Separation · MIDI · AI Tutor
            </div>
            <h1>Piano Automatic<br /><em>Transcription</em></h1>
            <h2>
              Upload a piano recording — we split the stems, generate MIDI,
              and guide you through every note with an interactive AI tutor.
            </h2>

            <div className="cta-row">
              <button className="primary-btn" onClick={scrollToDemo}>
                🎹 Try It Now
              </button>
              <a href="#workflow" className="ghost-btn">See How It Works</a>
            </div>
          </div>

          <div className="hero-visual">
            <img src={pianoBanner} alt="Piano keys with score" className="hero-image" />
          </div>
        </section>

        {/* ── Stats Strip ── */}
        <section className="stats-strip">
          {PIPELINE_STATS.map((s) => (
            <div key={s.label} className="stat-box">
              <span className="stat-label">{s.label}</span>
              <strong>{s.value}</strong>
            </div>
          ))}
        </section>

        {/* ── Workflow ── */}
        <section className="workflow" id="workflow">
          <div className="section-head">
            <div className="section-tag">How It Works</div>
            <h3>From Audio to Interactive Lesson in 4 Steps</h3>
            <p>Upload once — we handle the transcription and build your practice session automatically.</p>
          </div>
          <div className="workflow-grid">
            {STEPS.map((s) => (
              <div key={s.num} className="workflow-card">
                <div className="step-icon-lg">{s.icon}</div>
                <div className="step">{s.num}</div>
                <h4>{s.title}</h4>
                <p>{s.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ── Features ── */}
        <section className="features" id="features">
          <div className="section-head">
            <div className="section-tag">Features</div>
            <h3>Built Around Your Learning</h3>
            <p>Every part of the pipeline is designed to help you go from audio to confident performance.</p>
          </div>
          <div className="feature-grid">
            {FEATURES.map((f) => (
              <div key={f.title} className="feature-card">
                <div className="feature-icon-wrap">{f.icon}</div>
                <div>
                  <h4>{f.title}</h4>
                  <p>{f.desc}</p>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ── Demo Upload Zone ── */}
        <section className="demo-section" id="demo">
          <div className="section-head">
            <div className="section-tag">Try It Now</div>
            <h3>Upload a Recording to Get Started</h3>
            <p>Drop in any piano audio — we'll separate the stems and begin the transcription pipeline.</p>
          </div>
          <div
            className={`upload-zone${dragging ? " drag-over" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault(); setDragging(false);
              const f = e.dataTransfer.files[0];
              if (f) { setSelectedFile(f); setStatus(`Selected: ${f.name}`); setError(""); }
            }}
          >
            <div className="upload-zone-icon">🎼</div>
            <h4>Drop your audio file here</h4>
            <p>Supports MP3, WAV, M4A and more</p>
            <div className="upload-methods">
              <label className="method-btn">
                📁 <strong>Upload File</strong>
                <input type="file" accept="audio/*" onChange={handleFileChange} />
              </label>
              <button className="method-btn" onClick={handleSeparateStems}>
                🎤 <strong>Record Audio</strong>
              </button>
            </div>
            <p className="upload-note">
              {selectedFile ? `Ready: ${selectedFile.name}` : "No file selected yet"}
            </p>
          </div>
        </section>

        {/* ── Stem Results ── */}
        <section className="results" id="results">
          <div className="section-head">
            <div className="section-tag">Results</div>
            <h3>Stem Preview</h3>
            <p>Listen to each separated output and select the piano stem for transcription.</p>
          </div>
          {stems ? (
            <div className="results-grid">
              {Object.entries(stems).map(([name, path]) => (
                <div key={name} className="result-card">
                  <div className="result-top">
                    <h4>{name}</h4>
                    {name === "piano" && <span className="pill">Use This</span>}
                  </div>
                  <audio controls src={`${API_BASE}/${path}`} />
                  {name === "piano" && (
                    <button className="use-btn">Send to Tutor →</button>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="empty-box">
              No stems yet — upload a file above and click <strong>Separate Stems</strong>.
            </div>
          )}
        </section>

        {/* ── CTA Banner ── */}
        <section className="cta-banner">
          <h3>Ready to Learn Your Favourite Piece?</h3>
          <p>Upload your audio, let the AI transcribe it, and start practising with your personal tutor.</p>
          <button className="primary-btn" onClick={scrollToDemo}>
            Start Now — It's Free ▶
          </button>
        </section>

        {/* ── Footer ── */}
        <footer className="footer">
          <div className="footer-grid">
            <div className="footer-brand">
              <div className="logo">
                <span className="logo-icon">♪</span>
                Piano Automatic Transcription
              </div>
              <p>An AI pipeline for piano learners — stem separation, MIDI generation, and interactive tutoring, all in one place.</p>
            </div>
            {FOOTER_COLS.map((col) => (
              <div key={col.heading} className="footer-col">
                <h5>{col.heading}</h5>
                <ul>
                  {col.links.map((l) => (
                    <li key={l}><a href="#">{l}</a></li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
          <div className="footer-bottom">
            <span>© 2026 Piano Automatic Transcription · All rights reserved</span>
            <span>Privacy Policy · Terms of Use</span>
          </div>
        </footer>

      </div>
    </>
  );
}
