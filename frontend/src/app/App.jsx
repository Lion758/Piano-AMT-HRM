import { useState } from "react";
import "./App.css";
import pianoBanner from "../assets/piano-hero-dark.png";
import { API_BASE, resolveApiUrl } from "../shared/api.js";

const STEPS = [
  { icon: "🎵", num: 1, title: "WAV / MP3 / MID Input", desc: "Start with a WAV or MP3 recording, or upload an existing MID/MIDI file when you already have the score data." },
  { icon: "🧠", num: 2, title: "Transcribe Audio", desc: "Audio files are sent through the transcription model so notes, timing, and durations can be converted into symbolic events." },
  { icon: "🎼", num: 3, title: "MIDI Output", desc: "The result is a playable MIDI file that can be opened in the browser player, downloaded, reused, and analysed." },
  { icon: "▶", num: 4, title: "Learn in the Player", desc: "Learners can slow the piece down, follow the falling notes, and isolate difficult phrases before playing the full song." },
  { icon: "🎙️", num: 5, title: "Record Your Own Take", desc: "After practising, users can upload their own performance recording or MIDI take for comparison against the reference." },
  { icon: "🧑‍🏫", num: 6, title: "Interactive Tutor", desc: "The tutor uses the MIDI analysis and comparison data to explain mistakes, suggest practice plans, and guide the next attempt." },
];

const FEATURES = [
  { icon: "📁", title: "WAV, MP3, and MIDI Input", desc: "Use audio when you need transcription, or open MID/MIDI directly when the symbolic file already exists." },
  { icon: "🎼", title: "Automatic MIDI Generation", desc: "Audio is converted into a MIDI output that can drive playback, visualisation, analysis, and tutor feedback." },
  { icon: "🔁", title: "A/B Loop Markers", desc: "Set two points on the player timeline, repeat that exact snippet, and combine it with slower playback for targeted practice." },
  { icon: "🎙️", title: "Performance Comparison", desc: "Upload a learner recording or MIDI take after practice and compare it with the reference material." },
  { icon: "🧑‍🏫", title: "Interactive AI Tutor", desc: "The tutor turns transcription and comparison data into specific corrections, drills, and next-step guidance." },
  { icon: "📱", title: "Browser Workspace", desc: "Upload, transcribe, listen, slow down, loop, compare, and ask the tutor without installing a separate desktop tool." },
];

const FOOTER_COLS = [
  { heading: "Tool", links: ["How It Works", "Features", "Upload & Transcribe"] },
  { heading: "Resources", links: ["Documentation", "API Reference", "Changelog"] },
  { heading: "Project", links: ["About", "Contact", "GitHub"] },
];

const ACCEPTED_UPLOADS = ".wav,.mp3,.mid,.midi,audio/wav,audio/x-wav,audio/mpeg,audio/midi,audio/x-midi";

function isSupportedUpload(file) {
  if (!file) return false;
  const name = file.name || "";
  const type = file.type || "";
  return /\.(wav|mp3|mid|midi)$/i.test(name)
    || ["audio/wav", "audio/x-wav", "audio/mpeg", "audio/mp3", "audio/midi", "audio/x-midi"].includes(type);
}

function isMidiUpload(file) {
  if (!file) return false;
  const name = file.name || "";
  const type = file.type || "";
  return /\.(mid|midi)$/i.test(name) || type === "audio/midi" || type === "audio/x-midi";
}

function buildPianoHash(midiUrl, libraryItem = null) {
  const params = new URLSearchParams({ midi: midiUrl });
  if (libraryItem?.id) params.set("reference", libraryItem.id);
  if (libraryItem?.project) params.set("project", libraryItem.project);
  if (libraryItem?.title) params.set("referenceTitle", libraryItem.title);
  return `#/piano?${params.toString()}`;
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [status, setStatus] = useState("Upload WAV, MP3, or MID to begin.");
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const selectUploadFile = (file) => {
    if (!file) {
      setSelectedFile(null);
      setStatus("Upload WAV, MP3, or MID to begin.");
      return;
    }

    if (!isSupportedUpload(file)) {
      setSelectedFile(null);
      setError("Please choose a WAV, MP3, MID, or MIDI file.");
      setStatus("Unsupported file type.");
      return;
    }

    setSelectedFile(file);
    setError("");
    setStatus(file ? `Selected: ${file.name}` : "Upload WAV, MP3, or MID to begin.");
  };

  const handleFileChange = (e) => {
    selectUploadFile(e.target.files[0]);
  };

  const handleStartPipeline = async () => {
    if (!selectedFile) { setError("Please choose a WAV, MP3, or MID file first."); return; }
    if (!isSupportedUpload(selectedFile)) {
      setError("Please choose a WAV, MP3, MID, or MIDI file.");
      setStatus("Unsupported file type.");
      return;
    }

    setError("");

    if (isMidiUpload(selectedFile)) {
      const midiUrl = URL.createObjectURL(selectedFile);
      setStatus("Opening the tutor from your MIDI file...");
      window.location.hash = buildPianoHash(midiUrl);
      return;
    }

    setIsProcessing(true);
    setStatus("Uploading audio and transcribing to MIDI...");
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const response = await fetch(`${API_BASE}/transcribe-upload`, { method: "POST", body: formData });
      if (!response.ok) {
        let msg = "Failed to transcribe audio.";
        try { const p = await response.json(); if (p?.detail) msg = p.detail; } catch { /* ignore malformed error payload */ }
        throw new Error(msg);
      }
      const data = await response.json();
      const midiUrl = resolveApiUrl(data.midi_url);
      setStatus("MIDI output ready. Opening the tutor...");
      window.location.hash = buildPianoHash(midiUrl, data.library_item);
    } catch (err) {
      setError(err.message || "Something went wrong.");
      setStatus("Transcription failed.");
    } finally {
      setIsProcessing(false);
    }
  };

  const scrollToDemo = () =>
    document.getElementById("demo")?.scrollIntoView({ behavior: "smooth" });

  return (
    <>
      {/* Topbar */}
      <div className="topbar">
        Piano IQ: audio-to-MIDI transcription, practice playback, recording comparison, and AI-guided feedback.
      </div>

      <div className="page">

        {/* ── Navbar ── */}
        <header className="navbar">
          <div className="logo">
            <img className="logo-icon" src="/piano-iq-icon.svg" alt="" aria-hidden="true" />
            <span className="logo-copy">
              <span className="brand-name">Piano IQ</span>
              <span className="brand-tagline">Intelligent Piano Learning System</span>
            </span>
          </div>
          <nav className="navlinks">
            <a href="#workflow">How it works</a>
            <a href="#features">Features</a>
            <a href="#demo">Try it</a>
            <a href="#future-work">Future work</a>
          </nav>
          <div className="navactions">
            <button className="nav-btn">Login</button>
            <button className="nav-btn solid">Get Started</button>
          </div>
        </header>

        {/* ── Hero ── */}
        <section className="hero" style={{ "--hero-bg": `url(${pianoBanner})` }}>
          <div className="hero-copy">
            <div className="hero-badge">
              <span className="badge-dot" />
              WAV / MP3 / MID · MIDI Output · AI Tutor
            </div>
            <h1>Piano IQ<br /><em>Intelligent Piano Learning System</em></h1>
            <h2>
              Upload WAV or MP3 audio to generate MIDI, or open MID files directly.
              Learn the piece in the player, record your own take, then ask the tutor
              for focused feedback.
            </h2>
            <div className="cta-row">
              <button className="primary-btn" onClick={scrollToDemo}>
                Upload & Transcribe
              </button>
              <a href="#/piano" className="primary-btn primary-btn-secondary">
                Open Tutor
              </a>
              <a href="#workflow" className="ghost-btn">See How It Works</a>
            </div>
          </div>
        </section>

        {/* ── Workflow ── */}
        <section className="workflow" id="workflow">
          <div className="section-head">
            <div className="section-tag">How It Works</div>
            <h3>From Input File to Interactive Lesson</h3>
            <p>The current scope is direct transcription and MIDI-driven learning, with recording comparison feeding the tutor.</p>
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
            <div className="section-tag">Upload & Transcribe</div>
            <h3>Upload a File to Get Started</h3>
            <p>Use WAV or MP3 when you need transcription, or MID/MIDI when you already have a symbolic file.</p>
          </div>
          <div
            className={`upload-zone${dragging ? " drag-over" : ""}`}
            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
            onDragLeave={() => setDragging(false)}
            onDrop={(e) => {
              e.preventDefault(); setDragging(false);
              const f = e.dataTransfer.files[0];
              selectUploadFile(f);
            }}
          >
            <div className="upload-zone-icon">🎼</div>
            <h4>Drop your WAV, MP3, or MID file here</h4>
            <p>Audio becomes MIDI output; MIDI files open directly in the tutor player.</p>
            <div className="upload-methods">
              <label className="method-btn">
                📁 <strong>Upload File</strong>
                <input type="file" accept={ACCEPTED_UPLOADS} onChange={handleFileChange} />
              </label>
              <button
                className="method-btn method-btn-primary"
                onClick={handleStartPipeline}
                disabled={!selectedFile || isProcessing}
                type="button"
              >
                <strong>
                  {isProcessing
                    ? "Transcribing..."
                    : selectedFile && isMidiUpload(selectedFile)
                      ? "Open MIDI in Tutor"
                      : "Transcribe to MIDI"}
                </strong>
              </button>
            </div>

            <p className="upload-note">
              {selectedFile ? `Ready: ${selectedFile.name}` : "No file selected yet"}
            </p>
            <p className="upload-note">
              {error || status}
            </p>
          </div>
        </section>

        {/* ── Future Work ── */}
        <section className="future-work" id="future-work">
          <div className="section-head">
            <div className="section-tag">Future Work</div>
            <h3>Mixed-Audio Separation Comes Later</h3>
            <p>The current home pipeline is restricted to WAV/MP3 transcription and MID/MIDI playback. Full-mix stem separation is kept as future work so the main learning flow stays clear.</p>
          </div>
          <div className="future-work-grid">
            <div className="future-work-card">
              <span>Now</span>
              <strong>WAV / MP3 / MID input</strong>
              <p>Audio is transcribed into MIDI output. Existing MIDI files skip transcription and open directly for playback, looping, slow practice, comparison, and tutor feedback.</p>
            </div>
            <div className="future-work-card future-work-card-muted">
              <span>Later</span>
              <strong>Stem separation for full mixes</strong>
              <p>Full arrangement separation will be treated as a future enhancement instead of the first home-page pipeline step.</p>
            </div>
          </div>
        </section>

        {/* ── CTA Banner ── */}
        <section className="cta-banner">
          <h3>Ready to Learn Your Favourite Piece?</h3>
          <p>Upload audio or MIDI, open the player, loop the hard passage, and practise with your personal tutor.</p>
          <button className="primary-btn" onClick={scrollToDemo}>
            Start Now
          </button>
        </section>

        {/* ── Footer ── */}
        <footer className="footer">
          <div className="footer-grid">
            <div className="footer-brand">
              <div className="logo">
                <img className="logo-icon" src="/piano-iq-icon.svg" alt="" aria-hidden="true" />
                <span className="logo-copy">
                  <span className="brand-name">Piano IQ</span>
                  <span className="brand-tagline">Intelligent Piano Learning System</span>
                </span>
              </div>
              <p>An AI learning pipeline for transcription, MIDI playback, recording comparison, and interactive tutoring, all in one place.</p>
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
            <span>© 2026 Piano IQ · Intelligent Piano Learning System · All rights reserved</span>
            <span>Privacy Policy · Terms of Use</span>
          </div>
        </footer>

      </div>
    </>
  );
}
