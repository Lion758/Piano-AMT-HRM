# Project Structure

This repo is arranged around the pieces that make the final project run:

```text
.
|-- frontend/      React + Vite application
|-- Backend/       FastAPI service and runtime model integration
|-- Experiment/    Training, evaluation, and research model variants
|-- requirements/  Docker-friendly Python install entry points
|-- configs/       Deployment env examples
|-- models/        Model-code map for Docker packaging
|-- checkpoints/   Local or mounted model weights, ignored by git except docs
|-- docs/          Project maps and setup notes
|-- outputs/       Generated comparison/evaluation outputs, ignored by git
|-- reference/     External papers and reference implementations, ignored by git
`-- tmp/           Scratch notes and temporary local artifacts
```

## Where New Files Should Go

Use `frontend/` for UI work:

```text
frontend/src/app/       App shell and landing/upload flow
frontend/src/piano/     Piano tutor page, player controls, hooks, and helpers
frontend/src/shared/    Shared frontend utilities
frontend/public/        Static files copied by Vite
```

Use `Backend/` for the running API:

```text
Backend/app.py                         FastAPI routes
Backend/separation_service.py          Audio stem separation wrapper
Backend/transcription_service.py       Runtime transcription backend selector
Backend/Midi_Analysis/                 MIDI parsing, comparison, and tutor logic
Backend/efficient-seq2seq-piano-trans/ Production-supported transcription model code
Backend/tests/                         Backend API and MIDI analysis tests
```

Use `Backend/runtime/` for local files generated while the backend runs. This folder is ignored by git.

```text
Backend/runtime/uploads/          Uploaded audio and MIDI files
Backend/runtime/separated/        Spleeter stem outputs
Backend/runtime/transcriptions/   Generated MIDI files served by /transcriptions
Backend/runtime/tutor_sessions/   Tutor session summaries and state
Backend/runtime/midi_library/     Local MIDI library files and index
```

Use `Experiment/` for research and training work:

```text
Experiment/Efficient-Transformer-with-pedals/   Current pedal-aware research model
Experiment/Turbo/                               TurboQuant and related experiments
Experiment/transcription_model_and_frame_metrics.md
```

Use `requirements/` for Docker or deployment dependency entry points:

```text
requirements/backend.txt    Backend API plus production transcription dependencies
requirements/research.txt   Research/training dependencies
```

Use `configs/` for deployment-level config templates. Model configs stay inside each model package because Hydra resolves them relative to those package folders.

Use `checkpoints/` for large model weight files during Docker deployment. The backend checks `MODEL_CHECKPOINT_DIR`, which defaults to this folder, while still supporting the legacy package-local checkpoint path.

Use `models/` as the model-code map. The actual model modules stay in package-local folders so imports keep working.

Use `reference/` only for third-party papers, cloned reference projects, or comparison material. Keep application code out of this folder.

## Commit Hygiene

Before committing, source files should usually be in `frontend/`, `Backend/`, `Experiment/`, `requirements/`, `configs/`, `models/`, `checkpoints/`, or `docs/`. Runtime data, checkpoint binaries, datasets, generated outputs, and local references should stay ignored unless there is a specific reason to commit a small fixture.
