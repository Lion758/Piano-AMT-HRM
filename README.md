# Piano-AMT-HRM

Automatic piano music transcription with stem separation, MIDI generation, and an interactive piano tutor.

## Project Structure

```text
frontend/      React + Vite client for upload, playback, visualization, and tutor UI
Backend/       FastAPI backend, MIDI analysis, and production transcription integration
Experiment/    Research model variants, training, evaluation, and experiment notes
requirements/  Docker-friendly Python requirement entry points
configs/       Deployment env examples; model configs stay with each model package
models/        Model-code map for Docker packaging
checkpoints/   Local or mounted model weights, ignored by git except docs
docs/          Project maps and setup-oriented documentation
outputs/       Generated evaluation artifacts and demo outputs, ignored by git
reference/     External reference implementations and papers, ignored by git
tmp/           Scratch notes and temporary local artifacts
```

Backend-generated files are grouped under `Backend/runtime/` and ignored by git:

```text
Backend/runtime/uploads/          Uploaded audio and MIDI files
Backend/runtime/separated/        Stem separation outputs
Backend/runtime/transcriptions/   Generated MIDI files served by /transcriptions
Backend/runtime/tutor_sessions/   Tutor summaries, comparisons, and chat state
Backend/runtime/midi_library/     Local MIDI library files and index
```

For a fuller map of where new files should go, see
[`docs/PROJECT_STRUCTURE.md`](docs/PROJECT_STRUCTURE.md).

Docker-oriented files are grouped without moving import-sensitive model packages:

```text
requirements/backend.txt        Backend + production transcription dependencies
requirements/research.txt       Research/training dependencies
configs/backend.env.example     Backend environment variables for deployment
checkpoints/                    Mount or place large model weights here
Backend/Dockerfile              Backend image
frontend/Dockerfile             Frontend nginx image
docker-compose.yml              Local two-service Docker runner
.dockerignore                   Keeps runtime data and large local artifacts out of builds
```

Docker quick start:

```bash
docker compose up --build
```

Then open `http://localhost:5173`. See [`docs/DOCKER.md`](docs/DOCKER.md)
for checkpoint and optional Spleeter setup.

The main frontend source is organized as:

```text
frontend/src/
  app/          Landing/upload experience
  assets/       Images used by the UI
  piano/        Piano tutor page, player controls, MIDI hooks, and note helpers
  shared/       Shared frontend utilities such as API URL handling
```

## Frontend Commands

```bash
cd frontend
npm install
npm run dev
```

If Vite hits a Linux file watcher `ENOSPC` error, use:

```bash
npm run dev:poll
```

Build for deployment:

```bash
npm run build
```

## Backend Command

```bash
cd Backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
