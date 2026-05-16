# Frontend, FastAPI Backend, and Docker Integration Report

## Project Overview

Piano IQ is an automatic piano music transcription and learning system. The project combines a web frontend, a FastAPI backend, model inference code, MIDI analysis tools, and Docker deployment files into one full-stack application.

The main goal of the system is to let a user upload piano audio or MIDI, convert audio into MIDI when needed, play the generated MIDI in an interactive browser workspace, compare performances against a reference, and receive tutor-style feedback.

The project is organized into three main runtime parts:

```text
frontend/      React + Vite user interface
Backend/       FastAPI application and runtime services
docker-compose.yml
               Two-service Docker runner for frontend and backend
```

The frontend gives users the visual learning experience. The backend connects the frontend to the transcription model, MIDI analysis code, tutor workflow, file storage, and optional audio separation. Docker ties both parts together so the project can run consistently on another machine or server.

## System Architecture

The application follows a client-server architecture.

```text
User Browser
    |
    | HTTP requests
    v
React/Vite Frontend
    |
    | Fetch API calls
    v
FastAPI Backend
    |
    | Python service calls
    v
Transcription Model, MIDI Analysis, Tutor Logic, Runtime Storage
```

The frontend does not run the heavy model directly in the browser. Instead, it sends files and requests to FastAPI. FastAPI saves uploaded files, calls the correct backend service, returns JSON responses, and serves generated MIDI files back to the frontend.

## Frontend Implementation

The frontend is located in `frontend/`. It is built with React and Vite.

Important frontend files include:

```text
frontend/package.json
frontend/src/app/App.jsx
frontend/src/piano/PianoPage.jsx
frontend/src/piano/components/ChatPanel.jsx
frontend/src/piano/hooks/useMidi.js
frontend/src/shared/api.js
frontend/Dockerfile
frontend/nginx.conf
```

The main frontend tools are:

| Tool | Purpose |
| --- | --- |
| React | Builds the user interface as reusable components |
| Vite | Provides fast local development and production builds |
| Tone.js | Handles MIDI/audio playback behavior in the browser |
| @tonejs/midi | Parses MIDI files for notes, timing, tempo, and playback data |
| Nginx | Serves the built frontend inside Docker |

### Frontend User Flow

The frontend starts with the landing and upload workflow in `frontend/src/app/App.jsx`. It accepts WAV, MP3, MID, and MIDI files.

If the user uploads a MIDI file, the frontend can open it directly in the piano tutor view because MIDI is already symbolic music data.

If the user uploads an audio file such as WAV or MP3, the frontend sends the file to the backend endpoint:

```text
POST /transcribe-upload
```

The backend returns a generated MIDI URL. The frontend then opens the tutor/player route and loads that MIDI.

The main practice workspace is implemented in `frontend/src/piano/PianoPage.jsx`. This page includes:

- MIDI loading and parsing
- Falling-note visualization
- Piano keyboard display
- Playback controls
- Speed control
- Looping practice workflow
- MIDI library drawer
- Performance comparison panel
- Tutor chat panel

The chat/tutor UI is implemented in `frontend/src/piano/components/ChatPanel.jsx`. It either sends chat messages to the backend tutor session or falls back to local frontend guidance when no prepared backend tutor session exists.

## Frontend to Backend Connection

The frontend uses `frontend/src/shared/api.js` to decide which backend URL to call.

By default, it calls FastAPI on port `8000` using the same hostname as the frontend page:

```text
http://localhost:5173 -> http://localhost:8000
http://134.208.3.192:5173 -> http://134.208.3.192:8000
```

The API base can also be overridden during deployment with:

```text
VITE_API_BASE=https://your-backend.example.com
```

This is important because it lets the same frontend code run locally, in Docker, or on a shared server without hardcoding one backend address.

The frontend uses the browser Fetch API to call FastAPI. Examples include:

```text
POST /transcribe-upload
POST /midi/analyze
POST /tutor/prepare
POST /tutor/message
GET  /library/midis
POST /library/midis
```

## FastAPI Backend Implementation

The backend is located in `Backend/`. The main application file is:

```text
Backend/app.py
```

The backend uses FastAPI to expose HTTP endpoints for the frontend. It also enables CORS so the frontend can call the API from local development or the shared host.

Allowed frontend origins include:

```text
http://localhost:5173
http://127.0.0.1:5173
http://134.208.3.192:5173
```

Extra allowed origins can be added using the environment variable:

```text
FRONTEND_ORIGINS
```

### Backend Runtime Storage

The backend stores generated and uploaded files in `Backend/runtime/`. This folder is ignored by git because it contains runtime data, not source code.

Runtime folders include:

```text
Backend/runtime/uploads/
Backend/runtime/separated/
Backend/runtime/transcriptions/
Backend/runtime/tutor_sessions/
Backend/runtime/midi_library/
```

The backend also mounts static routes so the frontend can download or load generated files:

```text
/uploads
/separated
/transcriptions
```

For example, when transcription creates a MIDI file, the backend returns a URL such as:

```text
/transcriptions/generated_file.mid
```

The frontend resolves this into a full backend URL and loads it in the piano player.

## Main Backend Services

The FastAPI app connects several backend services together.

### Transcription Service

The transcription connection is handled by:

```text
Backend/transcription_service.py
```

This file selects the active transcription backend using:

```text
TRANSCRIPTION_MODEL_BACKEND
```

The default backend is:

```text
experiment_pedals
```

That backend points to:

```text
Experiment/Efficient-Transformer-with-pedals/
```

The service loads the model inference entry point, resolves the configuration, finds the checkpoint, runs inference, and writes the generated MIDI file into the transcription runtime directory.

The high-level transcription flow is:

```text
Audio upload
    -> FastAPI saves file
    -> transcription_service loads selected model backend
    -> model inference runs
    -> MIDI file is written to Backend/runtime/transcriptions/
    -> FastAPI returns midi_url to frontend
```

### MIDI Analysis Service

MIDI analysis is connected through:

```text
Backend/Midi_Analysis/
```

The `/midi/analyze` endpoint resolves the MIDI path, runs quick MIDI analysis, and returns metrics and practice recommendations to the frontend. The frontend uses this information to display useful practice context and tutor feedback.

### Tutor Workflow

The tutor workflow is handled inside `Backend/app.py` using helper functions and the MIDI analysis tools.

The main tutor endpoints are:

```text
POST /tutor/prepare
POST /tutor/message
```

`/tutor/prepare` creates a tutor session. It can work in two modes:

| Mode | Description |
| --- | --- |
| Solo | User uploads one performance/audio/MIDI and receives practice feedback |
| Compare | User uploads a performance and a reference MIDI, then the backend compares them |

The tutor session stores metadata, analysis files, comparison results, summary cards, suggested questions, and chat state under:

```text
Backend/runtime/tutor_sessions/
```

`/tutor/message` continues an existing tutor conversation by using the stored session ID.

### MIDI Library

The MIDI library allows generated or uploaded MIDI files to be reused later.

Main endpoints:

```text
GET  /library/midis
POST /library/midis
GET  /library/midis/{item_id}/download
```

This lets the frontend display saved MIDI items, group them by project, open a reference MIDI, or download a stored file.

### Optional Stem Separation

The backend also contains an optional Spleeter wrapper:

```text
Backend/separation_service.py
```

The endpoint is:

```text
POST /separate
```

Spleeter is optional because it adds TensorFlow and increases the Docker image size. The default Docker setup does not install it unless the image is built with:

```text
INSTALL_SPLEETER=true
```

## API Endpoint Summary

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/` | GET | Health message and active transcription backend |
| `/ping` | GET | Simple backend health check |
| `/transcribe-upload` | POST | Upload audio and return generated MIDI |
| `/transcribe` | POST | Run transcription from a provided audio path |
| `/midi/analyze` | POST | Analyze MIDI metrics and practice information |
| `/tutor/prepare` | POST | Create a solo or comparison tutor session |
| `/tutor/message` | POST | Send a message to the prepared tutor session |
| `/library/midis` | GET | List saved MIDI library items |
| `/library/midis` | POST | Upload a MIDI file to the library |
| `/library/midis/{item_id}/download` | GET | Download or load a saved MIDI file |
| `/separate` | POST | Optional Spleeter stem separation |

## Complete Application Workflow

The complete user workflow is:

1. The user opens the React frontend.
2. The user uploads a WAV, MP3, MID, or MIDI file.
3. If the file is already MIDI, the frontend opens it directly in the tutor/player.
4. If the file is audio, the frontend sends it to FastAPI using `/transcribe-upload`.
5. FastAPI saves the upload into `Backend/runtime/uploads/`.
6. FastAPI calls `run_transcription()` from `Backend/transcription_service.py`.
7. The selected transcription backend loads the model and checkpoint.
8. Inference generates a MIDI file into `Backend/runtime/transcriptions/`.
9. FastAPI returns the generated `midi_url`.
10. The frontend resolves the URL and opens the piano tutor page.
11. The tutor page parses the MIDI, visualizes notes, and enables playback controls.
12. The frontend can call `/midi/analyze` for musical metrics.
13. The user can prepare a tutor session or comparison through `/tutor/prepare`.
14. The user can chat with the tutor through `/tutor/message`.
15. Saved MIDI files can be reused through the MIDI library endpoints.

## Docker Integration

Docker is used to make the system easier to run in a consistent environment. The project has two main Docker services:

```text
backend    FastAPI app on port 8000
frontend   React/Vite build served by nginx on port 5173
```

The services are defined in:

```text
docker-compose.yml
```

### Backend Dockerfile

The backend Dockerfile is:

```text
Backend/Dockerfile
```

It uses a PyTorch CUDA runtime image by default:

```text
pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
```

The backend image installs system packages such as `ffmpeg`, `libsndfile1`, and `git`, then installs Python dependencies from:

```text
requirements/backend.txt
```

It copies the backend, experiment model code, configs, and checkpoint folder into the image:

```text
Backend/
Experiment/
configs/
checkpoints/
```

Finally, it starts FastAPI with Uvicorn:

```text
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Frontend Dockerfile

The frontend Dockerfile is:

```text
frontend/Dockerfile
```

It uses a two-stage build:

1. A Node image installs dependencies and runs `npm run build`.
2. An Nginx image serves the production files from `/usr/share/nginx/html`.

The frontend container exposes port `80` internally. Docker Compose maps it to port `5173` on the host.

### Docker Compose Configuration

The Docker Compose file connects both services.

Backend service:

```text
Host port 8000 -> container port 8000
Runtime volume: ./Backend/runtime -> /app/Backend/runtime
Checkpoint volume: ./checkpoints -> /app/checkpoints:ro
```

Frontend service:

```text
Host port 5173 -> container port 80
Depends on backend
```

Important backend environment variables:

```text
BACKEND_RUNTIME_DIR=/app/Backend/runtime
MODEL_CHECKPOINT_DIR=/app/checkpoints
TRANSCRIPTION_MODEL_BACKEND=experiment_pedals
FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173,http://134.208.3.192:5173
```

The backend also has a health check that calls:

```text
http://127.0.0.1:8000/ping
```

## Running the Project

From the repository root, the full application can be started with:

```bash
docker compose up --build
```

Local URLs:

```text
Frontend: http://localhost:5173
Backend:  http://localhost:8000/ping
API docs: http://localhost:8000/docs
```

Shared host URLs:

```text
Frontend: http://134.208.3.192:5173
Backend:  http://134.208.3.192:8000/ping
API docs: http://134.208.3.192:8000/docs
```

Before running transcription, model checkpoint files should be placed in:

```text
checkpoints/
```

Inside Docker, that folder is mounted as:

```text
/app/checkpoints
```

This keeps large model weights outside git while still making them available to the backend container.

## Why FastAPI Was Used

FastAPI is a good fit for this project because it supports:

- File uploads through multipart forms
- JSON request and response models
- Automatic API documentation at `/docs`
- CORS middleware for frontend/backend separation
- Easy integration with existing Python model code
- Uvicorn deployment for local and Docker execution

Because the transcription and MIDI analysis code is already Python-based, FastAPI allows the frontend to access the whole backend pipeline without rewriting model logic in JavaScript.

## Why React and Vite Were Used

React is used because the frontend has many interactive states:

- Upload status
- Playback state
- MIDI loading state
- Tutor chat state
- Comparison state
- Library drawer state
- Piano visualization state

Vite is used because it gives fast development startup, simple builds, and clean environment variable support through `VITE_API_BASE`.

Together, React and Vite allow the application to feel like a real browser workspace instead of a static upload page.

## Why Docker Was Added

Docker was added to solve environment and deployment problems.

Without Docker, each machine would need the correct Python version, Node version, model dependencies, audio libraries, FastAPI dependencies, and frontend build tools installed manually.

With Docker:

- The backend runs in a PyTorch-ready container.
- The frontend builds in a Node container and is served by Nginx.
- Ports are predictable: frontend on `5173`, backend on `8000`.
- Runtime files are mounted into `Backend/runtime/`.
- Checkpoints are mounted into `/app/checkpoints`.
- The same command can start the whole system.

## Final Summary

The project evolved from separate backend model code into a connected full-stack application. The FastAPI backend became the bridge between the machine learning pipeline and the user interface. The React/Vite frontend gave users a practical way to upload files, view generated MIDI, play and loop music, compare performances, and use tutor feedback. Docker then packaged the frontend and backend into a repeatable two-service setup so the system can run locally or on a shared host with fewer setup issues.

In the final architecture, the frontend focuses on user interaction, visualization, and learning workflow, while FastAPI manages file handling, transcription, MIDI analysis, tutor sessions, and model execution. Docker Compose connects both services and makes the full application easier to build, run, and deploy.
