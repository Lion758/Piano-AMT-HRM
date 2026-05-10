# Piano-AMT-HRM

Automatic piano music transcription with stem separation, MIDI generation, and an interactive piano tutor.

## Project Structure

```text
frontend/      React + Vite app for upload, playback, visualization, and tutor UI
Backend/       FastAPI service for separation, transcription, MIDI library, and tutor APIs
Experiment/    Model training, evaluation, and research experiments
outputs/       Evaluation artifacts and demo outputs
reference/     External reference implementations and supporting material
```

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

## Netlify

This repo includes `netlify.toml`, so Netlify can build the frontend from the repo root:

```text
Base directory: frontend
Build command: npm run build
Publish directory: frontend/dist
```

Set `VITE_API_BASE` in Netlify environment variables to the deployed backend URL, for example:

```text
VITE_API_BASE=https://your-backend.example.com
```

On the backend host, allow the Netlify frontend origin:

```text
FRONTEND_ORIGINS=https://your-site.netlify.app
```
