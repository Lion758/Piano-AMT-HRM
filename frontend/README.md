# Piano AMT Frontend

React + Vite frontend for the piano transcription and tutor workflow.

## Structure

```text
src/
  app/          Landing page and upload workflow
  assets/       UI images
  piano/        Piano tutor, MIDI playback, keyboard, and analysis panels
  shared/       Shared frontend utilities
```

## Commands

```bash
npm install
npm run dev
npm run build
npm run preview
```

If Linux file watching hits `ENOSPC`, use:

```bash
npm run dev:poll
```

## API URL

Local development defaults to `http://localhost:8000`. For deployment, set:

```text
VITE_API_BASE=https://your-backend.example.com
```

The backend should also allow the deployed frontend origin in `FRONTEND_ORIGINS`.
