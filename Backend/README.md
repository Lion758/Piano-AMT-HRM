# Backend

The backend contains the FastAPI service and the model code used by the app at runtime.

```text
app.py                         API routes and tutor workflow
separation_service.py          Spleeter stem separation wrapper
transcription_service.py       Transcription backend selector and output handling
Midi_Analysis/                 MIDI parsing, comparison, analysis, and GPT tutor helpers
efficient-seq2seq-piano-trans/ Production-supported transcription model package
tests/                         Backend tests
runtime/                       Local generated files, ignored by git
```

`runtime/` is created automatically and can also be moved by setting `BACKEND_RUNTIME_DIR`.
It stores uploads, separated stems, generated MIDI files, tutor session state, and the local MIDI library.

For Docker, use `requirements/backend.txt` from the repo root and mount checkpoint files at `checkpoints/` or set `MODEL_CHECKPOINT_DIR` to the mounted path.

Run the backend from this folder:

```bash
cd Backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
