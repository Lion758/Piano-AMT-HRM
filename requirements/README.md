# Requirements

Use this folder when building Docker images or setting up a fresh machine.

```text
backend.txt      FastAPI backend plus the production transcription package deps
research.txt     Extra research/training dependencies for Experiment/
separation.txt   Optional Spleeter dependency for stem separation
```

The root `environment.yml` is still kept for the existing Conda workflow. For Docker, prefer `backend.txt` from a PyTorch base image, then install frontend dependencies from `frontend/package.json`.
