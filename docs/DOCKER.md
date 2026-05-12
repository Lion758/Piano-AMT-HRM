# Docker Setup

This project has two Docker services:

```text
frontend   React/Vite app served by nginx on http://localhost:5173
backend    FastAPI app on http://localhost:8000
```

## Required Checkpoints

Put model checkpoint files in the top-level `checkpoints/` folder before running transcription. The backend mounts that folder at `/app/checkpoints` and uses:

```text
MODEL_CHECKPOINT_DIR=/app/checkpoints
```

If a config points to an old local checkpoint path, the backend also checks `MODEL_CHECKPOINT_DIR` for a file with the same basename. For example, a config path ending in `epoch=96-step=170000.ckpt` can be satisfied by:

```text
checkpoints/epoch=96-step=170000.ckpt
```

Checkpoint binaries are ignored by git.

## Run

From the repo root:

```bash
docker compose up --build
```

The default backend image uses a PyTorch CUDA runtime image. To build from a different PyTorch base image, set `PYTORCH_IMAGE`:

```bash
PYTORCH_IMAGE=your-pytorch-image-tag docker compose up --build
```

Open:

```text
Frontend: http://localhost:5173
Backend:  http://localhost:8000/ping
```

On the shared host, use the host IP instead:

```text
Frontend: http://134.208.3.192:5173
Backend:  http://134.208.3.192:8000/ping
FastAPI docs: http://134.208.3.192:8000/docs
```

By default, the frontend calls FastAPI on port `8000` using the same hostname
that loaded the page. For example, `http://134.208.3.192:5173` calls
`http://134.208.3.192:8000`. To force a different backend URL, set
`VITE_API_BASE` before rebuilding the frontend.

Backend runtime files are written to:

```text
Backend/runtime/
```

## Optional Stem Separation

The default backend image does not install Spleeter because it pulls TensorFlow and makes the image much larger. The API still starts normally, but `/separate` requires Spleeter.

To include Spleeter:

```bash
docker compose build --build-arg INSTALL_SPLEETER=true backend
docker compose up
```

## Useful Commands

Rebuild only the backend:

```bash
docker compose build backend
```

Run backend logs:

```bash
docker compose logs -f backend
```

Stop containers:

```bash
docker compose down
```
