# Checkpoints

Place large model checkpoint files here for Docker or deployment, or mount this folder as a volume.

This folder is ignored by git except for this README and `.gitkeep`. The backend also still supports the legacy local checkpoint folder:

```text
Backend/efficient-seq2seq-piano-trans/checkpoints/
```

At runtime, `Backend/transcription_service.py` checks `MODEL_CHECKPOINT_DIR` when a configured checkpoint path is not available at its original location. In Docker, set:

```text
MODEL_CHECKPOINT_DIR=/app/checkpoints
```
