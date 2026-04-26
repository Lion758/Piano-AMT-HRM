# Impulse Responses for Convolution Reverb Augmentation

Drop mono or stereo `.wav` impulse response files into this directory.
The training pipeline picks them up at dataloader-construction time via
`data.dataset_Audio2Midi._build_waveform_augmenter`, which scans for
`*.wav` and constructs an `audiomentations.ApplyImpulseResponse` step.

If the directory is empty, the reverb stage is silently skipped and
training proceeds with the remaining augmentations.

## Recommended source

- EchoThief Impulse Response Library — https://www.echothief.com
  Pick ~14 IRs with RT60 <= ~1.2s (rooms / small halls / studios).
  Avoid very long cathedral-style IRs: they smear onsets past the
  50 ms mir_eval tolerance and become label-corrupting.

## Constraints

- Sample rate is automatically resampled by audiomentations to match
  the input (16 kHz here), but providing 16 kHz or 44.1 kHz IRs avoids
  per-step resampling cost.
- Files must be readable by `soundfile` / `librosa`.
