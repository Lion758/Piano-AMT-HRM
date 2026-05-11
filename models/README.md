# Models

This folder documents where model code lives for Docker packaging.

The active application model code stays in package-local folders so imports and Hydra config loading keep working:

```text
Backend/efficient-seq2seq-piano-trans/model/        Production transcription model modules
Backend/efficient-seq2seq-piano-trans/config/       Production model configs
Experiment/Efficient-Transformer-with-pedals/model/ Pedal-aware research model modules
Experiment/Efficient-Transformer-with-pedals/config/ Pedal-aware research configs
Experiment/Turbo/efficient-seq2seq-piano-trans-main/model/ TurboQuant experiment modules
```

Put model weight files in `checkpoints/`, not here. That keeps source code, configs, and large binary weights separate for Docker builds.
