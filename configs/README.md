# Configs

This folder is for deployment-level configuration such as environment variable templates.

Model configuration files stay inside their model packages because Hydra and local imports expect those paths:

```text
Backend/efficient-seq2seq-piano-trans/config/
Experiment/Efficient-Transformer-with-pedals/config/
Experiment/Turbo/efficient-seq2seq-piano-trans-main/config/
```

For Docker, copy `backend.env.example` to your deployment environment and adjust the paths for your container or mounted volumes.
