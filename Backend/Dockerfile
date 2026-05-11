# syntax=docker/dockerfile:1

ARG PYTORCH_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BACKEND_RUNTIME_DIR=/app/Backend/runtime \
    MODEL_CHECKPOINT_DIR=/app/checkpoints \
    TRANSCRIPTION_MODEL_BACKEND=experiment_pedals

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ requirements/

ARG INSTALL_SPLEETER=false
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements/backend.txt \
    && if [ "$INSTALL_SPLEETER" = "true" ]; then python -m pip install -r requirements/separation.txt; fi

COPY Backend/ Backend/
COPY Experiment/ Experiment/
COPY configs/ configs/
COPY checkpoints/ checkpoints/

RUN mkdir -p /app/Backend/runtime /app/checkpoints

WORKDIR /app/Backend

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
