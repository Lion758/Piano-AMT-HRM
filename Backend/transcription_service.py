from __future__ import annotations

import hashlib
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType

from omegaconf import OmegaConf


BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "efficient-seq2seq-piano-trans"
INFERENCE_PATH = MODEL_ROOT / "inference.py"
TRANSCRIPTIONS_DIR = Path("transcriptions")

_RUNTIME_LOCK = threading.Lock()
_RUNTIMES: dict[str, ModuleType] = {}


def _selector_key(config_path: str | None, config_name: str | None) -> str:
    if config_path:
        return f"path:{Path(config_path).expanduser().resolve()}"
    return f"name:{(config_name or 'main_config').strip() or 'main_config'}"


def _ensure_model_root_on_path() -> None:
    model_root = str(MODEL_ROOT)
    if model_root not in sys.path:
        sys.path.insert(0, model_root)


def _load_runtime(selector_key: str) -> ModuleType:
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(selector_key)
        if runtime is not None:
            return runtime

        if not INFERENCE_PATH.is_file():
            raise FileNotFoundError(f"Inference entrypoint not found: {INFERENCE_PATH}")

        _ensure_model_root_on_path()

        module_name = f"seq2seq_inference_{hashlib.sha1(selector_key.encode('utf-8')).hexdigest()}"
        spec = importlib.util.spec_from_file_location(module_name, INFERENCE_PATH)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load inference module from {INFERENCE_PATH}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        _RUNTIMES[selector_key] = module
        return module


def _resolve_existing_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    candidates = [candidate]

    if not candidate.is_absolute():
        candidates.append(BASE_DIR / candidate)

    for current in candidates:
        resolved = current.resolve()
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(f"Audio file not found: {path_value}")


def _resolve_model_file(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    candidates = [candidate]

    if not candidate.is_absolute():
        candidates.append(MODEL_ROOT / candidate)
        candidates.append(BASE_DIR / candidate)

    for current in candidates:
        resolved = current.resolve()
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(f"Model file not found: {path_value}")


def _resolve_checkpoint(config) -> str:
    checkpoint_path = OmegaConf.select(config, "model.checkpoint_path")
    if checkpoint_path:
        resolved_checkpoint = _resolve_model_file(checkpoint_path)
        config.model.checkpoint_path = str(resolved_checkpoint)
        return str(resolved_checkpoint)

    resume_checkpoint = OmegaConf.select(config, "training.resume_ckpt_path")
    if resume_checkpoint:
        resolved_checkpoint = _resolve_model_file(resume_checkpoint)
        config.model.checkpoint_path = str(resolved_checkpoint)
        return str(resolved_checkpoint)

    raise ValueError("No checkpoint configured. Set model.checkpoint_path or training.resume_ckpt_path.")


def _resolve_output_path(audio_path: Path, midi_output_path: str | None) -> tuple[Path, str]:
    base_dir = TRANSCRIPTIONS_DIR.resolve()
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)

    if midi_output_path:
        requested = Path(midi_output_path).expanduser()
        if requested.suffix.lower() != ".mid":
            requested = requested.with_suffix(".mid")

        if requested.is_absolute():
            output_path = requested.resolve()
        elif requested.parts and requested.parts[0] == TRANSCRIPTIONS_DIR.name:
            output_path = requested.resolve()
        else:
            output_path = (TRANSCRIPTIONS_DIR / requested).resolve()
    else:
        output_name = f"{audio_path.parent.name}_{audio_path.stem}.mid"
        output_path = (TRANSCRIPTIONS_DIR / output_name).resolve()

    try:
        relative_path = output_path.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError("midi_output_path must resolve inside the transcriptions directory.") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path, relative_path.as_posix()


def run_transcription(
    audio_path: str,
    config_path: str | None = None,
    config_name: str | None = "main_config",
    midi_output_path: str | None = None,
) -> dict:
    if not audio_path or not audio_path.strip():
        raise ValueError("audio_path is required.")

    resolved_config_path = None
    if config_path:
        resolved_config_path = str(_resolve_model_file(config_path))

    selector = _selector_key(resolved_config_path, config_name)
    runtime = _load_runtime(selector)

    effective_config_name = (config_name or "main_config").strip() or "main_config"
    resolved_audio_path = _resolve_existing_path(audio_path)
    resolved_output_path, relative_output_path = _resolve_output_path(resolved_audio_path, midi_output_path)

    config = runtime.load_config(resolved_config_path, effective_config_name, [])
    _resolve_checkpoint(config)
    config.audio_path = str(resolved_audio_path)
    config.midi_path = str(resolved_output_path)

    runtime.run_inference(config)

    if not resolved_output_path.is_file():
        raise FileNotFoundError(f"Transcription did not produce a MIDI file: {resolved_output_path}")

    return {
        "message": "MIDI transcription completed successfully",
        "audio_path": str(resolved_audio_path),
        "midi_path": str(resolved_output_path),
        "midi_url": f"/transcriptions/{relative_output_path}",
        "config_path": resolved_config_path,
        "config_name": effective_config_name,
    }
