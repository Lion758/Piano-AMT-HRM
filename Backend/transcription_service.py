from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf


BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
RUNTIME_DIR = Path(os.getenv("BACKEND_RUNTIME_DIR", BASE_DIR / "runtime")).expanduser().resolve()
TRANSCRIPTIONS_DIR = RUNTIME_DIR / "transcriptions"
CHECKPOINTS_DIR = Path(os.getenv("MODEL_CHECKPOINT_DIR", REPO_ROOT / "checkpoints")).expanduser().resolve()
BACKEND_ENV_VAR = "TRANSCRIPTION_MODEL_BACKEND"
DEFAULT_BACKEND_NAME = "experiment_pedals"
CHECKPOINT_SUFFIXES = {".ckpt", ".pt", ".pth", ".safetensors", ".bin"}


@dataclass(frozen=True)
class RuntimeBackend:
    name: str
    root: Path

    @property
    def inference_path(self) -> Path:
        return self.root / "inference.py"


BACKEND_REGISTRY = {
    "experiment_pedals": RuntimeBackend(
        name="experiment_pedals",
        root=(REPO_ROOT / "Experiment" / "Efficient-Transformer-with-pedals").resolve(),
    ),
    "seq2seq": RuntimeBackend(
        name="seq2seq",
        root=(BASE_DIR / "efficient-seq2seq-piano-trans").resolve(),
    ),
}

_RUNTIME_LOCK = threading.Lock()
_RUNTIMES: dict[str, ModuleType] = {}
_ACTIVE_BACKEND_NAME: str | None = None


def _normalize_backend_name(name: str | None) -> str:
    return (name or DEFAULT_BACKEND_NAME).strip().lower() or DEFAULT_BACKEND_NAME


def _get_backend(backend_name: str | None = None) -> RuntimeBackend:
    requested_name = _normalize_backend_name(backend_name or os.getenv(BACKEND_ENV_VAR, DEFAULT_BACKEND_NAME))

    try:
        return BACKEND_REGISTRY[requested_name]
    except KeyError as exc:
        supported = ", ".join(sorted(BACKEND_REGISTRY))
        raise ValueError(
            f"Unsupported transcription backend '{requested_name}'. "
            f"Set {BACKEND_ENV_VAR} to one of: {supported}."
        ) from exc


def get_active_backend_name() -> str:
    global _ACTIVE_BACKEND_NAME

    backend = _get_backend()
    if _ACTIVE_BACKEND_NAME is None:
        _ACTIVE_BACKEND_NAME = backend.name
    elif _ACTIVE_BACKEND_NAME != backend.name:
        raise RuntimeError(
            "Transcription backend is process-sticky and has already been initialized as "
            f"'{_ACTIVE_BACKEND_NAME}'. Restart the server to switch to '{backend.name}'."
        )
    return _ACTIVE_BACKEND_NAME


def _selector_key(backend_name: str, config_path: str | None, config_name: str | None) -> str:
    if config_path:
        return f"{backend_name}:path:{Path(config_path).expanduser().resolve()}"
    return f"{backend_name}:name:{(config_name or 'main_config').strip() or 'main_config'}"


def _ensure_model_root_on_path(backend: RuntimeBackend) -> None:
    model_root = str(backend.root)
    if model_root not in sys.path:
        sys.path.insert(0, model_root)


def _prepare_runtime_env() -> None:
    os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def _patch_cpu_attention_backends() -> None:
    if torch.cuda.is_available():
        return

    attention_module = sys.modules.get("model.Attention")
    if attention_module is not None and not getattr(attention_module, "_cpu_fallback_patched", False):
        attention_module.flash_attn_func = None

        def _cpu_sliding_window_attention(self, query, key, value, decode=False):
            seq_len = query.size(1)
            mask = attention_module.make_sliding_window_mask(
                seq_len,
                int(self.window_size),
                bool(self.is_causal),
            ).to(query.device)
            attention_bias = torch.where(
                mask > 0,
                torch.zeros_like(mask, dtype=torch.float32, device=query.device),
                torch.full_like(mask, float("-inf"), dtype=torch.float32, device=query.device),
            )

            q = query.permute(0, 2, 1, 3).to(torch.float32)
            k = key.permute(0, 2, 1, 3).to(torch.float32)
            v = value.permute(0, 2, 1, 3).to(torch.float32)
            dropout_p = 0.0 if decode else float(self.dropout_rate)

            x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_bias, dropout_p=dropout_p)
            return x.permute(0, 2, 1, 3).to(query.dtype)

        attention_module.Multi_Head_Attention.flash_attn_sliding_window_attention = _cpu_sliding_window_attention
        attention_module._cpu_fallback_patched = True

    trm_module = sys.modules.get("model.trm_encoder")
    if trm_module is not None:
        trm_module._flash_attn_func = None

    hrm_layers_module = sys.modules.get("model.layers.hrm_layers")
    if hrm_layers_module is not None:
        hrm_layers_module.flash_attn_func = None


def _load_runtime(backend: RuntimeBackend, selector_key: str) -> ModuleType:
    with _RUNTIME_LOCK:
        runtime = _RUNTIMES.get(selector_key)
        if runtime is not None:
            return runtime

        if not backend.inference_path.is_file():
            raise FileNotFoundError(f"Inference entrypoint not found: {backend.inference_path}")

        _prepare_runtime_env()
        _ensure_model_root_on_path(backend)

        module_name = f"{backend.name}_inference_{hashlib.sha1(selector_key.encode('utf-8')).hexdigest()}"
        spec = importlib.util.spec_from_file_location(module_name, backend.inference_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load inference module from {backend.inference_path}")

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise

        _patch_cpu_attention_backends()
        _RUNTIMES[selector_key] = module
        return module


def _resolve_existing_path(path_value: str) -> Path:
    candidate = Path(path_value).expanduser()
    candidates = [candidate]

    if not candidate.is_absolute():
        candidates.append(BASE_DIR / candidate)
        candidates.append(REPO_ROOT / candidate)

    for current in candidates:
        resolved = current.resolve()
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(f"Audio file not found: {path_value}")


def _resolve_backend_file(path_value: str, backend: RuntimeBackend) -> Path:
    candidate = Path(path_value).expanduser()
    candidates = [candidate]

    if not candidate.is_absolute():
        candidates.append(backend.root / candidate)
        candidates.append(BASE_DIR / candidate)
        candidates.append(REPO_ROOT / candidate)
        if candidate.parts and candidate.parts[0] == CHECKPOINTS_DIR.name:
            candidates.append(CHECKPOINTS_DIR / Path(*candidate.parts[1:]))

    if candidate.suffix.lower() in CHECKPOINT_SUFFIXES:
        candidates.append(CHECKPOINTS_DIR / candidate.name)

    for current in candidates:
        resolved = current.resolve()
        if resolved.is_file():
            return resolved

    raise FileNotFoundError(f"Model/config file not found for backend '{backend.name}': {path_value}")


def _resolve_checkpoint(config, backend: RuntimeBackend) -> str:
    checkpoint_path = OmegaConf.select(config, "model.checkpoint_path")
    if checkpoint_path:
        resolved_checkpoint = _resolve_backend_file(checkpoint_path, backend)
        config.model.checkpoint_path = str(resolved_checkpoint)
        return str(resolved_checkpoint)

    resume_checkpoint = OmegaConf.select(config, "training.resume_ckpt_path")
    if resume_checkpoint:
        resolved_checkpoint = _resolve_backend_file(resume_checkpoint, backend)
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
            output_path = (TRANSCRIPTIONS_DIR.parent / requested).resolve()
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

    backend_name = get_active_backend_name()
    backend = _get_backend(backend_name)

    resolved_config_path = None
    if config_path:
        resolved_config_path = str(_resolve_backend_file(config_path, backend))

    effective_config_name = (config_name or "main_config").strip() or "main_config"
    selector = _selector_key(backend_name, resolved_config_path, effective_config_name)
    runtime = _load_runtime(backend, selector)

    resolved_audio_path = _resolve_existing_path(audio_path)
    resolved_output_path, relative_output_path = _resolve_output_path(resolved_audio_path, midi_output_path)

    config = runtime.load_config(resolved_config_path, effective_config_name, [])
    _resolve_checkpoint(config, backend)
    OmegaConf.update(config, "audio_path", str(resolved_audio_path), force_add=True)
    OmegaConf.update(config, "midi_path", str(resolved_output_path), force_add=True)

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
        "model_backend": backend_name,
    }
