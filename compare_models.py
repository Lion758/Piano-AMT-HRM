import argparse
import multiprocessing
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_BACKEND_DIR = ROOT_DIR / "Backend" / "efficient-seq2seq-piano-trans"


def ensure_backend_on_path(backend_dir: Path) -> None:
    backend_dir_str = str(backend_dir)
    if backend_dir_str not in sys.path:
        sys.path.insert(0, backend_dir_str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare efficient-seq2seq checkpoints by config, parameter counts, and decode benchmark results."
    )
    parser.add_argument(
        "--backend-dir",
        type=str,
        default=str(DEFAULT_BACKEND_DIR),
        help="Path to the efficient-seq2seq-piano-trans backend directory.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        dest="checkpoints",
        required=True,
        help="Checkpoint path. Repeat this flag for multiple checkpoints.",
    )
    parser.add_argument(
        "--config",
        action="append",
        dest="configs",
        default=None,
        help="Optional config YAML path matching a checkpoint by position. If omitted, auto-detect experiment_config.yaml.",
    )
    parser.add_argument(
        "--config-name",
        action="append",
        dest="config_names",
        default=None,
        help="Optional Hydra config name under <backend-dir>/config, such as experiment_T5_V4_HierarchyPool.",
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Optional audio file used for speed benchmarking. If omitted, benchmark on the first test batch.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Optional override for training.batch_inference during speed benchmarking.",
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=1,
        help="Number of test batches to benchmark when --audio-path is not provided.",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warmup generate() runs before timing.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save the comparison table as CSV.",
    )
    parser.add_argument(
        "--include-historical-metrics",
        action="store_true",
        help="Include checkpoint-side historical test metrics from test_metrics_summary.csv.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=("cuda" if torch.cuda.is_available() else "cpu"),
        help="Benchmark device, default is cuda when available otherwise cpu.",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Optional OmegaConf dotlist overrides applied to every loaded config.",
    )
    return parser.parse_args()


def build_trainer(config: Any):
    from train import MT3Trainer

    return MT3Trainer(config)


def find_default_config(checkpoint_path: Path) -> Optional[Path]:
    for candidate in (
        checkpoint_path.parent / "experiment_config.yaml",
        checkpoint_path.parent.parent / "experiment_config.yaml",
        checkpoint_path.parent.parent.parent / "experiment_config.yaml",
    ):
        if candidate.exists():
            return candidate
    return None


def load_config(config_path: Path, checkpoint_path: Path, overrides: List[str], batch_size: Optional[int]) -> Any:
    config_file = Path(config_path).expanduser().resolve()
    with initialize_config_dir(config_dir=str(config_file.parent), version_base=None):
        config = compose(config_name=config_file.stem, overrides=overrides)
    config.model.checkpoint_path = str(checkpoint_path)
    config.training.mode = "test"
    if batch_size is not None:
        config.training.batch_inference = batch_size
        config.training.batch_test = batch_size
    return config


def resolve_config_reference(
    checkpoint_path: Path,
    config_ref: Optional[str],
    config_name: Optional[str],
    backend_dir: Path,
) -> Path:
    if config_ref is not None:
        return Path(config_ref).expanduser().resolve()
    if config_name is not None:
        return (backend_dir / "config" / f"{config_name}.yaml").resolve()
    default_config = find_default_config(checkpoint_path)
    if default_config is None:
        raise FileNotFoundError(f"Could not find a config for {checkpoint_path}. Pass --config or --config-name explicitly.")
    return default_config.resolve()


def load_checkpoint(trainer, checkpoint_path: Path, strict: bool = False) -> None:
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    ignore_layers = getattr(trainer.config.model, "checkpoint_ignore_layres", None)
    if ignore_layers:
        for key in list(state_dict.keys()):
            if key in ignore_layers:
                del state_dict[key]
    trainer.model.load_state_dict(state_dict, strict=strict)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    return {
        "total_params": int(sum(p.numel() for p in model.parameters())),
        "trainable_params": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def extract_turbo_quant_config(config: Any) -> Dict[str, Any]:
    model_cfg = getattr(config, "model", None)
    turbo_cfg = getattr(model_cfg, "turbo_quant", None) if model_cfg is not None else None
    result: Dict[str, Any] = {
        "decoder_window_size": getattr(model_cfg, "decoder_window_size", None) if model_cfg is not None else None,
        "turbo_quant_enabled": False,
    }
    if turbo_cfg is None:
        return result

    result.update(
        {
            "turbo_quant_enabled": bool(getattr(turbo_cfg, "enabled", False)),
            "turbo_quant_n_bits": getattr(turbo_cfg, "n_bits", None),
            "turbo_quant_qjl_projection_dim": getattr(turbo_cfg, "qjl_projection_dim", None),
            "turbo_quant_enable_qjl": getattr(turbo_cfg, "enable_qjl", None),
            "turbo_quant_min_cache_len": getattr(turbo_cfg, "min_cache_len", None),
        }
    )
    return result


def locate_metrics_csv(checkpoint_path: Path) -> Optional[Path]:
    for candidate in (
        Path(str(checkpoint_path) + "_test") / "test_metrics_summary.csv",
        checkpoint_path.parent.parent / "test_metrics_summary.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def load_average_metrics(metrics_csv: Optional[Path]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "historical_metrics_available": False,
        "historical_metrics_csv": None,
    }
    if metrics_csv is None or not metrics_csv.exists():
        return out
    df = pd.read_csv(metrics_csv)
    if df.empty:
        return out
    avg_row = df.iloc[0].to_dict()
    key_map = {
        "batch_size": "historical_batch_size",
        "note_precision": "historical_note_onset_precision",
        "note_recall": "historical_note_onset_recall",
        "note_f1": "historical_note_onset_f1",
        "note+offset_precision": "historical_note_with_offset_precision",
        "note+offset_recall": "historical_note_with_offset_recall",
        "note+offset_f1": "historical_note_with_offset_f1",
        "note+offset+velocity_precision": "historical_note_with_offset_velocity_precision",
        "note+offset+velocity_recall": "historical_note_with_offset_velocity_recall",
        "note+offset+velocity_f1": "historical_note_with_offset_velocity_f1",
        "target_length": "historical_target_length",
    }
    for old_key, new_key in key_map.items():
        out[new_key] = avg_row.get(old_key)
    out["historical_metrics_available"] = True
    out["historical_metrics_csv"] = str(metrics_csv)
    return out


def prepare_audio_benchmark_batch(trainer, audio_path: Path, device: torch.device) -> Dict[str, Any]:
    import torchaudio
    from data.constants import DEFAULT_SAMPLE_RATE

    wav, sr = torchaudio.load(str(audio_path))
    wav = wav.to(device)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != DEFAULT_SAMPLE_RATE:
        wav = torchaudio.transforms.Resample(sr, DEFAULT_SAMPLE_RATE).to(device)(wav)

    with torch.no_grad():
        encoder_inputs = trainer.features_extracter.to(device)(wav[:, :-1]).transpose(-1, -2)
        encoder_inputs = encoder_inputs.detach()
        if trainer.config.data.amplitude_to_db and trainer.config.data.features != "mel":
            encoder_inputs = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(encoder_inputs)

    n_frames = trainer.config.data.n_frames
    if encoder_inputs.shape[1] % n_frames != 0:
        pad_len = n_frames - (encoder_inputs.shape[1] % n_frames)
        encoder_inputs = torch.nn.functional.pad(encoder_inputs, (0, 0, 0, pad_len), value=0.0)
    encoder_inputs = encoder_inputs.view(-1, n_frames, encoder_inputs.shape[-1])
    return {
        "encoder_inputs": encoder_inputs,
        "target_seq_length": int(trainer.config.data.max_token_length),
        "benchmark_source": str(audio_path),
        "num_clips": int(encoder_inputs.shape[0]),
    }


def prepare_test_benchmark_batches(trainer, device: torch.device, num_batches: int) -> List[Dict[str, Any]]:
    import torchaudio

    batches: List[Dict[str, Any]] = []
    iterator = iter(trainer.test_dataloader())
    for _ in range(max(1, num_batches)):
        batch = next(iterator)
        input_waves = batch["inputs"].to(device)
        with torch.no_grad():
            encoder_inputs = trainer.features_extracter.to(device)(input_waves[:, :-1]).transpose(-1, -2)
            encoder_inputs = encoder_inputs.detach()
            if trainer.config.data.amplitude_to_db and trainer.config.data.features != "mel":
                encoder_inputs = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(encoder_inputs)
        batches.append(
            {
                "encoder_inputs": encoder_inputs,
                "target_seq_length": int(batch["decoder_targets"].size(1)),
                "benchmark_source": "test_dataloader:first_batches",
                "num_clips": int(encoder_inputs.shape[0]),
            }
        )
    return batches


def reset_turbo_quant_runtime_stats(model: torch.nn.Module) -> None:
    for module in model.modules():
        cache = getattr(module, "turbo_quant_cache", None)
        if cache is not None and hasattr(cache, "reset_stats"):
            cache.reset_stats()


def collect_turbo_quant_runtime_stats(model: torch.nn.Module) -> Dict[str, Any]:
    caches = []
    for module in model.modules():
        cache = getattr(module, "turbo_quant_cache", None)
        if cache is not None and hasattr(cache, "get_stats"):
            caches.append(cache)

    if not caches:
        return {}

    short_prefix_steps = 0
    quantized_steps = 0
    max_cache_len = 0
    observed_window = None

    for cache in caches:
        stats = cache.get_stats()
        short_prefix_steps += int(stats.get("short_prefix_steps", 0))
        quantized_steps += int(stats.get("quantized_steps", 0))
        max_cache_len = max(max_cache_len, int(stats.get("max_cache_len", 0)))
        window_size = stats.get("last_window_size")
        if window_size is not None:
            observed_window = window_size if observed_window is None else max(observed_window, window_size)

    return {
        "turbo_quant_layers": len(caches),
        "turbo_quant_short_prefix_steps": short_prefix_steps,
        "turbo_quant_quantized_steps": quantized_steps,
        "turbo_quant_bypass_observed": short_prefix_steps > 0,
        "turbo_quant_path_observed": quantized_steps > 0,
        "turbo_quant_max_cache_len": max_cache_len if max_cache_len > 0 else None,
        "turbo_quant_observed_window": observed_window,
    }


def count_decoded_tokens(output_tokens: torch.Tensor) -> int:
    from data.constants import TOKEN_END, TOKEN_PAD

    if output_tokens.ndim != 2:
        return int(output_tokens.numel())

    decoded_tokens = 0
    for sequence in output_tokens:
        eos_positions = (sequence == TOKEN_END).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            decoded_tokens += int(eos_positions[0].item()) + 1
            continue

        pad_positions = (sequence == TOKEN_PAD).nonzero(as_tuple=False)
        if pad_positions.numel() > 0:
            decoded_tokens += int(pad_positions[0].item())
            continue

        decoded_tokens += int(sequence.numel())

    return decoded_tokens


def benchmark_generate(trainer, benchmark_batches: List[Dict[str, Any]], warmup_runs: int, device: torch.device) -> Dict[str, Any]:
    model = trainer.model.to(device).eval()
    total_decoded_tokens = 0
    total_clips = 0
    total_decode_seconds = 0.0
    peak_allocated_bytes = None
    peak_reserved_bytes = None
    baseline_allocated_bytes = None
    baseline_reserved_bytes = None

    if device.type == "cuda":
        torch.cuda.empty_cache()

    with torch.no_grad():
        first_batch = benchmark_batches[0]
        for _ in range(max(0, warmup_runs)):
            _ = model.generate(
                first_batch["encoder_inputs"],
                target_seq_length=first_batch["target_seq_length"],
                berak_on_eos=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)

        reset_turbo_quant_runtime_stats(model)

        for batch in benchmark_batches:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                # Record baseline BEFORE generate (model weights + features already on GPU)
                batch_baseline_allocated = int(torch.cuda.memory_allocated(device))
                batch_baseline_reserved = int(torch.cuda.memory_reserved(device))
            start = time.perf_counter()
            output_tokens = model.generate(
                batch["encoder_inputs"],
                target_seq_length=batch["target_seq_length"],
                berak_on_eos=True,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                batch_peak_allocated = int(torch.cuda.max_memory_allocated(device))
                batch_peak_reserved = int(torch.cuda.max_memory_reserved(device))
                peak_allocated_bytes = batch_peak_allocated if peak_allocated_bytes is None else max(peak_allocated_bytes, batch_peak_allocated)
                peak_reserved_bytes = batch_peak_reserved if peak_reserved_bytes is None else max(peak_reserved_bytes, batch_peak_reserved)
                baseline_allocated_bytes = batch_baseline_allocated if baseline_allocated_bytes is None else min(baseline_allocated_bytes, batch_baseline_allocated)
                baseline_reserved_bytes = batch_baseline_reserved if baseline_reserved_bytes is None else min(baseline_reserved_bytes, batch_baseline_reserved)
            total_decode_seconds += time.perf_counter() - start
            total_decoded_tokens += count_decoded_tokens(output_tokens)
            total_clips += int(batch["num_clips"])

    has_cuda_stats = peak_allocated_bytes is not None
    result = {
        "benchmark_source": benchmark_batches[0]["benchmark_source"],
        "benchmark_batches": len(benchmark_batches),
        "benchmark_clips": total_clips,
        "decode_seconds": total_decode_seconds,
        "decoded_tokens": total_decoded_tokens,
        "decoded_tokens_per_second": (total_decoded_tokens / total_decode_seconds) if total_decode_seconds > 0 else None,
        "clips_per_second": (total_clips / total_decode_seconds) if total_decode_seconds > 0 else None,
        "cuda_model_baseline_allocated_mb": (baseline_allocated_bytes / (1024 ** 2)) if has_cuda_stats else None,
        "cuda_peak_allocated_mb": (peak_allocated_bytes / (1024 ** 2)) if has_cuda_stats else None,
        "cuda_peak_reserved_mb": (peak_reserved_bytes / (1024 ** 2)) if has_cuda_stats else None,
        "cuda_generate_overhead_allocated_mb": ((peak_allocated_bytes - baseline_allocated_bytes) / (1024 ** 2)) if has_cuda_stats else None,
        "cuda_generate_overhead_reserved_mb": ((peak_reserved_bytes - baseline_reserved_bytes) / (1024 ** 2)) if has_cuda_stats else None,
    }
    result.update(collect_turbo_quant_runtime_stats(model))
    return result


def compare_one(
    checkpoint_path: Path,
    config_path: Path,
    overrides: List[str],
    batch_size: Optional[int],
    audio_path: Optional[Path],
    num_batches: int,
    warmup_runs: int,
    device: torch.device,
    include_historical_metrics: bool,
) -> Dict[str, Any]:
    config = load_config(config_path, checkpoint_path, overrides, batch_size)
    trainer = build_trainer(config)
    load_checkpoint(trainer, checkpoint_path, strict=bool(getattr(config.model, "strict_checkpoint", False)))

    result: Dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "config": str(config_path),
        "encoder_name": str(config.model.encoder_name),
        "notes": str(getattr(config.training, "notes", "")),
        "device": str(device),
    }
    result.update(count_parameters(trainer.model))
    result.update(extract_turbo_quant_config(config))
    if include_historical_metrics:
        result.update(load_average_metrics(locate_metrics_csv(checkpoint_path)))

    benchmark_batches = (
        [prepare_audio_benchmark_batch(trainer, audio_path, device)]
        if audio_path is not None
        else prepare_test_benchmark_batches(trainer, device, num_batches=num_batches)
    )
    result.update(benchmark_generate(trainer, benchmark_batches, warmup_runs=warmup_runs, device=device))
    return result


def format_int(value: Any) -> Any:
    return f"{value:,}" if isinstance(value, int) else value


def _subprocess_compare_one(
    result_queue: multiprocessing.Queue,
    index: int,
    backend_dir: str,
    checkpoint_path: str,
    config_path: str,
    overrides: List[str],
    batch_size: Optional[int],
    audio_path: Optional[str],
    num_batches: int,
    warmup_runs: int,
    device_str: str,
    include_historical_metrics: bool,
) -> None:
    """Run compare_one in an isolated subprocess with its own CUDA context."""
    try:
        ensure_backend_on_path(Path(backend_dir))
        result = compare_one(
            checkpoint_path=Path(checkpoint_path),
            config_path=Path(config_path),
            overrides=overrides,
            batch_size=batch_size,
            audio_path=Path(audio_path) if audio_path else None,
            num_batches=num_batches,
            warmup_runs=warmup_runs,
            device=torch.device(device_str),
            include_historical_metrics=include_historical_metrics,
        )
        result_queue.put((index, result))
    except Exception:
        result_queue.put((index, traceback.format_exc()))


def main() -> None:
    args = parse_args()
    backend_dir = Path(args.backend_dir).expanduser().resolve()
    if not backend_dir.exists():
        raise FileNotFoundError(f"Backend directory not found: {backend_dir}")
    ensure_backend_on_path(backend_dir)

    checkpoints = [Path(p).expanduser().resolve() for p in args.checkpoints]
    configs = args.configs or []
    config_names = args.config_names or []
    if configs and config_names:
        raise ValueError("Use either --config or --config-name, not both.")
    if configs and len(configs) not in (1, len(checkpoints)):
        raise ValueError("Pass either one --config for all checkpoints or one --config per checkpoint.")
    if config_names and len(config_names) not in (1, len(checkpoints)):
        raise ValueError("Pass either one --config-name for all checkpoints or one --config-name per checkpoint.")

    resolved_configs: List[Path] = []
    for idx, checkpoint in enumerate(checkpoints):
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        config_ref = configs[0 if len(configs) == 1 else idx] if configs else None
        config_name = config_names[0 if len(config_names) == 1 else idx] if config_names else None
        config_path = resolve_config_reference(
            checkpoint_path=checkpoint,
            config_ref=config_ref,
            config_name=config_name,
            backend_dir=backend_dir,
        )
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        resolved_configs.append(config_path)

    device = torch.device(args.device)
    audio_path = Path(args.audio_path).expanduser().resolve() if args.audio_path else None

    # Run each model in an isolated subprocess so each gets its own CUDA
    # context.  This prevents kernel-cache and warmup effects from leaking
    # between models and ensures timing results match standalone runs.
    mp_ctx = multiprocessing.get_context("spawn")
    result_queue: multiprocessing.Queue = mp_ctx.Queue()
    rows: List[Optional[Dict[str, Any]]] = [None] * len(checkpoints)

    for idx, (checkpoint, config_path) in enumerate(zip(checkpoints, resolved_configs)):
        proc = mp_ctx.Process(
            target=_subprocess_compare_one,
            kwargs=dict(
                result_queue=result_queue,
                index=idx,
                backend_dir=str(backend_dir),
                checkpoint_path=str(checkpoint),
                config_path=str(config_path),
                overrides=args.overrides,
                batch_size=args.batch_size,
                audio_path=str(audio_path) if audio_path else None,
                num_batches=args.num_batches,
                warmup_runs=args.warmup_runs,
                device_str=str(device),
                include_historical_metrics=args.include_historical_metrics,
            ),
        )
        proc.start()
        proc.join()

    for _ in range(len(checkpoints)):
        idx, result = result_queue.get()
        if isinstance(result, str):
            raise RuntimeError(f"Subprocess for checkpoint index {idx} failed:\n{result}")
        rows[idx] = result

    df = pd.DataFrame(rows)
    preferred_columns = [
        "checkpoint",
        "config",
        "encoder_name",
        "notes",
        "turbo_quant_enabled",
        "turbo_quant_n_bits",
        "turbo_quant_qjl_projection_dim",
        "turbo_quant_min_cache_len",
        "decoder_window_size",
        "total_params",
        "trainable_params",
        "decoded_tokens_per_second",
        "clips_per_second",
        "decode_seconds",
        "decoded_tokens",
        "benchmark_batches",
        "benchmark_clips",
        "benchmark_source",
        "cuda_model_baseline_allocated_mb",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
        "cuda_generate_overhead_allocated_mb",
        "cuda_generate_overhead_reserved_mb",
        "turbo_quant_bypass_observed",
        "turbo_quant_path_observed",
        "turbo_quant_short_prefix_steps",
        "turbo_quant_quantized_steps",
        "turbo_quant_max_cache_len",
        "turbo_quant_observed_window",
        "historical_metrics_available",
        "historical_batch_size",
        "historical_note_onset_precision",
        "historical_note_onset_recall",
        "historical_note_onset_f1",
        "historical_note_with_offset_precision",
        "historical_note_with_offset_recall",
        "historical_note_with_offset_f1",
        "historical_note_with_offset_velocity_precision",
        "historical_note_with_offset_velocity_recall",
        "historical_note_with_offset_velocity_f1",
        "historical_target_length",
        "historical_metrics_csv",
    ]
    df = df[[c for c in preferred_columns if c in df.columns] + [c for c in df.columns if c not in preferred_columns]]

    printable = df.copy()
    for column in (
        "total_params",
        "trainable_params",
        "decoded_tokens",
        "benchmark_batches",
        "benchmark_clips",
        "turbo_quant_layers",
        "turbo_quant_short_prefix_steps",
        "turbo_quant_quantized_steps",
        "turbo_quant_max_cache_len",
        "turbo_quant_observed_window",
        "turbo_quant_n_bits",
        "turbo_quant_qjl_projection_dim",
        "turbo_quant_min_cache_len",
        "decoder_window_size",
    ):
        if column in printable.columns:
            printable[column] = printable[column].map(format_int)
    for column in (
        "decoded_tokens_per_second",
        "clips_per_second",
        "cuda_model_baseline_allocated_mb",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
        "cuda_generate_overhead_allocated_mb",
        "cuda_generate_overhead_reserved_mb",
        "decode_seconds",
    ):
        if column in printable.columns:
            printable[column] = printable[column].map(lambda x: None if pd.isna(x) else round(float(x), 3))

    print(printable.to_string(index=False))
    if args.output_csv:
        output_csv = Path(args.output_csv).expanduser().resolve()
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"\nSaved CSV to {output_csv}")


if __name__ == "__main__":
    main()
