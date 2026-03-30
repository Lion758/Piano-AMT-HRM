import argparse
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare efficient-seq2seq checkpoints by params, speed, and available test metrics."
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
        help="Optional config path matching a checkpoint by position. If omitted, auto-detect experiment_config.yaml.",
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
    if metrics_csv is None or not metrics_csv.exists():
        return {}
    df = pd.read_csv(metrics_csv)
    if df.empty:
        return {}
    avg_row = df.iloc[0].to_dict()
    out: Dict[str, Any] = {}
    for key, value in avg_row.items():
        if key != "midi_path":
            out[f"metric_{key}"] = value
    out["metrics_csv"] = str(metrics_csv)
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
        "num_examples": int(encoder_inputs.shape[0]),
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
                "num_examples": int(encoder_inputs.shape[0]),
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


def benchmark_generate(trainer, benchmark_batches: List[Dict[str, Any]], warmup_runs: int, device: torch.device) -> Dict[str, Any]:
    model = trainer.model.to(device).eval()
    total_tokens = 0
    total_examples = 0
    total_seconds = 0.0
    peak_allocated_bytes = None
    peak_reserved_bytes = None

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
            total_seconds += time.perf_counter() - start
            total_tokens += int(output_tokens.numel())
            total_examples += int(output_tokens.shape[0])

    result = {
        "benchmark_source": benchmark_batches[0]["benchmark_source"],
        "benchmark_batches": len(benchmark_batches),
        "benchmark_examples": total_examples,
        "benchmark_seconds": total_seconds,
        "generated_tokens": total_tokens,
        "tokens_per_second": (total_tokens / total_seconds) if total_seconds > 0 else None,
        "examples_per_second": (total_examples / total_seconds) if total_seconds > 0 else None,
        "cuda_peak_allocated_mb": (peak_allocated_bytes / (1024 ** 2)) if peak_allocated_bytes is not None else None,
        "cuda_peak_reserved_mb": (peak_reserved_bytes / (1024 ** 2)) if peak_reserved_bytes is not None else None,
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


def main() -> None:
    args = parse_args()
    checkpoints = [Path(p).expanduser().resolve() for p in args.checkpoints]
    configs = args.configs or []
    if configs and len(configs) not in (1, len(checkpoints)):
        raise ValueError("Pass either one --config for all checkpoints or one --config per checkpoint.")

    resolved_configs: List[Path] = []
    for idx, checkpoint in enumerate(checkpoints):
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        config_path = Path(configs[0 if len(configs) == 1 else idx]).expanduser().resolve() if configs else find_default_config(checkpoint)
        if config_path is None or not config_path.exists():
            raise FileNotFoundError(f"Could not find a config for {checkpoint}. Pass --config explicitly.")
        resolved_configs.append(config_path)

    device = torch.device(args.device)
    audio_path = Path(args.audio_path).expanduser().resolve() if args.audio_path else None
    rows = [
        compare_one(
            checkpoint_path=checkpoint,
            config_path=config_path,
            overrides=args.overrides,
            batch_size=args.batch_size,
            audio_path=audio_path,
            num_batches=args.num_batches,
            warmup_runs=args.warmup_runs,
            device=device,
        )
        for checkpoint, config_path in zip(checkpoints, resolved_configs)
    ]

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
        "tokens_per_second",
        "examples_per_second",
        "cuda_peak_allocated_mb",
        "cuda_peak_reserved_mb",
        "turbo_quant_bypass_observed",
        "turbo_quant_path_observed",
        "turbo_quant_short_prefix_steps",
        "turbo_quant_quantized_steps",
        "turbo_quant_max_cache_len",
        "turbo_quant_observed_window",
        "benchmark_seconds",
        "generated_tokens",
        "benchmark_batches",
        "benchmark_examples",
        "benchmark_source",
        "metric_note_f1",
        "metric_note+offset_f1",
        "metric_note+offset+velocity_f1",
        "metrics_csv",
    ]
    df = df[[c for c in preferred_columns if c in df.columns] + [c for c in df.columns if c not in preferred_columns]]

    printable = df.copy()
    for column in (
        "total_params",
        "trainable_params",
        "generated_tokens",
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
    for column in ("tokens_per_second", "examples_per_second", "cuda_peak_allocated_mb", "cuda_peak_reserved_mb", "benchmark_seconds"):
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
