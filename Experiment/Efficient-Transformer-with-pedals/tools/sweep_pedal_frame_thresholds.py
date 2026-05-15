#!/usr/bin/env python
"""Sweep state-head pedal event extraction thresholds from saved eval JSON."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import types
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))


USING_FALLBACK_MIR_EVAL = False
MISSING_SYMUSIC = False


def _fallback_precision_recall_f1_overlap(
    ref_intervals,
    ref_pitches,
    est_intervals,
    est_pitches,
    onset_tolerance=0.05,
    offset_ratio=0.2,
    offset_min_tolerance=0.05,
    **_,
):
    ref_intervals = np.asarray(ref_intervals, dtype=np.float64).reshape(-1, 2)
    est_intervals = np.asarray(est_intervals, dtype=np.float64).reshape(-1, 2)
    ref_pitches = np.asarray(ref_pitches)
    est_pitches = np.asarray(est_pitches)

    matched_ref = set()
    matched_count = 0
    for est_idx, est_interval in enumerate(est_intervals):
        best_ref_idx = None
        best_onset_delta = None
        for ref_idx, ref_interval in enumerate(ref_intervals):
            if ref_idx in matched_ref:
                continue
            if ref_idx < len(ref_pitches) and est_idx < len(est_pitches) and ref_pitches[ref_idx] != est_pitches[est_idx]:
                continue
            onset_delta = abs(est_interval[0] - ref_interval[0])
            if onset_delta > onset_tolerance:
                continue
            if offset_ratio is not None:
                ref_duration = max(0.0, ref_interval[1] - ref_interval[0])
                offset_tolerance = max(offset_min_tolerance, offset_ratio * ref_duration)
                if abs(est_interval[1] - ref_interval[1]) > offset_tolerance:
                    continue
            if best_ref_idx is None or onset_delta < best_onset_delta:
                best_ref_idx = ref_idx
                best_onset_delta = onset_delta
        if best_ref_idx is not None:
            matched_ref.add(best_ref_idx)
            matched_count += 1

    precision = matched_count / len(est_intervals) if len(est_intervals) > 0 else 0.0
    recall = matched_count / len(ref_intervals) if len(ref_intervals) > 0 else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1, 0.0


def _install_optional_dependency_stubs() -> None:
    global MISSING_SYMUSIC, USING_FALLBACK_MIR_EVAL

    missing_music21 = importlib.util.find_spec("music21") is None
    missing_symusic = importlib.util.find_spec("symusic") is None
    missing_partitura = importlib.util.find_spec("partitura") is None
    MISSING_SYMUSIC = missing_symusic

    if missing_music21:
        sys.modules.setdefault("music21", types.ModuleType("music21"))

    if missing_symusic:
        symusic_module = types.ModuleType("symusic")
        symusic_module.Score = type("Score", (), {})
        symusic_module.TimeUnit = types.SimpleNamespace(second="second")
        sys.modules.setdefault("symusic", symusic_module)

    if missing_partitura:
        sys.modules.setdefault("partitura", types.ModuleType("partitura"))

    if missing_symusic or missing_partitura:
        pianoroll_parser_module = types.ModuleType("utils.pianoroll_parser")
        pianoroll_parser_module.get_notes_with_pedal = lambda midi_path: (None, None)
        sys.modules.setdefault("utils.pianoroll_parser", pianoroll_parser_module)

    if importlib.util.find_spec("mir_eval") is None:
        USING_FALLBACK_MIR_EVAL = True
        mir_eval_module = types.ModuleType("mir_eval")
        mir_eval_util_module = types.ModuleType("mir_eval.util")
        mir_eval_util_module.midi_to_hz = lambda values: np.asarray(values, dtype=np.float32)
        mir_eval_transcription_module = types.ModuleType("mir_eval.transcription")
        mir_eval_transcription_module.precision_recall_f1_overlap = _fallback_precision_recall_f1_overlap
        mir_eval_transcription_velocity_module = types.ModuleType("mir_eval.transcription_velocity")
        mir_eval_transcription_velocity_module.precision_recall_f1_overlap = _fallback_precision_recall_f1_overlap
        mir_eval_multipitch_module = types.ModuleType("mir_eval.multipitch")
        mir_eval_multipitch_module.evaluate = lambda *args, **kwargs: {}

        mir_eval_module.util = mir_eval_util_module
        mir_eval_module.transcription = mir_eval_transcription_module
        mir_eval_module.transcription_velocity = mir_eval_transcription_velocity_module
        mir_eval_module.multipitch = mir_eval_multipitch_module
        sys.modules.setdefault("mir_eval", mir_eval_module)
        sys.modules.setdefault("mir_eval.util", mir_eval_util_module)
        sys.modules.setdefault("mir_eval.transcription", mir_eval_transcription_module)
        sys.modules.setdefault("mir_eval.transcription_velocity", mir_eval_transcription_velocity_module)
        sys.modules.setdefault("mir_eval.multipitch", mir_eval_multipitch_module)


_install_optional_dependency_stubs()

from data.constants import DEFAULT_HOP_WIDTH, DEFAULT_SAMPLE_RATE, sm_tokenizer
from metrics import transcription_metrics


DEFAULT_EVENT_EXTRACTORS = ("state_hysteresis", "trend_dual_trigger")
DEFAULT_THRESHOLD_ON_VALUES = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
DEFAULT_THRESHOLD_OFF_VALUES = (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
DEFAULT_OFFSET_THRESHOLD_VALUES = (0.30, 0.40, 0.50, 0.60, 0.70)
DEFAULT_MIN_ON_DELTA_VALUES = (0.0, 0.005, 0.01)
DEFAULT_MIN_DOWN_FRAMES = (1, 2, 3, 4, 5, 6, 7, 8)
DEFAULT_MIN_UP_FRAMES = (1, 2, 3, 4)
DEFAULT_TARGET_METRIC = "frame_head_pedal+offset_f1"


@dataclass
class TrackData:
    name: str
    data_list: list[dict]
    pedal_frame_output: np.ndarray
    pedal_offset_output: np.ndarray | None
    reference_pedal_event_list: list[dict]
    piece_end_time: float


def parse_float_list(value: str) -> tuple[float, ...]:
    return tuple(float(item.strip()) for item in value.split(",") if item.strip())


def parse_int_list(value: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in value.split(",") if item.strip())


def parse_string_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def load_reference_pedal_events(midi_path: str) -> list[dict]:
    tsv_path = os.path.splitext(midi_path)[0] + ".midi-notes.tsv"
    if os.path.exists(tsv_path):
        reference_df = pd.read_csv(tsv_path, sep="\t")
    else:
        if MISSING_SYMUSIC:
            raise RuntimeError(
                f"Reference TSV is missing for {midi_path}, and symusic is not installed for MIDI fallback parsing."
            )
        reference_df = sm_tokenizer.midi_to_dataframe(midi_path)
    return transcription_metrics.reference_pedal_events_from_dataframe(reference_df)


def load_track(json_path: Path, sec_per_frame: float) -> TrackData:
    with json_path.open("r") as f:
        track_json = json.load(f)

    data_list = sorted(track_json["data_list"], key=lambda row: row["frame_offsets"])
    if len(data_list) == 0:
        raise ValueError(f"{json_path} has an empty data_list.")
    if "pedal_frame_output" not in data_list[0]:
        raise ValueError(f"{json_path} does not contain pedal_frame_output.")

    pedal_frame_output = transcription_metrics.collect_trimmed_frame_arrays(
        data_list,
        "pedal_frame_output",
    )
    pedal_offset_output = None
    if "pedal_offset_output" in data_list[0]:
        _, pedal_offset_output = transcription_metrics.collect_trimmed_frame_arrays(
            data_list,
            "pedal_frame_output",
            "pedal_offset_output",
        )
    reference_pedal_event_list = load_reference_pedal_events(data_list[0]["midi_path"])
    total_frames = max(
        int(row.get("total_frames", row["frame_offsets"] + len(row["pedal_frame_output"])))
        for row in data_list
    )
    piece_end_time = total_frames * sec_per_frame
    if len(reference_pedal_event_list) > 0:
        piece_end_time = max(
            piece_end_time,
            max(float(event["time"]) for event in reference_pedal_event_list),
        )

    return TrackData(
        name=str(track_json.get("audio_name") or json_path.stem),
        data_list=data_list,
        pedal_frame_output=pedal_frame_output,
        pedal_offset_output=pedal_offset_output,
        reference_pedal_event_list=reference_pedal_event_list,
        piece_end_time=piece_end_time,
    )


def valid_threshold_pairs(
    threshold_on_values: tuple[float, ...],
    threshold_off_values: tuple[float, ...],
) -> list[tuple[float, float]]:
    return [
        (threshold_on, threshold_off)
        for threshold_on, threshold_off in product(threshold_on_values, threshold_off_values)
        if threshold_off < threshold_on
    ]


def mean_or_nan(values: list[float]) -> float:
    valid_values = [value for value in values if not math.isnan(value)]
    if len(valid_values) == 0:
        return float("nan")
    return float(np.mean(valid_values))


def default_distance_steps(row: dict) -> float:
    distance = (
        abs(row["frame_head_threshold_on"] - 0.50) / 0.05
        + abs(row["frame_head_threshold_off"] - 0.40) / 0.05
        + abs(row["frame_head_min_down_frames"] - 3)
        + abs(row["frame_head_min_up_frames"] - 2)
    )
    if row.get("frame_head_event_extractor") == "trend_dual_trigger":
        distance += abs(row["frame_head_offset_threshold"] - 0.50) / 0.10
        distance += abs(row["frame_head_min_on_delta"] - 0.0) / 0.005
    return distance


def score_combination(
    tracks: list[TrackData],
    raw_spans_by_track: list[list[tuple[int, int]]],
    sec_per_frame: float,
    event_extractor: str,
    threshold_on: float,
    threshold_off: float,
    offset_threshold: float | None,
    min_on_delta: float | None,
    min_down_frames: int,
    min_up_frames: int,
) -> dict:
    pedal_precision = []
    pedal_recall = []
    pedal_f1 = []
    pedal_offset_precision = []
    pedal_offset_recall = []
    pedal_offset_f1 = []
    reference_precision = []
    reference_recall = []
    reference_f1 = []
    predicted_span_counts = []
    reference_span_counts = []
    scored_track_count = 0

    for track, raw_spans in zip(tracks, raw_spans_by_track):
        frame_head_pedal_event_list = transcription_metrics.pedal_frame_spans_to_events(
            raw_spans,
            sec_per_frame=sec_per_frame,
            min_pedal_down_frames=min_down_frames,
            min_pedal_up_frames=min_up_frames,
        )
        piece_end_time = track.piece_end_time
        predicted_spans = transcription_metrics.pedal_events_to_spans(
            frame_head_pedal_event_list,
            piece_end_time=piece_end_time,
        )
        reference_spans = transcription_metrics.pedal_events_to_spans(
            track.reference_pedal_event_list,
            piece_end_time=piece_end_time,
        )
        predicted_span_counts.append(float(len(predicted_spans)))
        reference_span_counts.append(float(len(reference_spans)))

        if len(track.reference_pedal_event_list) == 0:
            pedal_precision.append(float("nan"))
            pedal_recall.append(float("nan"))
            pedal_f1.append(float("nan"))
            pedal_offset_precision.append(float("nan"))
            pedal_offset_recall.append(float("nan"))
            pedal_offset_f1.append(float("nan"))
            reference_precision.append(float("nan"))
            reference_recall.append(float("nan"))
            reference_f1.append(float("nan"))
            continue

        scored_track_count += 1
        diagnostic_metrics, _, _ = transcription_metrics.cal_pedal_metrics(
            frame_head_pedal_event_list,
            track.reference_pedal_event_list,
            piece_end_time=piece_end_time,
        )
        reference_metrics, _, _ = transcription_metrics.cal_reference_pedal_metrics(
            frame_head_pedal_event_list,
            track.reference_pedal_event_list,
            piece_end_time=piece_end_time,
        )

        pedal_precision.append(diagnostic_metrics["pedal_precision"])
        pedal_recall.append(diagnostic_metrics["pedal_recall"])
        pedal_f1.append(diagnostic_metrics["pedal_f1"])
        pedal_offset_precision.append(diagnostic_metrics["pedal+offset_precision"])
        pedal_offset_recall.append(diagnostic_metrics["pedal+offset_recall"])
        pedal_offset_f1.append(diagnostic_metrics["pedal+offset_f1"])
        reference_precision.append(reference_metrics["pedal_precision"])
        reference_recall.append(reference_metrics["pedal_recall"])
        reference_f1.append(reference_metrics["pedal_f1"])

    row = {
        "frame_head_event_extractor": event_extractor,
        "frame_head_threshold_on": threshold_on,
        "frame_head_threshold_off": threshold_off,
        "frame_head_offset_threshold": offset_threshold,
        "frame_head_min_on_delta": min_on_delta,
        "frame_head_min_down_frames": min_down_frames,
        "frame_head_min_up_frames": min_up_frames,
        "track_count": len(tracks),
        "scored_track_count": scored_track_count,
        "frame_head_pedal_precision": mean_or_nan(pedal_precision),
        "frame_head_pedal_recall": mean_or_nan(pedal_recall),
        "frame_head_pedal_f1": mean_or_nan(pedal_f1),
        "frame_head_pedal+offset_precision": mean_or_nan(pedal_offset_precision),
        "frame_head_pedal+offset_recall": mean_or_nan(pedal_offset_recall),
        "frame_head_pedal+offset_f1": mean_or_nan(pedal_offset_f1),
        "frame_head_reference_pedal_precision": mean_or_nan(reference_precision),
        "frame_head_reference_pedal_recall": mean_or_nan(reference_recall),
        "frame_head_reference_pedal_f1": mean_or_nan(reference_f1),
        "mean_predicted_pedal_span_count": mean_or_nan(predicted_span_counts),
        "mean_reference_pedal_span_count": mean_or_nan(reference_span_counts),
    }
    row["default_distance_steps"] = default_distance_steps(row)
    return row


def choose_best_row(df: pd.DataFrame, target_metric: str) -> pd.Series:
    if target_metric not in df.columns:
        raise ValueError(f"Unknown target metric {target_metric!r}. Available columns: {list(df.columns)}")

    best_target = df[target_metric].max()
    candidates = df[df[target_metric] >= best_target - 0.001].copy()
    best_pedal_f1 = candidates["frame_head_pedal_f1"].max()
    candidates = candidates[candidates["frame_head_pedal_f1"] == best_pedal_f1].copy()
    best_reference_f1 = candidates["frame_head_reference_pedal_f1"].max()
    candidates = candidates[candidates["frame_head_reference_pedal_f1"] == best_reference_f1].copy()
    candidates = candidates.sort_values(
        by=[
            "default_distance_steps",
            "frame_head_event_extractor",
            "frame_head_threshold_on",
            "frame_head_threshold_off",
            "frame_head_offset_threshold",
            "frame_head_min_on_delta",
            "frame_head_min_down_frames",
            "frame_head_min_up_frames",
        ],
        ascending=[True, True, True, True, True, True, True, True],
    )
    return candidates.iloc[0]


def format_row_summary(row: pd.Series) -> str:
    offset_text = ""
    if row["frame_head_event_extractor"] == "trend_dual_trigger":
        offset_text = (
            f", offset={row['frame_head_offset_threshold']:.2f}, "
            f"min_delta={row['frame_head_min_on_delta']:.3f}"
        )
    return (
        f"extractor={row['frame_head_event_extractor']}, "
        f"on={row['frame_head_threshold_on']:.2f}, "
        f"off={row['frame_head_threshold_off']:.2f}"
        f"{offset_text}, "
        f"min_down={int(row['frame_head_min_down_frames'])}, "
        f"min_up={int(row['frame_head_min_up_frames'])}, "
        f"pedal+offset_f1={row['frame_head_pedal+offset_f1']:.6f}, "
        f"pedal_f1={row['frame_head_pedal_f1']:.6f}, "
        f"reference_pedal_f1={row['frame_head_reference_pedal_f1']:.6f}"
    )


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    eval_dir = Path(args.eval_dir)
    json_paths = sorted(eval_dir.glob("*.output.json"))
    if len(json_paths) == 0:
        raise ValueError(f"No .output.json files found in {eval_dir}.")
    if USING_FALLBACK_MIR_EVAL:
        print("Warning: mir_eval is not installed; using local interval-matching fallback metrics.")

    sec_per_frame = DEFAULT_HOP_WIDTH / DEFAULT_SAMPLE_RATE
    print(f"Loading {len(json_paths)} track JSON files from {eval_dir} ...")
    tracks = [load_track(json_path, sec_per_frame=sec_per_frame) for json_path in json_paths]

    event_extractors = args.event_extractors
    valid_event_extractors = {"state_hysteresis", "trend_dual_trigger"}
    unknown_extractors = sorted(set(event_extractors) - valid_event_extractors)
    if unknown_extractors:
        raise ValueError(
            f"Unknown event extractors {unknown_extractors}; expected {sorted(valid_event_extractors)}."
        )

    threshold_pairs = valid_threshold_pairs(args.threshold_on_values, args.threshold_off_values)
    print(
        f"Evaluating extractors={event_extractors}, {len(threshold_pairs)} threshold pairs, "
        f"{len(args.min_down_frames) * len(args.min_up_frames)} min-frame settings ..."
    )

    rows = []
    for event_extractor in event_extractors:
        if event_extractor == "trend_dual_trigger":
            missing_offsets = [track.name for track in tracks if track.pedal_offset_output is None]
            if missing_offsets:
                print(
                    "Skipping trend_dual_trigger because pedal_offset_output is unavailable "
                    f"for {len(missing_offsets)} tracks."
                )
                continue

        for pair_idx, (threshold_on, threshold_off) in enumerate(threshold_pairs, start=1):
            if event_extractor == "state_hysteresis":
                raw_spans_by_track = [
                    transcription_metrics.pedal_frame_output_to_raw_spans(
                        track.pedal_frame_output,
                        threshold_on=threshold_on,
                        threshold_off=threshold_off,
                    )
                    for track in tracks
                ]
                for min_down_frames, min_up_frames in product(args.min_down_frames, args.min_up_frames):
                    rows.append(
                        score_combination(
                            tracks,
                            raw_spans_by_track,
                            sec_per_frame=sec_per_frame,
                            event_extractor=event_extractor,
                            threshold_on=threshold_on,
                            threshold_off=threshold_off,
                            offset_threshold=np.nan,
                            min_on_delta=np.nan,
                            min_down_frames=min_down_frames,
                            min_up_frames=min_up_frames,
                        )
                    )
            else:
                for offset_threshold, min_on_delta in product(
                    args.offset_threshold_values,
                    args.min_on_delta_values,
                ):
                    raw_spans_by_track = [
                        transcription_metrics.pedal_frame_offset_outputs_to_raw_spans(
                            track.pedal_frame_output,
                            track.pedal_offset_output,
                            threshold_on=threshold_on,
                            threshold_off=threshold_off,
                            offset_threshold=offset_threshold,
                            min_on_delta=min_on_delta,
                        )
                        for track in tracks
                    ]
                    for min_down_frames, min_up_frames in product(
                        args.min_down_frames,
                        args.min_up_frames,
                    ):
                        rows.append(
                            score_combination(
                                tracks,
                                raw_spans_by_track,
                                sec_per_frame=sec_per_frame,
                                event_extractor=event_extractor,
                                threshold_on=threshold_on,
                                threshold_off=threshold_off,
                                offset_threshold=offset_threshold,
                                min_on_delta=min_on_delta,
                                min_down_frames=min_down_frames,
                                min_up_frames=min_up_frames,
                            )
                        )
            print(
                f"Finished {event_extractor} threshold pair {pair_idx}/{len(threshold_pairs)}: "
                f"on={threshold_on:.2f}, off={threshold_off:.2f}"
            )

    if len(rows) == 0:
        raise ValueError("No sweep rows were generated. Check requested event extractors and cached outputs.")

    df = pd.DataFrame(rows)
    best_row = choose_best_row(df, args.target_metric)
    baseline = df[
        (df["frame_head_event_extractor"] == "state_hysteresis")
        & (df["frame_head_threshold_on"] == 0.50)
        & (df["frame_head_threshold_off"] == 0.40)
        & (df["frame_head_min_down_frames"] == 3)
        & (df["frame_head_min_up_frames"] == 2)
    ]

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"Wrote sweep CSV: {output_csv}")
    print("Best row:", format_row_summary(best_row))
    if len(baseline) > 0:
        print("Baseline defaults:", format_row_summary(baseline.iloc[0]))
    else:
        print("Baseline defaults: not included in this sweep grid.")
    return df


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sweep pedal frame-head event extraction thresholds from saved evaluation JSON files."
    )
    parser.add_argument("--eval-dir", required=True, help="Directory containing *.output.json files.")
    parser.add_argument("--output-csv", required=True, help="Path to write the sweep CSV.")
    parser.add_argument("--target-metric", default=DEFAULT_TARGET_METRIC)
    parser.add_argument(
        "--event-extractors",
        type=parse_string_list,
        default=DEFAULT_EVENT_EXTRACTORS,
        help="Comma-separated event extractors: state_hysteresis,trend_dual_trigger.",
    )
    parser.add_argument(
        "--threshold-on-values",
        type=parse_float_list,
        default=DEFAULT_THRESHOLD_ON_VALUES,
        help="Comma-separated frame_head_threshold_on values.",
    )
    parser.add_argument(
        "--threshold-off-values",
        type=parse_float_list,
        default=DEFAULT_THRESHOLD_OFF_VALUES,
        help="Comma-separated frame_head_threshold_off values.",
    )
    parser.add_argument(
        "--offset-threshold-values",
        type=parse_float_list,
        default=DEFAULT_OFFSET_THRESHOLD_VALUES,
        help="Comma-separated frame_head_offset_threshold values for trend_dual_trigger.",
    )
    parser.add_argument(
        "--min-on-delta-values",
        type=parse_float_list,
        default=DEFAULT_MIN_ON_DELTA_VALUES,
        help="Comma-separated frame_head_min_on_delta values for trend_dual_trigger.",
    )
    parser.add_argument(
        "--min-down-frames",
        type=parse_int_list,
        default=DEFAULT_MIN_DOWN_FRAMES,
        help="Comma-separated frame_head_min_down_frames values.",
    )
    parser.add_argument(
        "--min-up-frames",
        type=parse_int_list,
        default=DEFAULT_MIN_UP_FRAMES,
        help="Comma-separated frame_head_min_up_frames values.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run_sweep(args)


if __name__ == "__main__":
    main()
