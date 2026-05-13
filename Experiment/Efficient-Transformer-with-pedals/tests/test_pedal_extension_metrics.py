from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules.setdefault("music21", types.ModuleType("music21"))
sys.modules.setdefault("pretty_midi", types.ModuleType("pretty_midi"))

symusic_module = sys.modules.setdefault("symusic", types.ModuleType("symusic"))
if not hasattr(symusic_module, "Score"):
    symusic_module.Score = type("Score", (), {})
if not hasattr(symusic_module, "TimeUnit"):
    symusic_module.TimeUnit = types.SimpleNamespace(second="second")

utils_module = sys.modules.setdefault("utils", types.ModuleType("utils"))
if not hasattr(utils_module, "__path__"):
    utils_module.__path__ = []
pianoroll_parser_module = types.ModuleType("utils.pianoroll_parser")
pianoroll_parser_module.get_notes_with_pedal = lambda midi_path: (None, None)
sys.modules.setdefault("utils.pianoroll_parser", pianoroll_parser_module)

mir_eval_module = sys.modules.setdefault("mir_eval", types.ModuleType("mir_eval"))
mir_eval_util_module = types.ModuleType("mir_eval.util")
mir_eval_util_module.midi_to_hz = lambda values: np.asarray(values, dtype=np.float32)
sys.modules.setdefault("mir_eval.util", mir_eval_util_module)
mir_eval_transcription_module = types.ModuleType("mir_eval.transcription")
mir_eval_transcription_module.precision_recall_f1_overlap = lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0)
sys.modules.setdefault("mir_eval.transcription", mir_eval_transcription_module)
mir_eval_transcription_velocity_module = types.ModuleType("mir_eval.transcription_velocity")
mir_eval_transcription_velocity_module.precision_recall_f1_overlap = lambda *args, **kwargs: (0.0, 0.0, 0.0, 0.0)
sys.modules.setdefault("mir_eval.transcription_velocity", mir_eval_transcription_velocity_module)
mir_eval_multipitch_module = types.ModuleType("mir_eval.multipitch")
mir_eval_multipitch_module.evaluate = lambda *args, **kwargs: {}
sys.modules.setdefault("mir_eval.multipitch", mir_eval_multipitch_module)
mir_eval_module.util = mir_eval_util_module
mir_eval_module.transcription = mir_eval_transcription_module
mir_eval_module.transcription_velocity = mir_eval_transcription_velocity_module
mir_eval_module.multipitch = mir_eval_multipitch_module

from data.pedal_extension_utils import (
    extend_notes_with_reference_pedal_spans,
    pedal_events_to_spans,
)
from data.symbolic_music_tokenizer import extend_offsets_with_pedals
import metrics.transcription_metrics as transcription_metrics


def _make_note(pitch, onset, offset, velocity=80):
    return {
        "pitch": pitch,
        "onset": float(onset),
        "offset": float(offset),
        "duration": float(offset - onset),
        "velocity": velocity,
        "staff": 0,
    }


def test_extend_offsets_with_pedals_keeps_same_pitch_repeats_uncapped():
    notes = [
        _make_note(60, 0.0, 0.5, velocity=90),
        _make_note(60, 1.0, 1.1, velocity=70),
    ]
    pedal_events = [
        {"time": 0.25, "type": "PedalOn"},
        {"time": 2.0, "type": "PedalOff"},
    ]

    extended_notes = extend_offsets_with_pedals(notes, pedal_events)
    capped_notes = extend_offsets_with_pedals(notes, pedal_events, next_onset_cap=True)

    assert [note["offset"] for note in extended_notes] == [2.0, 2.0]
    assert [note["offset"] for note in capped_notes] == [1.0, 2.0]


def test_reference_pedal_extension_caps_same_pitch_repeats_inside_span():
    notes = [
        _make_note(60, 0.0, 0.5, velocity=90),
        _make_note(60, 1.0, 1.1, velocity=70),
    ]
    pedal_spans = [{"onset": 0.25, "offset": 2.0}]

    extended_notes = extend_notes_with_reference_pedal_spans(notes, pedal_spans)

    assert [note["offset"] for note in extended_notes] == [1.0, 2.0]


def test_reference_pedal_extension_does_not_cap_different_pitches():
    notes = [
        _make_note(60, 0.0, 0.5),
        _make_note(64, 1.0, 1.1),
    ]
    pedal_spans = [{"onset": 0.25, "offset": 2.0}]

    extended_notes = extend_notes_with_reference_pedal_spans(notes, pedal_spans)

    assert [note["offset"] for note in extended_notes] == [2.0, 2.0]


def test_reference_pedal_extension_does_not_shorten_notes_outside_pedal_span():
    notes = [
        _make_note(60, 0.0, 2.5),
        _make_note(60, 1.0, 1.1),
    ]
    pedal_spans = [{"onset": 0.25, "offset": 2.0}]

    extended_notes = extend_notes_with_reference_pedal_spans(notes, pedal_spans)

    assert [note["offset"] for note in extended_notes] == [2.5, 2.0]


def test_reference_pedal_extension_uses_strict_pedal_boundaries():
    notes = [
        _make_note(60, 0.0, 0.25),
        _make_note(61, 0.5, 1.0),
        _make_note(62, 0.5, 0.75),
    ]
    pedal_spans = [{"onset": 0.25, "offset": 1.0}]

    extended_notes = extend_notes_with_reference_pedal_spans(notes, pedal_spans)

    assert [note["offset"] for note in extended_notes] == [0.25, 1.0, 1.0]


def test_extend_offsets_with_pedals_closes_unmatched_final_pedal_on():
    notes = [_make_note(64, 0.0, 0.75)]
    pedal_events = [{"time": 0.5, "type": "PedalOn"}]

    pedal_spans = pedal_events_to_spans(pedal_events, piece_end_time=3.0)
    extended_notes = extend_offsets_with_pedals(notes, pedal_events, piece_end_time=3.0)

    assert pedal_spans == [{"onset": 0.5, "offset": 3.0}]
    assert extended_notes[0]["offset"] == 3.0


def test_same_time_pedal_off_then_on_preserves_both_spans_and_boundary_extension():
    pedal_events = [
        {"time": 0.0, "type": "PedalOn"},
        {"time": 1.0, "type": "PedalOff"},
        {"time": 1.0, "type": "PedalOn"},
        {"time": 2.0, "type": "PedalOff"},
    ]
    notes = [_make_note(67, 0.25, 1.0)]

    pedal_spans = pedal_events_to_spans(pedal_events, piece_end_time=2.0)
    extended_notes = extend_offsets_with_pedals(notes, pedal_events, piece_end_time=2.0)

    assert pedal_spans == [
        {"onset": 0.0, "offset": 1.0},
        {"onset": 1.0, "offset": 2.0},
    ]
    assert extended_notes[0]["offset"] == 2.0


def test_float32_pedal_min_duration_can_collapse_at_long_times():
    onset = 50.0
    offset = onset + transcription_metrics.PEDAL_MIN_DURATION

    assert np.float32(offset) == np.float32(onset)
    assert np.float64(offset) > np.float64(onset)


def test_pedal_spans_to_intervals_keep_long_time_boundary_spans_strictly_positive():
    pedal_events = [
        {"time": 49.5, "type": "PedalOn"},
        {"time": 50.0, "type": "PedalOff"},
        {"time": 50.0, "type": "PedalOn"},
    ]

    pedal_spans = transcription_metrics.pedal_events_to_spans(pedal_events, piece_end_time=50.0)
    intervals = transcription_metrics._pedal_spans_to_intervals(pedal_spans)

    assert len(pedal_spans) == 2
    assert np.float32(pedal_spans[-1]["offset"]) == np.float32(pedal_spans[-1]["onset"])
    assert intervals.dtype == np.float64
    assert np.all(intervals[:, 1] > intervals[:, 0])


def test_cal_pedal_metrics_handles_long_time_boundary_spans(monkeypatch):
    pedal_events = [
        {"time": 49.5, "type": "PedalOn"},
        {"time": 50.0, "type": "PedalOff"},
        {"time": 50.0, "type": "PedalOn"},
    ]
    interval_metric_calls = []

    def _validate_interval_metrics(reference_intervals, reference_pitches, estimated_intervals, estimated_pitches, **kwargs):
        reference_intervals = np.asarray(reference_intervals)
        estimated_intervals = np.asarray(estimated_intervals)
        assert reference_intervals.dtype == np.float64
        assert estimated_intervals.dtype == np.float64
        assert np.all(reference_intervals[:, 1] > reference_intervals[:, 0])
        assert np.all(estimated_intervals[:, 1] > estimated_intervals[:, 0])
        interval_metric_calls.append(kwargs)
        return 1.0, 1.0, 1.0, 1.0

    monkeypatch.setattr(transcription_metrics, "evaluate_notes", _validate_interval_metrics)

    metric_dict, output_pedal_spans, target_pedal_spans = transcription_metrics.cal_pedal_metrics(
        pedal_events,
        pedal_events,
        piece_end_time=50.0,
    )

    assert len(interval_metric_calls) == 2
    assert metric_dict["pedal_f1"] == 1.0
    assert metric_dict["pedal+offset_f1"] == 1.0
    assert output_pedal_spans == target_pedal_spans


def test_cal_pedal_extended_note_metrics_uses_reference_style_and_diagnostic_names(monkeypatch):
    output_notes = [
        _make_note(60, 0.0, 0.5, velocity=90),
        _make_note(60, 1.0, 1.1, velocity=70),
    ]
    output_pedal_events = [
        {"time": 0.25, "type": "PedalOn"},
        {"time": 1.5, "type": "PedalOff"},
    ]
    tsv_df = pd.DataFrame(
        {
            "type": ["note", "note", "PedalOn", "PedalOff"],
            "onset_sec": [0.0, 1.0, 0.25, 1.5],
            "offset_sec": [1.5, 1.5, 0.25, 1.5],
            "offset_sec_truth": [0.5, 1.1, 0.25, 1.5],
            "pitch": [60, 60, 0, 0],
            "velocity": [90, 70, 0, 0],
        }
    )

    note_calls = []
    velocity_calls = []

    def _capture_note_metrics(reference_intervals, reference_pitches, estimated_intervals, estimated_pitches, **kwargs):
        note_calls.append({
            "reference_intervals": np.array(reference_intervals, copy=True),
            "estimated_intervals": np.array(estimated_intervals, copy=True),
            "reference_pitches": np.array(reference_pitches, copy=True),
            "estimated_pitches": np.array(estimated_pitches, copy=True),
        })
        return 0.11, 0.22, 0.33, 0.0

    def _capture_velocity_metrics(reference_intervals, reference_pitches, reference_velocities, estimated_intervals, estimated_pitches, estimated_velocities, **kwargs):
        velocity_calls.append({
            "reference_intervals": np.array(reference_intervals, copy=True),
            "estimated_intervals": np.array(estimated_intervals, copy=True),
            "reference_velocities": np.array(reference_velocities, copy=True),
            "estimated_velocities": np.array(estimated_velocities, copy=True),
        })
        return 0.44, 0.55, 0.66, 0.0

    monkeypatch.setattr(transcription_metrics, "evaluate_notes", _capture_note_metrics)
    monkeypatch.setattr(transcription_metrics, "evaluate_notes_with_velocity", _capture_velocity_metrics)

    metric_dict, metric_inputs = transcription_metrics.cal_pedal_extended_note_metrics(
        output_notes,
        output_pedal_events,
        tsv_df,
        piece_end_time=1.5,
    )

    expected_reference_intervals = np.array([[0.0, 1.0], [1.0, 1.5]], dtype=np.float32)
    expected_uncapped_intervals = np.array([[0.0, 1.5], [1.0, 1.5]], dtype=np.float32)
    np.testing.assert_allclose(metric_inputs["gt_interval_ext"], expected_reference_intervals)
    np.testing.assert_allclose(metric_inputs["out_interval_ext"], expected_reference_intervals)
    np.testing.assert_allclose(metric_inputs["diagnostic_gt_interval_ext_uncapped"], expected_uncapped_intervals)
    np.testing.assert_allclose(metric_inputs["diagnostic_out_interval_ext_uncapped"], expected_uncapped_intervals)
    assert metric_inputs["pedal_extended_target_source"] == "offset_sec_truth+pedal_events"

    assert len(note_calls) == 2
    assert len(velocity_calls) == 2
    np.testing.assert_allclose(note_calls[0]["reference_intervals"], expected_reference_intervals)
    np.testing.assert_allclose(note_calls[0]["estimated_intervals"], expected_reference_intervals)
    np.testing.assert_allclose(velocity_calls[0]["reference_intervals"], expected_reference_intervals)
    np.testing.assert_allclose(velocity_calls[0]["estimated_intervals"], expected_reference_intervals)
    np.testing.assert_allclose(note_calls[1]["reference_intervals"], expected_uncapped_intervals)
    np.testing.assert_allclose(note_calls[1]["estimated_intervals"], expected_uncapped_intervals)
    np.testing.assert_allclose(velocity_calls[1]["reference_intervals"], expected_uncapped_intervals)
    np.testing.assert_allclose(velocity_calls[1]["estimated_intervals"], expected_uncapped_intervals)

    assert metric_dict == {
        "note+offset_precision_pedal_extended": 0.11,
        "note+offset_recall_pedal_extended": 0.22,
        "note+offset_f1_pedal_extended": 0.33,
        "note+offset+velocity_precision_pedal_extended": 0.44,
        "note+offset+velocity_recall_pedal_extended": 0.55,
        "note+offset+velocity_f1_pedal_extended": 0.66,
        "diagnostic_note+offset_precision_pedal_extended_uncapped": 0.11,
        "diagnostic_note+offset_recall_pedal_extended_uncapped": 0.22,
        "diagnostic_note+offset_f1_pedal_extended_uncapped": 0.33,
        "diagnostic_note+offset+velocity_precision_pedal_extended_uncapped": 0.44,
        "diagnostic_note+offset+velocity_recall_pedal_extended_uncapped": 0.55,
        "diagnostic_note+offset+velocity_f1_pedal_extended_uncapped": 0.66,
    }


def test_pedal_extended_metric_inputs_fall_back_to_cached_offsets_for_older_tsv():
    output_notes = [
        _make_note(60, 0.0, 0.5, velocity=90),
        _make_note(60, 1.0, 1.1, velocity=70),
    ]
    output_pedal_events = [
        {"time": 0.25, "type": "PedalOn"},
        {"time": 1.5, "type": "PedalOff"},
    ]
    tsv_df = pd.DataFrame(
        {
            "type": ["note", "note"],
            "onset_sec": [0.0, 1.0],
            "offset_sec": [1.5, 1.5],
            "pitch": [60, 60],
            "velocity": [90, 70],
        }
    )

    metric_inputs = transcription_metrics.build_pedal_extended_note_metric_inputs(
        output_notes,
        output_pedal_events,
        tsv_df,
        piece_end_time=1.5,
    )

    np.testing.assert_allclose(
        metric_inputs["gt_interval_ext"],
        np.array([[0.0, 1.5], [1.0, 1.5]], dtype=np.float32),
    )
    np.testing.assert_allclose(
        metric_inputs["out_interval_ext"],
        np.array([[0.0, 1.0], [1.0, 1.5]], dtype=np.float32),
    )
    assert metric_inputs["pedal_extended_target_source"] == "offset_sec"


def test_reference_pedal_metrics_use_paper_scorer_arguments(monkeypatch):
    calls = {}

    def _capture_reference_metric(**kwargs):
        calls.update({key: np.array(value, copy=True) if key.endswith("intervals") or key.endswith("pitches") else value for key, value in kwargs.items()})
        return 0.12, 0.34, 0.56, 0.0

    monkeypatch.setattr(transcription_metrics, "evaluate_notes", _capture_reference_metric)

    pedal_events = [
        {"time": 0.1, "type": "PedalOn"},
        {"time": 1.2, "type": "PedalOff"},
    ]
    metric_dict, output_pedal_spans, reference_pedal_spans = transcription_metrics.cal_reference_pedal_metrics(
        pedal_events,
        pedal_events,
        piece_end_time=2.0,
    )

    assert metric_dict == {
        "pedal_precision": 0.12,
        "pedal_recall": 0.34,
        "pedal_f1": 0.56,
    }
    assert output_pedal_spans == reference_pedal_spans
    np.testing.assert_allclose(calls["ref_intervals"], np.array([[0.1, 1.2]], dtype=np.float64))
    np.testing.assert_allclose(calls["est_intervals"], np.array([[0.1, 1.2]], dtype=np.float64))
    np.testing.assert_allclose(calls["ref_pitches"], np.ones(1))
    np.testing.assert_allclose(calls["est_pitches"], np.ones(1))
    assert calls["onset_tolerance"] == 0.2
    assert calls["offset_ratio"] == 0.2
    assert calls["offset_min_tolerance"] == 0.05


def test_reference_pedal_events_from_dataframe_preserves_exact_times_and_repedal_order():
    tsv_df = pd.DataFrame(
        {
            "type": ["PedalOn", "note", "PedalOn", "PedalOff"],
            "onset_sec": [1.0, 0.5, 2.0, 1.0],
        }
    )

    pedal_events = transcription_metrics.reference_pedal_events_from_dataframe(tsv_df)

    assert pedal_events == [
        {"time": 1.0, "type": "PedalOff"},
        {"time": 1.0, "type": "PedalOn"},
        {"time": 2.0, "type": "PedalOn"},
    ]


def test_pedal_frame_metrics_threshold_and_trim_padded_tail():
    data_list = [
        {
            "frame_offsets": 0,
            "total_frames": 6,
            "pedal_frame_output": [0.6, 0.5, 0.4, 0.9],
            "pedal_frame_target": [1, 1, 0, 1],
        },
        {
            "frame_offsets": 4,
            "total_frames": 6,
            "pedal_frame_output": [0.2, 0.8, 0.9, 0.9],
            "pedal_frame_target": [0, 1, 1, 1],
        },
    ]

    pedal_frame_outputs, pedal_frame_targets = transcription_metrics.collect_trimmed_pedal_frame_arrays(data_list)
    np.testing.assert_allclose(pedal_frame_outputs, np.array([0.6, 0.5, 0.4, 0.9, 0.2, 0.8], dtype=np.float32))
    np.testing.assert_allclose(pedal_frame_targets, np.array([1, 1, 0, 1, 0, 1], dtype=np.float32))

    metric_dict = transcription_metrics.cal_pedal_frame_metrics(pedal_frame_outputs, pedal_frame_targets)

    assert metric_dict == {
        "pedal_frame_precision": 1.0,
        "pedal_frame_recall": 0.75,
        "pedal_frame_f1": 6 / 7,
    }


def test_pedal_frame_output_to_events_hysteresis_merge_filter_and_empty():
    probs = [0.0, 0.0, 0.6, 0.7, 0.8, 0.7, 0.3, 0.6, 0.9, 0.39, 0.1, 0.0]
    raw_spans = transcription_metrics.pedal_frame_output_to_raw_spans(
        probs,
        threshold_on=0.5,
        threshold_off=0.4,
    )

    events = transcription_metrics.pedal_frame_output_to_events(
        probs,
        sec_per_frame=0.02,
        threshold_on=0.5,
        threshold_off=0.4,
        min_pedal_down_frames=3,
        min_pedal_up_frames=2,
    )

    assert raw_spans == [(2, 6), (7, 9)]
    assert events == [
        {"time": 0.04, "type": "PedalOn"},
        {"time": 0.18, "type": "PedalOff"},
    ]
    assert transcription_metrics.pedal_frame_spans_to_events(
        raw_spans,
        sec_per_frame=0.02,
        min_pedal_down_frames=3,
        min_pedal_up_frames=2,
    ) == events
    assert transcription_metrics.pedal_frame_output_to_events(
        [0.0, 0.6, 0.6, 0.1],
        sec_per_frame=0.02,
        min_pedal_down_frames=3,
    ) == []
    assert transcription_metrics.pedal_frame_output_to_events([], sec_per_frame=0.02) == []


def test_collect_trimmed_frame_arrays_is_generic_for_pedal_boundary_heads():
    data_list = [
        {
            "frame_offsets": 0,
            "total_frames": 5,
            "pedal_onset_output": [0.9, 0.2, 0.1],
            "pedal_onset_target": [1.0, 0.5, 0.0],
            "pedal_offset_output": [0.1, 0.8, 0.2],
            "pedal_offset_target": [0.0, 1.0, 0.5],
        },
        {
            "frame_offsets": 3,
            "total_frames": 5,
            "pedal_onset_output": [0.4, 0.7, 0.9],
            "pedal_onset_target": [0.0, 1.0, 1.0],
            "pedal_offset_output": [0.6, 0.3, 0.2],
            "pedal_offset_target": [1.0, 0.0, 0.0],
        },
    ]

    onset_outputs, onset_targets = transcription_metrics.collect_trimmed_frame_arrays(
        data_list,
        "pedal_onset_output",
        "pedal_onset_target",
    )
    offset_outputs, offset_targets = transcription_metrics.collect_trimmed_frame_arrays(
        data_list,
        "pedal_offset_output",
        "pedal_offset_target",
    )

    np.testing.assert_allclose(onset_outputs, np.array([0.9, 0.2, 0.1, 0.4, 0.7], dtype=np.float32))
    np.testing.assert_allclose(onset_targets, np.array([1.0, 0.5, 0.0, 0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(offset_outputs, np.array([0.1, 0.8, 0.2, 0.6, 0.3], dtype=np.float32))
    np.testing.assert_allclose(offset_targets, np.array([0.0, 1.0, 0.5, 1.0, 0.0], dtype=np.float32))


def test_pedal_boundary_frame_metrics_use_central_soft_target_only():
    metric_dict = transcription_metrics.cal_binary_frame_metrics(
        "pedal_onset_frame",
        frame_output=[0.6, 0.6, 0.7, 0.9],
        frame_target=[1.0, 0.946, 0.607, 0.0],
        target_threshold=0.99,
    )

    assert metric_dict == {
        "pedal_onset_frame_precision": 1 / 4,
        "pedal_onset_frame_recall": 1.0,
        "pedal_onset_frame_f1": 0.4,
    }


def test_choose_pedal_event_list_honors_source_and_fallback():
    decoder_events = [{"time": 0.1, "type": "PedalOn"}]
    frame_head_events = [{"time": 0.08, "type": "PedalOn"}]

    events, source_used = transcription_metrics.choose_pedal_event_list(
        decoder_events,
        frame_head_events,
        "decoder",
    )
    assert events is decoder_events
    assert source_used == "decoder"

    events, source_used = transcription_metrics.choose_pedal_event_list(
        decoder_events,
        frame_head_events,
        "frame_head",
    )
    assert events is frame_head_events
    assert source_used == "frame_head"

    events, source_used = transcription_metrics.choose_pedal_event_list(
        decoder_events,
        None,
        "frame_head",
    )
    assert events is decoder_events
    assert source_used == "decoder_fallback_frame_head_unavailable"
