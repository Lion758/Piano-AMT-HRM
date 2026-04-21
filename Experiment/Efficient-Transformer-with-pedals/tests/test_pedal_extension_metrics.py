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

from data.pedal_extension_utils import pedal_events_to_spans
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


def test_cal_pedal_extended_note_metrics_uses_tsv_semantics_and_metric_names(monkeypatch):
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

    note_call = {}
    velocity_call = {}

    def _capture_note_metrics(reference_intervals, reference_pitches, estimated_intervals, estimated_pitches, **kwargs):
        note_call["reference_intervals"] = np.array(reference_intervals, copy=True)
        note_call["estimated_intervals"] = np.array(estimated_intervals, copy=True)
        note_call["reference_pitches"] = np.array(reference_pitches, copy=True)
        note_call["estimated_pitches"] = np.array(estimated_pitches, copy=True)
        return 0.11, 0.22, 0.33, 0.0

    def _capture_velocity_metrics(reference_intervals, reference_pitches, reference_velocities, estimated_intervals, estimated_pitches, estimated_velocities, **kwargs):
        velocity_call["reference_intervals"] = np.array(reference_intervals, copy=True)
        velocity_call["estimated_intervals"] = np.array(estimated_intervals, copy=True)
        velocity_call["reference_velocities"] = np.array(reference_velocities, copy=True)
        velocity_call["estimated_velocities"] = np.array(estimated_velocities, copy=True)
        return 0.44, 0.55, 0.66, 0.0

    monkeypatch.setattr(transcription_metrics, "evaluate_notes", _capture_note_metrics)
    monkeypatch.setattr(transcription_metrics, "evaluate_notes_with_velocity", _capture_velocity_metrics)

    metric_dict, metric_inputs = transcription_metrics.cal_pedal_extended_note_metrics(
        output_notes,
        output_pedal_events,
        tsv_df,
        piece_end_time=1.5,
    )

    expected_intervals = np.array([[0.0, 1.5], [1.0, 1.5]], dtype=np.float32)
    np.testing.assert_allclose(metric_inputs["gt_interval_ext"], expected_intervals)
    np.testing.assert_allclose(metric_inputs["out_interval_ext"], expected_intervals)
    np.testing.assert_allclose(note_call["reference_intervals"], expected_intervals)
    np.testing.assert_allclose(note_call["estimated_intervals"], expected_intervals)
    np.testing.assert_allclose(velocity_call["reference_intervals"], expected_intervals)
    np.testing.assert_allclose(velocity_call["estimated_intervals"], expected_intervals)

    assert metric_dict == {
        "note+offset_precision_pedal_extended": 0.11,
        "note+offset_recall_pedal_extended": 0.22,
        "note+offset_f1_pedal_extended": 0.33,
        "note+offset+velocity_precision_pedal_extended": 0.44,
        "note+offset+velocity_recall_pedal_extended": 0.55,
        "note+offset+velocity_f1_pedal_extended": 0.66,
    }
