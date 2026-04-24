from pathlib import Path
import sys

import pretty_midi


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from Midi_Analysis.error_analysis import ErrorAnalysis
from Midi_Analysis.json_summarization import JSONSummarization
from Midi_Analysis.midi_parser import MIDIParser


def _note(pitch: int, start: float, end: float, velocity: int = 80) -> dict:
    return {
        "pitch": pitch,
        "start": start,
        "end": end,
        "duration": end - start,
        "velocity": velocity,
        "track_id": 0,
    }


def _pedaling_payload(segments: list[dict], coverage_ratio: float = 0.0) -> dict:
    raw_events = []
    for seg in segments:
        raw_events.append({"time": seg["start"], "value": seg.get("start_value", 80)})
        raw_events.append({"time": seg["end"], "value": seg.get("end_value", 0)})
    return {
        "raw_events": raw_events,
        "segments": segments,
        "summary": {
            "pedal_coverage_ratio": coverage_ratio,
        },
    }


def test_midi_parser_extracts_cc64_events_and_segments(tmp_path):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="Piano")
    instrument.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=3.0))
    instrument.control_changes.extend(
        [
            pretty_midi.ControlChange(number=64, value=0, time=0.0),
            pretty_midi.ControlChange(number=64, value=70, time=0.5),
            pretty_midi.ControlChange(number=64, value=100, time=1.0),
            pretty_midi.ControlChange(number=64, value=63, time=1.5),
            pretty_midi.ControlChange(number=64, value=80, time=2.0),
            pretty_midi.ControlChange(number=64, value=0, time=2.5),
        ]
    )
    midi.instruments.append(instrument)

    midi_path = tmp_path / "pedal.mid"
    midi.write(str(midi_path))

    parsed = MIDIParser().parse_midi(str(midi_path))
    pedaling = parsed["pedaling"]

    assert pedaling["available"] is True
    assert len(pedaling["raw_events"]) == 6
    assert len(pedaling["events"]) == 4
    assert len(pedaling["segments"]) == 2

    first_segment = pedaling["segments"][0]
    assert first_segment["start"] == 0.5
    assert first_segment["end"] == 1.5
    assert round(first_segment["duration"], 3) == 1.0
    assert first_segment["max_value"] == 100

    summary = pedaling["summary"]
    assert summary["pedal_down_count"] == 2
    assert summary["pedal_up_count"] == 2
    assert round(summary["average_hold_duration"], 3) == 0.75


def test_reference_pedal_analysis_and_summary_surface_pedal_metrics():
    reference_data = {
        "notes": [
            _note(60, 0.0, 0.5),
            _note(62, 1.0, 1.5),
            _note(64, 2.0, 2.5),
            _note(65, 3.0, 3.5),
            _note(67, 4.0, 4.5),
        ],
        "pedaling": _pedaling_payload(
            [
                {"start": 1.0, "end": 2.0, "duration": 1.0, "start_value": 80, "end_value": 0},
                {"start": 3.0, "end": 3.5, "duration": 0.5, "start_value": 82, "end_value": 0},
            ],
            coverage_ratio=0.30,
        ),
        "harmony": {
            "chords": [
                {"start_time": 1.0, "end_time": 2.0, "chord_name": "C:maj"},
                {"start_time": 2.1, "end_time": 3.0, "chord_name": "G:maj"},
            ]
        },
        "structure": {
            "phrases": [
                {"phrase_id": 1, "start_time": 0.0, "end_time": 2.0},
                {"phrase_id": 2, "start_time": 2.0, "end_time": 4.5},
            ]
        },
        "total_duration": 5.0,
    }
    performance_data = {
        "notes": [
            _note(60, 0.0, 0.5),
            _note(62, 1.0, 1.5),
            _note(64, 2.0, 2.5),
            _note(65, 3.0, 3.5),
            _note(67, 4.0, 4.5),
        ],
        "pedaling": _pedaling_payload(
            [
                {"start": 1.05, "end": 2.25, "duration": 1.2, "start_value": 84, "end_value": 0},
                {"start": 4.0, "end": 4.4, "duration": 0.4, "start_value": 76, "end_value": 0},
            ],
            coverage_ratio=0.32,
        ),
        "harmony": {
            "chords": [
                {"start_time": 1.0, "end_time": 2.0, "chord_name": "C:maj"},
                {"start_time": 2.1, "end_time": 3.0, "chord_name": "G:maj"},
            ]
        },
        "structure": {
            "phrases": [
                {"phrase_id": 1, "start_time": 0.0, "end_time": 2.0},
                {"phrase_id": 2, "start_time": 2.0, "end_time": 4.5},
            ]
        },
        "total_duration": 5.0,
    }
    alignment = [
        {
            "reference_note": reference_data["notes"][i],
            "performance_note": performance_data["notes"][i],
            "time_difference": 0.0,
            "pitch_difference": 0,
            "velocity_difference": 0,
            "alignment_confidence": 1.0,
            "error_type": "none",
        }
        for i in range(len(reference_data["notes"]))
    ]

    analysis = ErrorAnalysis(
        {
            "reference": reference_data,
            "performance": performance_data,
            "alignment": alignment,
        }
    ).analyze_performance()

    pedaling = analysis["metrics"]["pedaling"]
    assert pedaling["pedal_analysis_available"] is True
    assert pedaling["missed_pedals"] == 1
    assert pedaling["extra_pedals"] == 1
    assert pedaling["late_release_count"] >= 1
    assert pedaling["harmonic_blur_count"] >= 1
    assert pedaling["phrase_boundary_clearance_issues"] >= 1
    assert analysis["pedaling_recommendations"]

    summary = JSONSummarization(
        {
            "reference_data": reference_data,
            "performance_data": performance_data,
            "alignment": alignment,
            "alignment_statistics": {},
            "error_analysis": analysis,
        }
    ).create_summary()

    assert "pedaling" in summary
    assert summary["pedaling"]["available"] is True
    assert summary["pedaling"]["metrics"]["missed_pedals"] == 1
    assert summary["pedaling"]["practice_suggestions"]
    assert summary["error_analysis_summary"]["pedaling"]["priority"] in {"medium", "high"}
