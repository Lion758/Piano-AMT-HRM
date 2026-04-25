from pathlib import Path
import sys
import time

import pretty_midi


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from Midi_Analysis import quick_analyze
from Midi_Analysis.analyzer import MIDIAnalyzer
from Midi_Analysis.midi_parser import MIDIParser
from Midi_Analysis.simple_phrase_segmentation import SimplePhraseSegmenter
from Midi_Analysis.solo_performance_analysis import SoloPerformanceAnalysis


def _write_midi(tmp_path, filename, notes, pedal_events=None):
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0, name="Piano")
    for note in notes:
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(note["velocity"]),
                pitch=int(note["pitch"]),
                start=float(note["start"]),
                end=float(note["end"]),
            )
        )
    for event in pedal_events or []:
        instrument.control_changes.append(
            pretty_midi.ControlChange(number=64, value=int(event["value"]), time=float(event["time"]))
        )
    midi.instruments.append(instrument)
    path = tmp_path / filename
    midi.write(str(path))
    return path


def _note(pitch, start, duration=0.3, velocity=80):
    return {
        "pitch": int(pitch),
        "start": float(start),
        "end": float(start + duration),
        "velocity": int(velocity),
    }


def _feedback_strings(result):
    strings = []
    analysis = result.get("performance_analysis", {})
    for observation in analysis.get("observations", []):
        strings.append(str(observation.get("interpretation", "")))
        strings.append(str(observation.get("student_friendly_feedback", "")))
    strings.extend(str(item) for item in analysis.get("strengths", []))
    strings.extend(str(item) for item in result.get("practice_recommendations", []))
    for recommendation in result.get("gpt_ready_summary", {}).get("practice_recommendations", []):
        strings.append(str(recommendation.get("focus", "")))
        strings.append(str(recommendation.get("why", "")))
        strings.append(str(recommendation.get("exercise", "")))
    return [text.casefold() for text in strings if text]


def _assert_no_forbidden_solo_feedback(result):
    forbidden_terms = [
        "wrong note",
        "wrong notes",
        "incorrect rhythm",
        "bad fingering",
        "wrist",
        "posture",
        "grade",
        "performance score",
        "pedaling is wrong",
    ]
    combined = "\n".join(_feedback_strings(result))
    for term in forbidden_terms:
        assert term not in combined


def test_simple_phrase_segmenter_empty_schema_is_stable():
    result = SimplePhraseSegmenter({"notes": []}).segment()
    assert result["phrases"] == []
    assert result["phrase_count"] == 0
    assert result["segmentation_method"] == "simple_gap_and_time"
    assert result["segmentation_confidence"] == 0.0
    assert result["limitations"]


def test_simple_phrase_segmenter_can_fallback_to_fixed_windows():
    parsed_data = {
        "notes": [
            {"pitch": 60, "start": 0.0, "end": 0.4, "duration": 0.4, "velocity": 70},
            {"pitch": 62, "start": 7.0, "end": 7.4, "duration": 0.4, "velocity": 72},
            {"pitch": 64, "start": 14.0, "end": 14.4, "duration": 0.4, "velocity": 74},
        ],
        "total_duration": 20.0,
    }
    result = SimplePhraseSegmenter(parsed_data).segment()
    assert result["phrase_count"] >= 2
    assert result["segmentation_confidence"] == 0.45


def test_solo_analysis_returns_observation_only_structure_and_no_scoring(tmp_path):
    notes = []
    start = 0.0
    iois = [0.35] * 10 + [0.8, 0.15, 0.7, 0.2, 0.65, 0.2, 0.35, 0.35] + [0.35] * 8
    velocities = [82] * 8 + [55] * 5 + [90] * 5 + [78] * 9
    for index, ioi in enumerate(iois):
        notes.append(_note(60 + (index % 5), start, duration=0.22 if index % 3 else 0.12, velocity=velocities[index]))
        start += ioi
    notes.append(_note(72, start + 1.25, duration=0.4, velocity=68))

    pedal_events = [
        {"time": 0.6, "value": 90},
        {"time": 5.3, "value": 0},
        {"time": 5.5, "value": 100},
        {"time": 10.8, "value": 0},
    ]
    midi_path = _write_midi(tmp_path, "solo_observation.mid", notes, pedal_events=pedal_events)

    result = MIDIAnalyzer().analyze_solo_performance(str(midi_path))

    assert result["analysis_type"] == "solo_performance"
    assert "musical_structure" in result
    assert "performance_analysis" in result
    assert "gpt_ready_summary" in result
    assert "performance_score" not in result
    assert "grade" not in result
    assert "note_accuracy" not in result

    analysis = result["performance_analysis"]
    assert analysis["mode"] == "solo_no_reference"
    assert analysis["no_score"] is True
    assert analysis["observations"]
    assert analysis["summary_statistics"]["phrase_count"] == result["musical_structure"]["phrase_count"]
    assert any(observation["category"] == "timing" for observation in analysis["observations"])
    assert any(observation["category"] == "continuity" for observation in analysis["observations"])

    summary = result["gpt_ready_summary"]
    assert summary["metadata"]["analysis_type"] == "solo_performance"
    assert summary["metadata"]["no_reference_provided"] is True
    assert summary["metadata"]["no_score"] is True
    assert summary["limitations"]
    assert isinstance(summary["practice_recommendations"], list)
    assert summary["practice_recommendations"]
    assert isinstance(summary["practice_recommendations"][0], dict)

    _assert_no_forbidden_solo_feedback(result)


def test_quick_analyze_returns_rich_solo_payload_and_pedal_aliases(tmp_path):
    motif = [
        (60, 0.00, 88),
        (62, 0.30, 86),
        (64, 0.60, 84),
        (65, 0.90, 86),
        (67, 1.20, 88),
        (60, 2.00, 62),
        (62, 2.30, 60),
        (64, 2.60, 58),
        (65, 2.90, 60),
        (67, 3.20, 62),
        (60, 4.00, 90),
        (62, 4.30, 88),
        (64, 4.60, 86),
        (65, 4.90, 88),
        (67, 5.20, 90),
    ]
    notes = [_note(pitch, start, duration=0.22, velocity=velocity) for pitch, start, velocity in motif]
    pedal_events = [
        {"time": 0.4, "value": 100},
        {"time": 1.6, "value": 0},
        {"time": 2.2, "value": 95},
        {"time": 3.8, "value": 0},
    ]
    midi_path = _write_midi(tmp_path, "solo_quick.mid", notes, pedal_events=pedal_events)

    result = quick_analyze(str(midi_path))

    assert result["analysis_type"] == "solo_performance"
    assert "musical_structure" in result
    assert "performance_analysis" in result
    assert "gpt_ready_summary" in result
    assert "performance_score" not in result
    assert "grade" not in result

    parsed = result["parsed_data"]
    assert parsed["pedaling"]["available"] is True
    assert parsed["pedals"]
    assert parsed["pedal_segments"]
    assert parsed["pedals"][0]["type"] == "sustain"
    assert {"start_time", "end_time", "duration"} <= set(parsed["pedal_segments"][0].keys())
    assert len([obs for obs in result["performance_analysis"]["observations"] if obs["category"] == "repetition"]) <= 5

    _assert_no_forbidden_solo_feedback(result)


def test_solo_register_balance_feedback_stays_cautious(tmp_path):
    notes = []
    start = 0.0
    for _ in range(10):
        notes.append(_note(48, start, duration=0.45, velocity=84))
        notes.append(_note(72, start + 0.01, duration=0.45, velocity=62))
        start += 0.6
    midi_path = _write_midi(tmp_path, "register_balance.mid", notes)

    result = quick_analyze(str(midi_path))
    register_observations = [
        observation
        for observation in result["performance_analysis"]["observations"]
        if observation["category"] == "register_balance"
    ]
    assert register_observations
    combined_text = " ".join(
        f"{observation['interpretation']} {observation['student_friendly_feedback']}"
        for observation in register_observations
    ).casefold()
    assert "if the melody is intended" in combined_text or "may" in combined_text


def test_parser_aliases_and_solo_empty_observation_schema(tmp_path):
    notes = [_note(60, 0.0, duration=1.0, velocity=80)]
    pedal_events = [{"time": 0.1, "value": 100}, {"time": 0.9, "value": 0}]
    midi_path = _write_midi(tmp_path, "pedal_alias.mid", notes, pedal_events=pedal_events)

    parsed = MIDIParser().parse_midi(str(midi_path))
    assert parsed["pedaling"]["available"] is True
    assert len(parsed["pedals"]) == 2
    assert len(parsed["pedal_segments"]) == 1
    assert parsed["pedals"][0]["state"] == "down"
    assert parsed["pedal_segments"][0]["duration"] == 0.8

    empty_analysis = SoloPerformanceAnalysis({"notes": [], "pedals": [], "pedal_segments": []}, {"phrases": []}).analyze()
    assert empty_analysis["no_score"] is True
    assert empty_analysis["observations"]
    assert empty_analysis["summary_statistics"]["note_count"] == 0


def test_solo_runtime_smoke_test_on_bundled_midi():
    sample_path = BACKEND_DIR / "Midi_Analysis" / "TESTA.midi"
    assert sample_path.exists()

    started = time.time()
    result = quick_analyze(str(sample_path))
    elapsed = time.time() - started

    assert result["analysis_type"] == "solo_performance"
    assert elapsed < 10.0
    assert result["gpt_ready_summary"]["limitations"]
