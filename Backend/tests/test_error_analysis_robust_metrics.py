from pathlib import Path
import sys


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from Midi_Analysis.error_analysis import ErrorAnalysis


def _note(pitch: int, start: float, duration: float, velocity: int = 80) -> dict:
    return {
        "pitch": pitch,
        "start": start,
        "end": start + duration,
        "duration": duration,
        "velocity": velocity,
        "track_id": 0,
    }


def test_timing_metrics_trim_worst_five_percent_after_tempo_normalization():
    reference_notes = []
    performance_notes = []
    alignment = []

    for i in range(40):
        ref_start = i * 0.5
        ref_note = _note(60 + (i % 12), ref_start, 0.25)

        perf_start = 1.25 + ref_start * 2.0
        if i in {10, 20}:
            perf_start += 4.0
        perf_note = _note(ref_note["pitch"], perf_start, 0.5)

        reference_notes.append(ref_note)
        performance_notes.append(perf_note)
        alignment.append(
            {
                "reference_note": ref_note,
                "performance_note": perf_note,
                "pitch_difference": 0,
                "time_difference": perf_start - ref_start,
                "error_type": "none",
            }
        )

    result = ErrorAnalysis(
        {
            "reference": {"notes": reference_notes},
            "performance": {"notes": performance_notes},
            "alignment": alignment,
        }
    ).analyze_performance()

    metrics = result["metrics"]
    assert metrics["alignment_filter"]["trimmed_pair_count"] == 2
    assert metrics["alignment_filter"]["scored_pair_count"] == 38
    assert metrics["alignment_filter"]["tempo_normalization"]["scale"] == 2.0
    assert metrics["timing_errors"]["std_error_ms"] == 0.0
    assert metrics["rhythmic_consistency"]["average_duration_ratio"] == 1.0
    assert metrics["rhythmic_consistency"]["average_ioi_ratio"] == 1.0


def test_rhythmic_consistency_groups_chord_onsets_before_ioi_scoring():
    reference_notes = []
    performance_notes = []
    alignment = []

    for onset_index, ref_start in enumerate([0.0, 1.0, 2.0]):
        for chord_index, pitch in enumerate([60, 64, 67]):
            ref_note = _note(pitch, ref_start, 0.5)
            perf_start = ref_start + chord_index * 0.03
            perf_note = _note(pitch, perf_start, 0.5)

            reference_notes.append(ref_note)
            performance_notes.append(perf_note)
            alignment.append(
                {
                    "reference_note": ref_note,
                    "performance_note": perf_note,
                    "pitch_difference": 0,
                    "time_difference": perf_start - ref_start,
                    "error_type": "none",
                }
            )

    result = ErrorAnalysis(
        {
            "reference": {"notes": reference_notes},
            "performance": {"notes": performance_notes},
            "alignment": alignment,
        }
    ).analyze_performance()

    rhythm = result["metrics"]["rhythmic_consistency"]
    assert rhythm["onset_grouping"]["group_count"] == 3
    assert rhythm["onset_grouping"]["chord_group_count"] == 3
    assert rhythm["onset_grouping"]["grouped_ioi_count"] == 2
    assert rhythm["average_ioi_ratio"] == 1.0
    assert rhythm["ioi_match_score"] == 1.0
