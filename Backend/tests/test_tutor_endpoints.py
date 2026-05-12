from pathlib import Path
import sys

from fastapi.testclient import TestClient


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import app as backend_app


def _configure_temp_dirs(monkeypatch, tmp_path):
    upload_dir = tmp_path / "uploads"
    transcriptions_dir = tmp_path / "transcriptions"
    tutor_sessions_dir = tmp_path / "tutor_sessions"
    midi_library_dir = tmp_path / "midi_library"

    upload_dir.mkdir()
    transcriptions_dir.mkdir()
    tutor_sessions_dir.mkdir()
    midi_library_dir.mkdir()

    monkeypatch.setattr(backend_app, "UPLOAD_DIR", upload_dir)
    monkeypatch.setattr(backend_app, "TRANSCRIPTIONS_DIR", transcriptions_dir)
    monkeypatch.setattr(backend_app, "TUTOR_SESSIONS_DIR", tutor_sessions_dir)
    monkeypatch.setattr(backend_app, "MIDI_LIBRARY_DIR", midi_library_dir)
    monkeypatch.setattr(backend_app, "MIDI_LIBRARY_INDEX_PATH", midi_library_dir / "index.json")

    return upload_dir, transcriptions_dir, tutor_sessions_dir


def _fake_run_transcription(transcriptions_dir):
    def _runner(audio_path, config_path=None, config_name="main_config", midi_output_path=None):
        relative_output = Path(midi_output_path)
        midi_path = transcriptions_dir / relative_output
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        midi_path.write_bytes(b"MThd")
        return {
            "message": "MIDI transcription completed successfully",
            "audio_path": audio_path,
            "midi_path": str(midi_path),
            "midi_url": f"/transcriptions/{relative_output.as_posix()}",
            "config_path": config_path,
            "config_name": config_name,
            "model_backend": "test",
        }

    return _runner


def _fake_start_tutor_session(session_root, mode, summary_cards, summary_path=None, analysis_path=None):
    backend_app._write_json(
        session_root / "tutor_session.json",
        {
            "engine": "fallback",
            "mode": mode,
            "summary_path": str(summary_path) if summary_path else None,
            "analysis_path": str(analysis_path) if analysis_path else None,
            "history": [],
        },
    )
    return "fallback", "Tutor session ready."


def test_resolve_midi_path_accepts_library_download_url(monkeypatch, tmp_path):
    _configure_temp_dirs(monkeypatch, tmp_path)
    item_id = "a" * 32
    stored_filename = f"{item_id}_reference.mid"
    stored_path = backend_app.MIDI_LIBRARY_DIR / stored_filename
    stored_path.write_bytes(b"MThd")
    backend_app._write_midi_library_index(
        [
            {
                "id": item_id,
                "stored_filename": stored_filename,
                "original_filename": "reference.mid",
            }
        ]
    )

    resolved = backend_app._resolve_midi_path(
        midi_url=f"http://testserver/library/midis/{item_id}/download"
    )

    assert resolved == stored_path.resolve()


def test_tutor_prepare_solo_mode(monkeypatch, tmp_path):
    _, transcriptions_dir, tutor_sessions_dir = _configure_temp_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_app, "run_transcription", _fake_run_transcription(transcriptions_dir))
    monkeypatch.setattr(backend_app, "_start_tutor_session", _fake_start_tutor_session)
    monkeypatch.setattr(
        backend_app,
        "_load_quick_analyze",
        lambda: (lambda midi_path: {
            "analysis_type": "solo_performance",
            "metrics": {
                "note_count": 42,
                "total_duration": 12.5,
                "notes_per_second": 3.36,
                "velocity_stats": {"dynamic_range": 48},
                "duration_stats": {"mean": 0.38},
            },
            "practice_recommendations": [
                "Practice in short sections and listen for even note placement.",
            ],
        }),
    )

    client = TestClient(backend_app.app)
    response = client.post(
        "/tutor/prepare",
        files={
            "performance_audio": ("student.wav", b"fake-audio", "audio/wav"),
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "solo"
    assert payload["performance_midi_url"].startswith("/transcriptions/tutor_sessions/")
    assert payload["summary_cards"]["overall_assessment"]["headline"] == "Solo practice feedback is ready"
    assert payload["tutor"]["opening_message"] == "Tutor session ready."
    assert payload["project"]["name"] == "student"
    assert payload["reference_library_item"]["role"] == "reference"
    assert payload["reference_library_item"]["project"] == "student"
    assert payload["reference_library_item"]["tutor_session_id"] == payload["tutor"]["session_id"]

    session_root = tutor_sessions_dir / payload["tutor"]["session_id"]
    assert (session_root / "session_meta.json").is_file()
    assert (session_root / "solo_analysis.json").is_file()
    assert (session_root / "transcribed_performance.mid").is_file()


def test_tutor_prepare_compare_mode_and_fallback_message(monkeypatch, tmp_path):
    _, transcriptions_dir, tutor_sessions_dir = _configure_temp_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_app, "run_transcription", _fake_run_transcription(transcriptions_dir))
    monkeypatch.setattr(backend_app, "_start_tutor_session", _fake_start_tutor_session)

    def fake_compare_performance(reference_path, performance_path, output_dir, alignment_backend="native", alignment_model="automatic_hdtw_sym"):
        output = Path(output_dir)
        result = {
            "gpt_ready_summary": {
                "performance_overview": {
                    "overall_assessment": {
                        "grade": "B",
                        "score": 82.4,
                        "performance_level": "Intermediate",
                    },
                    "key_metrics": {
                        "note_accuracy": "87.0%",
                        "timing_consistency": "+/-18.0 ms",
                    },
                    "analysis_reliability": {
                        "is_reliable": True,
                    },
                    "strengths": ["Good dynamic expression"],
                    "performance_characteristics": ["Clear melodic shaping"],
                },
                "error_analysis_summary": {
                    "note_accuracy": {
                        "summary": "A few note accuracy slips appear in the busiest passage.",
                    },
                    "timing": {
                        "summary": "The pulse is mostly steady with a little rushing before cadences.",
                    },
                    "rhythm": {
                        "summary": "Rhythmic consistency is generally solid.",
                    },
                    "categorized_errors": {
                        "note_errors": {
                            "missing_notes": {"count": 4},
                            "extra_notes": {"count": 2},
                        },
                    },
                },
                "practice_recommendations": {
                    "immediate_focus": [
                        "Clean up the busiest right-hand run one bar at a time.",
                    ],
                    "practice_schedule": {
                        "daily_focus": {
                            "warmup": "5 minutes of slow scales",
                            "piece_work": "10 minutes on the busiest run",
                        },
                    },
                    "general_practice_tips": [
                        "Use a metronome and count subdivisions.",
                    ],
                    "specific_exercises": [
                        "Hands-separate work on the right-hand run",
                    ],
                },
                "progress_metrics": {
                    "improvement_areas": {
                        "highest_priority": ["Note accuracy"],
                    },
                },
            },
        }
        backend_app._write_json(output / "full_analysis.json", result)
        backend_app._write_json(output / "gpt_summary.json", result["gpt_ready_summary"])

    monkeypatch.setattr(backend_app, "_load_compare_performance", lambda: fake_compare_performance)

    client = TestClient(backend_app.app)
    prepare_response = client.post(
        "/tutor/prepare",
        files=[
            ("performance_audio", ("student.wav", b"fake-audio", "audio/wav")),
            ("reference_midi", ("reference.mid", b"MThd", "audio/midi")),
        ],
    )

    assert prepare_response.status_code == 200
    payload = prepare_response.json()
    assert payload["mode"] == "compare"
    assert payload["summary_cards"]["overall_assessment"]["headline"] == "Generated MIDI comparison is ready"
    assert payload["project"]["name"] == "reference"
    assert payload["reference_library_item"]["role"] == "reference"
    assert payload["performance_library_item"]["role"] == "performance"
    assert payload["performance_library_item"]["project"] == "reference"
    assert payload["performance_library_item"]["related_reference_id"] == payload["reference_library_item"]["id"]
    assert payload["suggested_questions"] == [
        "What should I practice first?",
        "Where am I least accurate?",
        "Give me a 15-minute plan.",
    ]

    message_response = client.post(
        "/tutor/message",
        json={
            "session_id": payload["tutor"]["session_id"],
            "message": "Where am I least accurate?",
        },
    )

    assert message_response.status_code == 200
    assert "note-level gap" in message_response.json()["reply"]

    session_root = tutor_sessions_dir / payload["tutor"]["session_id"]
    assert (session_root / "full_analysis.json").is_file()
    assert (session_root / "gpt_summary.json").is_file()
    assert (session_root / "tutor_session.json").is_file()


def test_tutor_prepare_rejects_non_midi_reference(monkeypatch, tmp_path):
    _, transcriptions_dir, _ = _configure_temp_dirs(monkeypatch, tmp_path)
    monkeypatch.setattr(backend_app, "run_transcription", _fake_run_transcription(transcriptions_dir))

    client = TestClient(backend_app.app)
    response = client.post(
        "/tutor/prepare",
        files=[
            ("performance_audio", ("student.wav", b"fake-audio", "audio/wav")),
            ("reference_midi", ("reference.txt", b"not-midi", "text/plain")),
        ],
    )

    assert response.status_code == 400
    assert "MIDI file" in response.json()["detail"]
