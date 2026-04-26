import json
import os
import re
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from separation_service import run_spleeter
from transcription_service import get_active_backend_name, run_transcription

app = FastAPI()

DEFAULT_ALLOWED_ORIGINS = {
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://134.208.3.192:5173",
}
EXTRA_ALLOWED_ORIGINS = {
    origin.strip()
    for origin in os.getenv("FRONTEND_ORIGINS", "").split(",")
    if origin.strip()
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=sorted(DEFAULT_ALLOWED_ORIGINS | EXTRA_ALLOWED_ORIGINS),
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
SEPARATED_DIR = Path("separated")
TRANSCRIPTIONS_DIR = Path("transcriptions")
TUTOR_SESSIONS_DIR = Path("tutor_sessions")
MIDI_LIBRARY_DIR = Path("midi_library")
MIDI_LIBRARY_INDEX_PATH = MIDI_LIBRARY_DIR / "index.json"

UPLOAD_DIR.mkdir(exist_ok=True)
SEPARATED_DIR.mkdir(exist_ok=True)
TRANSCRIPTIONS_DIR.mkdir(exist_ok=True)
TUTOR_SESSIONS_DIR.mkdir(exist_ok=True)
MIDI_LIBRARY_DIR.mkdir(exist_ok=True)

app.mount("/separated", StaticFiles(directory="separated"), name="separated")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
app.mount("/transcriptions", StaticFiles(directory="transcriptions"), name="transcriptions")


class TranscriptionRequest(BaseModel):
    audio_path: str
    config_path: str | None = None
    config_name: str | None = "main_config"
    midi_output_path: str | None = None


class MidiAnalysisRequest(BaseModel):
    midi_url: str | None = None
    midi_path: str | None = None


class TutorMessageRequest(BaseModel):
    session_id: str
    message: str


def _save_upload(file: UploadFile) -> Path:
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    save_path = UPLOAD_DIR / unique_name

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return save_path


def _validate_midi_filename(filename: str | None, label: str = "MIDI file") -> None:
    suffix = Path(filename or "").suffix.lower()
    if suffix not in {".mid", ".midi"}:
        raise ValueError(f"{label} must be a MIDI file (.mid or .midi).")


def _sanitize_filename_part(value: str | None, fallback: str = "midi") -> str:
    base = Path(value or fallback).stem or fallback
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", base).strip("._-")
    return cleaned[:80] or fallback


def _read_midi_library_index() -> list[dict[str, Any]]:
    if not MIDI_LIBRARY_INDEX_PATH.is_file():
        return []

    try:
        with open(MIDI_LIBRARY_INDEX_PATH, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return []

    items = payload.get("items", payload) if isinstance(payload, dict) else payload
    return [item for item in items if isinstance(item, dict)]


def _write_midi_library_index(items: list[dict[str, Any]]) -> None:
    MIDI_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    with open(MIDI_LIBRARY_INDEX_PATH, "w", encoding="utf-8") as handle:
        json.dump({"items": items}, handle, indent=2, ensure_ascii=False, default=str)


def _format_midi_library_item(item: dict[str, Any]) -> dict[str, Any]:
    item_id = str(item.get("id", ""))
    return {
        "id": item_id,
        "title": item.get("title") or item.get("original_filename") or "Untitled MIDI",
        "project": item.get("project") or "General",
        "original_filename": item.get("original_filename") or item.get("stored_filename") or "midi.mid",
        "created_at": item.get("created_at"),
        "size_bytes": item.get("size_bytes", 0),
        "download_url": f"/library/midis/{item_id}/download",
    }


def _add_midi_path_to_library(
    source_path: str | Path,
    *,
    original_filename: str | None = None,
    title: str | None = None,
    project: str | None = None,
) -> dict[str, Any]:
    source = Path(source_path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"MIDI file not found: {source}")

    display_filename = Path(original_filename or source.name).name
    _validate_midi_filename(display_filename)

    item_id = uuid.uuid4().hex
    suffix = Path(display_filename).suffix.lower() or source.suffix.lower() or ".mid"
    safe_name = _sanitize_filename_part(display_filename)
    stored_filename = f"{item_id}_{safe_name}{suffix}"
    destination = (MIDI_LIBRARY_DIR / stored_filename).resolve()
    library_root = MIDI_LIBRARY_DIR.resolve()

    try:
        destination.relative_to(library_root)
    except ValueError as exc:
        raise ValueError("MIDI library destination is outside the library directory.") from exc

    MIDI_LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)

    item = {
        "id": item_id,
        "stored_filename": stored_filename,
        "original_filename": display_filename,
        "title": str(title or Path(display_filename).stem).strip() or Path(display_filename).stem,
        "project": str(project or "General").strip() or "General",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": destination.stat().st_size,
    }
    items = _read_midi_library_index()
    items.insert(0, item)
    _write_midi_library_index(items)
    return _format_midi_library_item(item)


def _get_midi_library_item(item_id: str) -> dict[str, Any]:
    normalized = (item_id or "").strip()
    if not re.fullmatch(r"[A-Fa-f0-9]{32}", normalized):
        raise ValueError("Invalid MIDI library item id.")

    for item in _read_midi_library_index():
        if item.get("id") == normalized:
            return item

    raise FileNotFoundError(f"MIDI library item not found: {normalized}")


def _resolve_midi_library_path(item_id: str) -> tuple[Path, dict[str, Any]]:
    item = _get_midi_library_item(item_id)
    stored_filename = Path(str(item.get("stored_filename", ""))).name
    resolved = (MIDI_LIBRARY_DIR / stored_filename).resolve()
    library_root = MIDI_LIBRARY_DIR.resolve()

    try:
        resolved.relative_to(library_root)
    except ValueError as exc:
        raise ValueError("Resolved MIDI library path is outside the library directory.") from exc

    if not resolved.is_file():
        raise FileNotFoundError(f"MIDI library file not found: {item.get('original_filename', item_id)}")

    return resolved, item


def _resolve_midi_path(midi_url: str | None = None, midi_path: str | None = None) -> Path:
    if midi_path and midi_path.strip():
        requested = Path(midi_path.strip()).expanduser()
        resolved = requested.resolve() if requested.is_absolute() else (Path.cwd() / requested).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"MIDI file not found: {resolved}")
        return resolved

    if midi_url and midi_url.strip():
        parsed = urlparse(midi_url.strip())
        route_path = unquote(parsed.path or "")
        if not route_path.startswith("/transcriptions/"):
            raise ValueError("Only MIDI files under /transcriptions can be analyzed.")

        relative_path = route_path.removeprefix("/transcriptions/").lstrip("/")
        resolved = (TRANSCRIPTIONS_DIR / relative_path).resolve()
        transcriptions_root = TRANSCRIPTIONS_DIR.resolve()

        try:
            resolved.relative_to(transcriptions_root)
        except ValueError as exc:
            raise ValueError("Resolved MIDI path is outside the transcriptions directory.") from exc

        if not resolved.is_file():
            raise FileNotFoundError(f"MIDI file not found: {resolved}")

        return resolved

    raise ValueError("Either midi_url or midi_path is required.")


def _build_analysis_overview(metrics: dict, recommendations: list[str]) -> str:
    note_count = int(metrics.get("note_count", 0) or 0)
    duration = float(metrics.get("total_duration", 0.0) or 0.0)
    notes_per_second = float(metrics.get("notes_per_second", 0.0) or 0.0)
    dynamic_range = float(metrics.get("velocity_stats", {}).get("dynamic_range", 0.0) or 0.0)
    avg_duration = float(metrics.get("duration_stats", {}).get("mean", 0.0) or 0.0)
    pitch_range = metrics.get("pitch_range", {})
    pitch_min = pitch_range.get("min", 0)
    pitch_max = pitch_range.get("max", 0)

    overview_parts = [
        f"This MIDI contains {note_count} notes across {duration:.1f} seconds.",
        f"The average note density is {notes_per_second:.2f} notes per second.",
        f"Velocity dynamic range is {dynamic_range:.1f}, and average note duration is {avg_duration:.2f} seconds.",
        f"The pitch span runs from MIDI note {pitch_min} to {pitch_max}.",
    ]

    pedaling = metrics.get("pedaling", {})
    if isinstance(pedaling, dict) and pedaling.get("pedal_analysis_available"):
        if pedaling.get("mode") == "solo":
            overview_parts.append(
                "Pedal analysis found "
                f"{int(pedaling.get('pedal_segment_count', 0) or 0)} sustain spans "
                f"with an average hold of {float(pedaling.get('average_hold_duration', 0.0) or 0.0):.2f} seconds."
            )
        else:
            overview_parts.append(
                "Pedal comparison is available, including missed/extra pedals and release timing."
            )

    if recommendations:
        overview_parts.append(f"Top practice suggestion: {recommendations[0]}")

    return " ".join(overview_parts)


def _load_quick_analyze():
    from Midi_Analysis.analyzer import quick_analyze

    return quick_analyze


def _load_compare_performance():
    from Midi_Analysis.analyzer import compare_performance

    return compare_performance


def _load_gpt_tutor_cls():
    from Midi_Analysis.gpt_tutor import GPTTutor

    return GPTTutor


def _build_missing_dependency_detail(exc: ModuleNotFoundError) -> str:
    missing = exc.name or "unknown"
    return (
        "MIDI analysis is unavailable because the optional dependency "
        f"'{missing}' is not installed. Install the MIDI analysis dependencies and retry."
    )


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _clean_text_list(values: list[Any] | None) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def _build_stat(label: str, value: Any) -> dict[str, str]:
    return {"label": label, "value": str(value)}


def _create_tutor_session_dir() -> tuple[str, Path]:
    session_id = uuid.uuid4().hex
    root = (TUTOR_SESSIONS_DIR / session_id).resolve()
    root.mkdir(parents=True, exist_ok=False)
    return session_id, root


def _resolve_tutor_session_dir(session_id: str) -> Path:
    normalized = (session_id or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_-]{8,80}", normalized):
        raise ValueError("Invalid tutor session id.")

    resolved = (TUTOR_SESSIONS_DIR / normalized).resolve()
    tutor_root = TUTOR_SESSIONS_DIR.resolve()
    try:
        resolved.relative_to(tutor_root)
    except ValueError as exc:
        raise ValueError("Tutor session path is outside the sessions directory.") from exc

    if not resolved.is_dir():
        raise FileNotFoundError(f"Tutor session not found: {normalized}")
    return resolved


def _validate_reference_midi(file: UploadFile | None) -> None:
    if file is None:
        return

    _validate_midi_filename(file.filename, "Reference file")


def _extract_compare_focus(gpt_summary: dict[str, Any]) -> list[str]:
    practice = gpt_summary.get("practice_recommendations", {})
    error_summary = gpt_summary.get("error_analysis_summary", {})
    improvement_areas = gpt_summary.get("progress_metrics", {}).get("improvement_areas", {})

    focus = _clean_text_list(
        practice.get("immediate_focus", [])
        + improvement_areas.get("highest_priority", [])
        + [
            error_summary.get("note_accuracy", {}).get("summary", ""),
            error_summary.get("timing", {}).get("summary", ""),
            error_summary.get("rhythm", {}).get("summary", ""),
        ]
    )
    return focus[:4]


def _extract_compare_plan(gpt_summary: dict[str, Any]) -> list[str]:
    practice = gpt_summary.get("practice_recommendations", {})
    schedule = practice.get("practice_schedule", {})
    daily_focus = schedule.get("daily_focus", {})

    plan = [
        f"{label.replace('_', ' ').title()}: {value}"
        for label, value in daily_focus.items()
        if str(value or "").strip()
    ]
    plan.extend(_clean_text_list(practice.get("general_practice_tips", []))[:2])
    plan.extend(_clean_text_list(practice.get("specific_exercises", []))[:2])
    return _clean_text_list(plan)[:6]


def _build_solo_summary_cards(analysis: dict[str, Any]) -> dict[str, Any]:
    metrics = analysis.get("metrics", {})
    recommendations = _clean_text_list(analysis.get("practice_recommendations", []))
    note_count = int(metrics.get("note_count", 0) or 0)
    total_duration = float(metrics.get("total_duration", 0.0) or 0.0)
    notes_per_second = float(metrics.get("notes_per_second", 0.0) or 0.0)
    dynamic_range = float(metrics.get("velocity_stats", {}).get("dynamic_range", 0.0) or 0.0)
    avg_duration = float(metrics.get("duration_stats", {}).get("mean", 0.0) or 0.0)

    strengths: list[str] = []
    if dynamic_range >= 30:
        strengths.append("There is already some dynamic contrast in the performance.")
    if 0 < notes_per_second <= 8:
        strengths.append("The pacing is in a range that should respond well to slow, deliberate repetition.")
    if avg_duration > 0:
        strengths.append("The MIDI capture is detailed enough to coach articulation and note length.")
    strengths = _clean_text_list(strengths) or ["Your performance is captured and ready for targeted practice feedback."]

    immediate_focus = recommendations[:3] or [
        "Practice in short sections and listen for even note placement.",
        "Slow the passage down until every attack feels controlled.",
    ]
    practice_plan = _clean_text_list(
        [
            "Warm up with one slow pass at 50-70% of target speed.",
            immediate_focus[0] if immediate_focus else "",
            "Loop the hardest 2-4 bars until the notes feel even and relaxed.",
            "Finish with one full run while keeping the pulse steady.",
        ]
    )

    return {
        "overall_assessment": {
            "eyebrow": "Solo MIDI analysis",
            "headline": "Solo practice feedback is ready",
            "summary": _build_analysis_overview(metrics, recommendations),
            "stats": [
                _build_stat("Notes", note_count),
                _build_stat("Duration", f"{total_duration:.1f}s"),
                _build_stat("Density", f"{notes_per_second:.2f}/s"),
                _build_stat("Dynamic range", f"{dynamic_range:.1f}"),
            ],
        },
        "strengths": strengths,
        "immediate_focus": immediate_focus,
        "practice_plan": practice_plan,
    }


def _build_compare_summary_cards(result: dict[str, Any]) -> dict[str, Any]:
    gpt_summary = result.get("gpt_ready_summary", {})
    overview = gpt_summary.get("performance_overview", {})
    assessment = overview.get("overall_assessment", {})
    key_metrics = overview.get("key_metrics", {})
    reliability = overview.get("analysis_reliability", {})

    strengths = _clean_text_list(
        overview.get("strengths", []) + overview.get("performance_characteristics", [])
    ) or ["The comparison is ready and tied to the original reference performance."]
    immediate_focus = _extract_compare_focus(gpt_summary) or [
        "Start by correcting the highest-priority note and rhythm issues one short section at a time."
    ]
    practice_plan = _extract_compare_plan(gpt_summary) or [
        "Spend 5 minutes on a slow reference listen-through.",
        immediate_focus[0],
        "End with one full run at a controlled tempo and compare it again.",
    ]

    summary_parts = []
    grade = str(assessment.get("grade", "")).strip()
    score = assessment.get("score")
    performance_level = str(assessment.get("performance_level", "")).strip()
    if grade or performance_level:
        headline = " ".join(part for part in [grade, performance_level] if part)
        summary_parts.append(f"Comparison complete: {headline}.")
    if isinstance(score, (int, float)):
        summary_parts.append(f"Current overall score: {float(score):.1f}.")
    if reliability.get("is_reliable") is False:
        reason = str(reliability.get("reason", "")).replace("_", " ").strip()
        if reason:
            summary_parts.append(f"Comparison confidence is limited because {reason}.")
    note_accuracy_summary = gpt_summary.get("error_analysis_summary", {}).get("note_accuracy", {}).get("summary")
    if note_accuracy_summary:
        summary_parts.append(str(note_accuracy_summary))

    return {
        "overall_assessment": {
            "eyebrow": "Reference comparison",
            "headline": "Comparison feedback is ready",
            "summary": " ".join(summary_parts).strip() or "Your performance has been compared against the original MIDI.",
            "stats": [
                stat
                for stat in [
                    _build_stat("Grade", grade) if grade else None,
                    _build_stat("Score", f"{float(score):.1f}") if isinstance(score, (int, float)) else None,
                    _build_stat("Accuracy", key_metrics.get("note_accuracy", "")) if key_metrics.get("note_accuracy") else None,
                    _build_stat("Timing", key_metrics.get("timing_consistency", "")) if key_metrics.get("timing_consistency") else None,
                ]
                if stat is not None
            ],
        },
        "strengths": strengths,
        "immediate_focus": immediate_focus,
        "practice_plan": practice_plan,
    }


def _normalize_summary_cards(summary_cards: dict[str, Any]) -> dict[str, Any]:
    overall = dict(summary_cards.get("overall_assessment", {}) or {})
    raw_stats = overall.get("stats", [])
    normalized_stats: list[dict[str, str]] = []
    for item in raw_stats or []:
        if isinstance(item, dict):
            label = str(item.get("label", "")).strip()
            value = str(item.get("value", "")).strip()
            if label and value:
                normalized_stats.append({"label": label, "value": value})
            continue
        try:
            parsed = json.loads(str(item))
        except json.JSONDecodeError:
            continue
        label = str(parsed.get("label", "")).strip()
        value = str(parsed.get("value", "")).strip()
        if label and value:
            normalized_stats.append({"label": label, "value": value})

    overall["stats"] = normalized_stats
    return {
        "overall_assessment": overall,
        "strengths": _clean_text_list(summary_cards.get("strengths", [])),
        "immediate_focus": _clean_text_list(summary_cards.get("immediate_focus", [])),
        "practice_plan": _clean_text_list(summary_cards.get("practice_plan", [])),
    }


def _build_suggested_questions(mode: str) -> list[str]:
    if mode == "compare":
        return [
            "What should I practice first?",
            "Where am I least accurate?",
            "Give me a 15-minute plan.",
        ]
    return [
        "What should I practice first?",
        "How can I improve the timing?",
        "Give me a 15-minute plan.",
    ]


def _build_fallback_opening_message(mode: str, summary_cards: dict[str, Any]) -> str:
    overall = summary_cards.get("overall_assessment", {})
    strengths = _clean_text_list(summary_cards.get("strengths", []))
    immediate_focus = _clean_text_list(summary_cards.get("immediate_focus", []))

    intro = overall.get("headline") or ("Comparison feedback is ready" if mode == "compare" else "Solo feedback is ready")
    summary = overall.get("summary", "")
    parts = [str(intro).strip().rstrip(".") + "."]
    if summary:
        parts.append(str(summary).strip())
    if strengths:
        parts.append(f"One strength to keep: {strengths[0]}.")
    if immediate_focus:
        parts.append(f"Start here: {immediate_focus[0]}.")
    parts.append("Ask what to practice first, where accuracy drops, or for a short practice plan.")
    return " ".join(parts)


def _load_tutor_context(meta: dict[str, Any]) -> dict[str, Any]:
    for key in ("summary_path", "analysis_path"):
        raw_path = str(meta.get(key, "")).strip()
        if not raw_path:
            continue
        path = Path(raw_path)
        if path.is_file():
            return _read_json(path)
    return {}


def _build_practice_plan_reply(summary_cards: dict[str, Any], total_minutes: int = 15) -> str:
    plan = _clean_text_list(summary_cards.get("practice_plan", []))
    if not plan:
        return "Use three short rounds: slow work, loop the hardest bars, then play a clean full pass at a controlled tempo."

    slots = [5, 5, max(total_minutes - 10, 5)]
    lines = []
    for minutes, instruction in zip(slots, plan[:3]):
        lines.append(f"{minutes} min: {instruction}")
    return "Here is a simple practice block. " + " ".join(lines)


def _build_compare_accuracy_reply(context: dict[str, Any], summary_cards: dict[str, Any]) -> str:
    error_summary = context.get("error_analysis_summary", {})
    categorized = error_summary.get("categorized_errors", {}).get("note_errors", {})
    note_summary = error_summary.get("note_accuracy", {}).get("summary", "")
    missing = categorized.get("missing_notes", {}).get("count")
    extra = categorized.get("extra_notes", {}).get("count")

    parts = []
    if note_summary:
        parts.append(str(note_summary).strip())
    if missing is not None or extra is not None:
        detail = []
        if missing is not None:
            detail.append(f"{missing} missing notes")
        if extra is not None:
            detail.append(f"{extra} extra notes")
        parts.append("The biggest note-level gap right now is " + " and ".join(detail) + ".")
    if summary_cards.get("immediate_focus"):
        parts.append(f"Start by isolating this first: {summary_cards['immediate_focus'][0]}.")
    return " ".join(parts) or "Accuracy is the first thing to work on, so slow the hardest passage down and match the reference one small section at a time."


def _build_compare_timing_reply(context: dict[str, Any]) -> str:
    timing = context.get("error_analysis_summary", {}).get("timing", {})
    rhythm = context.get("error_analysis_summary", {}).get("rhythm", {})
    parts = _clean_text_list([timing.get("summary", ""), rhythm.get("summary", "")])
    if parts:
        parts.append("Use a metronome, count subdivisions out loud, and only speed up after the rhythm feels even.")
        return " ".join(parts)
    return "Use a metronome and keep the pulse steady while you loop the hardest bars. When the rhythm feels even three times in a row, nudge the speed up a little."


def _build_solo_timing_reply(context: dict[str, Any]) -> str:
    metrics = context.get("metrics", {})
    note_density = float(metrics.get("notes_per_second", 0.0) or 0.0)
    if note_density > 0:
        return (
            f"This performance sits around {note_density:.2f} notes per second. "
            "Start well below target speed, keep the pulse even, and only increase tempo after three clean repeats."
        )
    return "Start below performance speed, play with a metronome, and focus on even attacks before you speed it back up."


def _build_dynamic_reply(context: dict[str, Any], summary_cards: dict[str, Any]) -> str:
    if "error_analysis_summary" in context:
        dynamics = context.get("error_analysis_summary", {}).get("dynamics", {}).get("summary", "")
        if dynamics:
            return str(dynamics).strip() + " Keep listening for contrast between arrival points and quieter transition notes."

    metrics = context.get("metrics", {})
    dynamic_range = float(metrics.get("velocity_stats", {}).get("dynamic_range", 0.0) or 0.0)
    if dynamic_range > 0:
        return (
            f"Your solo MIDI shows a dynamic range of {dynamic_range:.1f}. "
            "Shape repeated phrases on purpose so the louder notes feel like destinations rather than accidents."
        )

    strengths = _clean_text_list(summary_cards.get("strengths", []))
    if strengths:
        return f"One encouraging sign is this: {strengths[0]}. Keep that, then exaggerate the contrast between soft preparation notes and the main beats."
    return "Practice one phrase twice: once intentionally softer, once intentionally fuller, so the contrast becomes something you can control on demand."


def _build_fallback_tutor_reply(message: str, meta: dict[str, Any], context: dict[str, Any]) -> str:
    normalized = message.lower()
    summary_cards = _normalize_summary_cards(meta.get("summary_cards", {}))
    strengths = summary_cards.get("strengths", [])
    immediate_focus = summary_cards.get("immediate_focus", [])
    mode = meta.get("mode", "solo")

    if any(token in normalized for token in ["practice first", "start with", "first thing"]):
        if immediate_focus:
            return f"Start with this first: {immediate_focus[0]}. Once that feels steadier, move to {immediate_focus[1] if len(immediate_focus) > 1 else 'a second short section at the same slow tempo'}."
        return "Start with the hardest short passage, slow it down, and repeat it until the notes and rhythm feel even."

    if "15-minute" in normalized or "practice plan" in normalized:
        return _build_practice_plan_reply(summary_cards, total_minutes=15)

    if any(token in normalized for token in ["least accurate", "accuracy", "miss", "wrong note", "extra note"]):
        if mode == "compare":
            return _build_compare_accuracy_reply(context, summary_cards)
        if immediate_focus:
            return f"The first accuracy target is {immediate_focus[0]}. Slow that section down and aim for three clean repetitions before you speed it up."
        return "Accuracy improves fastest when you shrink the passage, slow it down, and repeat only the exact notes that keep slipping."

    if any(token in normalized for token in ["timing", "rhythm", "rush", "drag", "tempo"]):
        return _build_compare_timing_reply(context) if mode == "compare" else _build_solo_timing_reply(context)

    if any(token in normalized for token in ["dynamic", "expression", "phrasing", "musical"]):
        return _build_dynamic_reply(context, summary_cards)

    if any(token in normalized for token in ["strength", "good", "well"]):
        if strengths:
            return "Here is something worth keeping: " + " ".join(strengths[:2]) + ". Build the next practice round around keeping that while tightening the weaker spots."
        return "A useful strength right now is that you already have a full performance to coach from, so we can focus on one clear improvement at a time."

    overall = summary_cards.get("overall_assessment", {})
    summary = str(overall.get("summary", "")).strip()
    if summary and immediate_focus:
        return f"{summary} The next practical step is {immediate_focus[0]}."
    if immediate_focus:
        return f"I’d keep the next rep simple: {immediate_focus[0]}. After that, ask me for a shorter drill or a timing-focused plan."
    return "Let’s keep the next pass focused: choose one short section, slow it down, and make the pulse and note placement feel consistent before you speed it up."


def _persist_tutor_state(state_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    _write_json(state_path, payload)
    return payload


def _start_tutor_session(
    session_root: Path,
    mode: str,
    summary_cards: dict[str, Any],
    summary_path: Path | None = None,
    analysis_path: Path | None = None,
) -> tuple[str, str]:
    state_path = session_root / "tutor_session.json"
    fallback_message = _build_fallback_opening_message(mode, summary_cards)
    fallback_state = {
        "engine": "fallback",
        "mode": mode,
        "summary_path": str(summary_path) if summary_path else None,
        "analysis_path": str(analysis_path) if analysis_path else None,
        "history": [],
    }

    if mode != "compare" or summary_path is None or not summary_path.is_file():
        _persist_tutor_state(state_path, fallback_state)
        return "fallback", fallback_message

    try:
        GPTTutor = _load_gpt_tutor_cls()
        tutor = GPTTutor(model=os.getenv("OPENAI_MODEL", "gpt-5-mini"))
        response = tutor.start_session(
            summary=str(summary_path),
            student_question="Please give a concise opening assessment, one strength, and the first thing this student should practice.",
            max_output_tokens=700,
        )
        state_payload = tutor.save_state(state_path)
        state_payload.update({"engine": "openai", "mode": mode, "analysis_path": str(analysis_path) if analysis_path else None})
        _persist_tutor_state(state_path, state_payload)
        return "openai", response.get("text", "").strip() or fallback_message
    except Exception as exc:
        fallback_state["fallback_reason"] = str(exc)
        _persist_tutor_state(state_path, fallback_state)
        return "fallback", fallback_message


def _build_tutor_prepare_response(
    mode: str,
    performance_midi_url: str,
    summary_cards: dict[str, Any],
    suggested_questions: list[str],
    session_id: str,
    opening_message: str,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "performance_midi_url": performance_midi_url,
        "summary_cards": _normalize_summary_cards(summary_cards),
        "suggested_questions": suggested_questions,
        "tutor": {
            "session_id": session_id,
            "opening_message": opening_message,
        },
    }


def _prepare_tutor_session(
    performance_audio: UploadFile,
    reference_midi: UploadFile | None = None,
    reference_midi_library_id: str | None = None,
    config_path: str | None = None,
    config_name: str | None = "main_config",
) -> dict[str, Any]:
    _validate_reference_midi(reference_midi)
    library_reference_path: Path | None = None
    library_reference_item: dict[str, Any] | None = None
    normalized_library_id = (reference_midi_library_id or "").strip()
    if reference_midi is not None and normalized_library_id:
        raise ValueError("Choose either an uploaded reference MIDI or a library MIDI, not both.")

    if normalized_library_id:
        library_reference_path, library_reference_item = _resolve_midi_library_path(normalized_library_id)

    session_id, session_root = _create_tutor_session_dir()
    saved_audio = _save_upload(performance_audio)
    saved_reference = _save_upload(reference_midi) if reference_midi is not None else None
    reference_source_path = saved_reference or library_reference_path

    audio_suffix = saved_audio.suffix or Path(performance_audio.filename or "").suffix or ".bin"
    shutil.copy2(saved_audio, session_root / f"performance_input{audio_suffix}")
    if reference_source_path is not None:
        reference_suffix = (
            reference_source_path.suffix
            or Path(reference_midi.filename if reference_midi is not None else "").suffix
            or ".mid"
        )
        shutil.copy2(reference_source_path, session_root / f"reference_original{reference_suffix}")

    reference_library_item = None
    if saved_reference is not None:
        reference_library_item = _add_midi_path_to_library(
            saved_reference,
            original_filename=reference_midi.filename,
            title=Path(reference_midi.filename or "Reference MIDI").stem,
            project="Reference uploads",
        )
    elif library_reference_item is not None:
        reference_library_item = _format_midi_library_item(library_reference_item)

    transcription = run_transcription(
        audio_path=str(saved_audio),
        config_path=config_path,
        config_name=config_name,
        midi_output_path=f"tutor_sessions/{session_id}/performance.mid",
    )

    performance_midi_path = Path(transcription["midi_path"]).resolve()
    session_midi_path = session_root / "transcribed_performance.mid"
    shutil.copy2(performance_midi_path, session_midi_path)
    performance_library_item = _add_midi_path_to_library(
        performance_midi_path,
        original_filename=f"{Path(performance_audio.filename or 'performance').stem}_transcription.mid",
        title=f"{Path(performance_audio.filename or 'Performance').stem} transcription",
        project="Transcribed performances",
    )

    mode = "compare" if reference_source_path is not None else "solo"
    summary_path: Path | None = None
    analysis_path: Path | None = None

    if mode == "compare":
        compare_performance = _load_compare_performance()
        compare_performance(
            reference_path=str(reference_source_path),
            performance_path=str(performance_midi_path),
            output_dir=str(session_root),
        )
        summary_path = session_root / "gpt_summary.json"
        analysis_path = session_root / "full_analysis.json"
        summary_cards = _build_compare_summary_cards(_read_json(analysis_path))
    else:
        quick_analyze = _load_quick_analyze()
        solo_analysis = quick_analyze(str(performance_midi_path))
        analysis_path = session_root / "solo_analysis.json"
        _write_json(analysis_path, solo_analysis)
        summary_cards = _build_solo_summary_cards(solo_analysis)

    summary_cards = _normalize_summary_cards(summary_cards)
    suggested_questions = _build_suggested_questions(mode)
    engine, opening_message = _start_tutor_session(
        session_root=session_root,
        mode=mode,
        summary_cards=summary_cards,
        summary_path=summary_path,
        analysis_path=analysis_path,
    )

    session_meta = {
        "session_id": session_id,
        "mode": mode,
        "engine": engine,
        "performance_audio_path": str(saved_audio),
        "performance_midi_path": str(performance_midi_path),
        "performance_midi_url": transcription["midi_url"],
        "performance_library_item": performance_library_item,
        "reference_midi_path": str(reference_source_path) if reference_source_path is not None else None,
        "reference_library_item": reference_library_item,
        "summary_path": str(summary_path) if summary_path is not None else None,
        "analysis_path": str(analysis_path) if analysis_path is not None else None,
        "summary_cards": summary_cards,
        "suggested_questions": suggested_questions,
    }
    _write_json(session_root / "session_meta.json", session_meta)

    return _build_tutor_prepare_response(
        mode=mode,
        performance_midi_url=transcription["midi_url"],
        summary_cards=summary_cards,
        suggested_questions=suggested_questions,
        session_id=session_id,
        opening_message=opening_message,
    )


def _load_tutor_session_files(session_id: str) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    session_root = _resolve_tutor_session_dir(session_id)
    meta_path = session_root / "session_meta.json"
    state_path = session_root / "tutor_session.json"

    if not meta_path.is_file():
        raise FileNotFoundError(f"Tutor session metadata not found: {session_id}")
    if not state_path.is_file():
        raise FileNotFoundError(f"Tutor session state not found: {session_id}")

    return session_root, _read_json(meta_path), state_path, _read_json(state_path)


def _reply_to_tutor_message(session_id: str, message: str) -> dict[str, str]:
    text = (message or "").strip()
    if not text:
        raise ValueError("message is required.")

    session_root, meta, state_path, state = _load_tutor_session_files(session_id)
    context = _load_tutor_context(meta)

    if state.get("engine") == "openai":
        try:
            GPTTutor = _load_gpt_tutor_cls()
            tutor = GPTTutor(model=state.get("model"))
            tutor.load_state(state_path)
            result = tutor.ask(text, max_output_tokens=700)
            saved_state = tutor.save_state(state_path)
            saved_state.update(
                {
                    "engine": "openai",
                    "mode": meta.get("mode", "compare"),
                    "analysis_path": meta.get("analysis_path"),
                }
            )
            _persist_tutor_state(state_path, saved_state)
            reply = result.get("text", "").strip()
            if reply:
                return {"reply": reply}
        except Exception as exc:
            state = {
                "engine": "fallback",
                "mode": meta.get("mode", "compare"),
                "summary_path": meta.get("summary_path"),
                "analysis_path": meta.get("analysis_path"),
                "history": state.get("history", []),
                "fallback_reason": str(exc),
            }

    reply = _build_fallback_tutor_reply(text, meta, context)
    history = list(state.get("history", []))
    history.append({"role": "user", "text": text})
    history.append({"role": "tutor", "text": reply})
    state.update(
        {
            "engine": "fallback",
            "mode": meta.get("mode", "solo"),
            "summary_path": meta.get("summary_path"),
            "analysis_path": meta.get("analysis_path"),
            "history": history[-20:],
        }
    )
    _persist_tutor_state(state_path, state)

    meta["engine"] = state["engine"]
    _write_json(session_root / "session_meta.json", meta)
    return {"reply": reply}


@app.get("/")
async def root():
    return {
        "message": "FastAPI backend is running",
        "transcription_backend": get_active_backend_name(),
    }


@app.get("/ping")
async def ping():
    return {
        "message": "pong",
        "transcription_backend": get_active_backend_name(),
    }


@app.get("/library/midis")
async def list_midi_library():
    items = [
        _format_midi_library_item(item)
        for item in _read_midi_library_index()
    ]
    return {"items": items}


@app.post("/library/midis")
async def upload_midi_to_library(
    file: UploadFile = File(...),
    title: str | None = Form(None),
    project: str | None = Form("General"),
):
    try:
        _validate_midi_filename(file.filename)
        saved_path = _save_upload(file)
        item = _add_midi_path_to_library(
            saved_path,
            original_filename=file.filename,
            title=title,
            project=project,
        )
        return {"item": item}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save MIDI to library: {str(exc)}") from exc


@app.get("/library/midis/{item_id}/download")
async def download_midi_from_library(item_id: str):
    try:
        midi_path, item = _resolve_midi_library_path(item_id)
        return FileResponse(
            midi_path,
            media_type="audio/midi",
            filename=str(item.get("original_filename") or midi_path.name),
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    save_path = _save_upload(file)

    try:
        stems = run_spleeter(str(save_path), str(SEPARATED_DIR))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spleeter failed: {str(e)}")

    return {
        "message": "File uploaded and separated successfully",
        "original_filename": file.filename,
        "saved_path": str(save_path),
        "stems": stems,
    }


@app.post("/transcribe-upload")
async def transcribe_upload(
    file: UploadFile = File(...),
    config_path: str | None = Form(None),
    config_name: str | None = Form("main_config"),
    midi_output_path: str | None = Form(None),
):
    save_path = _save_upload(file)

    try:
        result = run_transcription(
            audio_path=str(save_path),
            config_path=config_path,
            config_name=config_name,
            midi_output_path=midi_output_path,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}") from exc

    result.update(
        {
            "original_filename": file.filename,
            "saved_path": str(save_path),
        }
    )
    midi_path = result.get("midi_path")
    if midi_path:
        try:
            result["library_item"] = _add_midi_path_to_library(
                midi_path,
                original_filename=f"{Path(file.filename or 'performance').stem}_transcription.mid",
                title=f"{Path(file.filename or 'Performance').stem} transcription",
                project="Transcribed performances",
            )
        except Exception:
            result["library_item"] = None
    return result


@app.post("/transcribe")
async def transcribe(request: TranscriptionRequest):
    try:
        return run_transcription(
            audio_path=request.audio_path,
            config_path=request.config_path,
            config_name=request.config_name,
            midi_output_path=request.midi_output_path,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(exc)}") from exc


@app.post("/tutor/prepare")
async def tutor_prepare(
    performance_audio: UploadFile = File(...),
    reference_midi: UploadFile | None = File(None),
    reference_midi_library_id: str | None = Form(None),
    config_path: str | None = Form(None),
    config_name: str | None = Form("main_config"),
):
    try:
        return _prepare_tutor_session(
            performance_audio=performance_audio,
            reference_midi=reference_midi,
            reference_midi_library_id=reference_midi_library_id,
            config_path=config_path,
            config_name=config_name,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=_build_missing_dependency_detail(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tutor preparation failed: {str(exc)}") from exc


@app.post("/tutor/message")
async def tutor_message(request: TutorMessageRequest):
    try:
        return _reply_to_tutor_message(request.session_id, request.message)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=_build_missing_dependency_detail(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Tutor chat failed: {str(exc)}") from exc


@app.post("/midi/analyze")
async def analyze_midi(request: MidiAnalysisRequest):
    try:
        resolved_midi_path = _resolve_midi_path(request.midi_url, request.midi_path)
        quick_analyze = _load_quick_analyze()
        analysis = quick_analyze(str(resolved_midi_path))
        metrics = analysis.get("metrics", {})
        recommendations = analysis.get("practice_recommendations", [])

        return {
            "message": "MIDI analysis completed successfully",
            "midi_path": str(resolved_midi_path),
            "analysis_type": analysis.get("analysis_type"),
            "metrics": metrics,
            "practice_recommendations": recommendations,
            "parsed_metadata": analysis.get("parsed_data", {}).get("metadata", {}),
            "pedaling": analysis.get("parsed_data", {}).get("pedaling", {}),
            "pedaling_analysis": (
                analysis.get("performance_analysis", {}).get("metrics", {}).get("pedaling", {})
                or metrics.get("pedaling", {})
            ),
            "analysis_overview": _build_analysis_overview(metrics, recommendations),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModuleNotFoundError as exc:
        raise HTTPException(status_code=503, detail=_build_missing_dependency_detail(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"MIDI analysis failed: {str(exc)}") from exc
