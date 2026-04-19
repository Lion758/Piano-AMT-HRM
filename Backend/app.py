import os
import shutil
import uuid
from pathlib import Path
from urllib.parse import unquote, urlparse

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
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

UPLOAD_DIR.mkdir(exist_ok=True)
SEPARATED_DIR.mkdir(exist_ok=True)
TRANSCRIPTIONS_DIR.mkdir(exist_ok=True)

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


def _save_upload(file: UploadFile) -> Path:
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    save_path = UPLOAD_DIR / unique_name

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    return save_path


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

    if recommendations:
        overview_parts.append(f"Top practice suggestion: {recommendations[0]}")

    return " ".join(overview_parts)


def _load_quick_analyze():
    from Midi_Analysis.analyzer import quick_analyze

    return quick_analyze


def _build_missing_dependency_detail(exc: ModuleNotFoundError) -> str:
    missing = exc.name or "unknown"
    return (
        "MIDI analysis is unavailable because the optional dependency "
        f"'{missing}' is not installed. Install the MIDI analysis dependencies and retry."
    )


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
