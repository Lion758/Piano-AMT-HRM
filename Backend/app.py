from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from separation_service import run_spleeter
from transcription_service import run_transcription

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://134.208.3.192:5173"],
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


@app.get("/")
async def root():
    return {"message": "FastAPI backend is running"}


@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.post("/separate")
async def separate(file: UploadFile = File(...)):
    unique_name = f"{uuid.uuid4()}_{file.filename}"
    save_path = UPLOAD_DIR / unique_name

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

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
