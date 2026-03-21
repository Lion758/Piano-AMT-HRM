from fastapi import FastAPI, UploadFile, File, HTTPException
from pathlib import Path
import shutil
import uuid

from separation_service import run_spleeter

app = FastAPI()

UPLOAD_DIR = Path("uploads")
SEPARATED_DIR = Path("separated")

UPLOAD_DIR.mkdir(exist_ok=True)
SEPARATED_DIR.mkdir(exist_ok=True)


@app.get("/")
async def root():
    return {"message": "FastAPI backend is running"}


@app.get("/ping")
async def ping():
    return {"message": "pong"}


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
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