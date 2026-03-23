from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import shutil
import uuid
from fastapi.middleware.cors import CORSMiddleware

from separation_service import run_spleeter

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

UPLOAD_DIR.mkdir(exist_ok=True)
SEPARATED_DIR.mkdir(exist_ok=True)

app.mount("/separated", StaticFiles(directory="separated"), name="separated")


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