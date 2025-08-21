# app/main.py
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from app.stt import transcribe_audio
from app import nlp as nlp_mod

app = FastAPI(title="Meeting Action Tracker")

class TranscriptIn(BaseModel):
    text: str

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Save uploaded file to a temp path
    tmp_path = "temp_upload.wav"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    text = transcribe_audio(tmp_path)
    return {"transcript": text}

@app.post("/extract")
async def extract(body: TranscriptIn):
    tasks = nlp_mod.extract_tasks(body.text)
    summary = nlp_mod.summarize(body.text)
    return {"summary": summary, "tasks": tasks}

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    tmp_path = "temp_upload.wav"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())
    transcript = transcribe_audio(tmp_path)
    tasks = nlp_mod.extract_tasks(transcript)
    summary = nlp_mod.summarize(transcript)
    return {
        "transcript": transcript,
        "summary": summary,
        "tasks": tasks
    }
