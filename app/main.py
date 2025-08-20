from fastapi import FastAPI, UploadFile, File
from app.stt import transcribe_audio

app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with open("temp.wav", "wb") as f:
        f.write(await file.read())
    text = transcribe_audio("temp.wav")
    return {"transcript": text}