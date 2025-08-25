import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.stt import transcribe_audio
from app import nlp as nlp_mod

# DB imports
from app.db import SessionLocal
from app.models import Meeting  # adjust import if your Meeting model lives elsewhere

# ---- Logging setup ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meeting_tracker")

app = FastAPI(title="Meeting Action Tracker")

class TranscriptIn(BaseModel):
    text: str

class ExtractIn(BaseModel):
    id: int
    text: str

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    """
    Step 1 (draft): Transcribe audio, create a draft Meeting row
    with transcript only, and return its id.
    """
    tmp_path = "temp_upload.wav"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    text = transcribe_audio(tmp_path)
    logger.info("E2E DEBUG /transcribe transcript_preview=%r", (text or "")[:150])

    # Create DRAFT row: transcript only
    db = SessionLocal()
    try:
        meeting = Meeting(
            transcript=text,
            summary="",       # draft
            tasks_json=[],    # draft (empty JSON array)
        )
        db.add(meeting)
        db.commit()
        db.refresh(meeting)
        logger.info("E2E DEBUG /transcribe saved draft id=%s", meeting.id)
        meeting_id = meeting.id
    finally:
        db.close()

    return {"id": meeting_id, "transcript": text}

@app.post("/extract")
async def extract(body: ExtractIn):
    """
    Step 2 (finalize): Given an existing meeting id + transcript text,
    run NLP and update the existing row with summary and tasks_json.
    """
    logger.info("E2E DEBUG /extract id=%s", body.id)

    # Run NLP
    tasks = nlp_mod.extract_tasks(body.text)
    summary = nlp_mod.summarize(body.text)

    logger.info("E2E DEBUG /extract tasks_count=%d example=%r", len(tasks), tasks[0] if tasks else None)
    logger.info("E2E DEBUG /extract summary_preview=%r", (summary or "")[:150])

    # Update existing row
    db = SessionLocal()
    try:
        meeting = db.get(Meeting, body.id)
        if not meeting:
            logger.error("E2E ERROR /extract Meeting id=%s not found", body.id)
            raise HTTPException(status_code=404, detail=f"Meeting id {body.id} not found")

        # Keep transcript consistent with what we just analyzed
        meeting.transcript = body.text
        meeting.summary = summary
        meeting.tasks_json = tasks

        db.commit()
        db.refresh(meeting)
        logger.info("E2E DEBUG /extract updated id=%s tasks_len=%d", meeting.id, len(tasks))
    finally:
        db.close()

    return {"id": body.id, "summary": summary, "tasks": tasks}

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    One-shot path: transcribe -> NLP -> save full record.
    """
    tmp_path = "temp_upload.wav"
    with open(tmp_path, "wb") as f:
        f.write(await file.read())

    transcript = transcribe_audio(tmp_path)
    logger.info("E2E DEBUG /process-audio transcript_preview=%r", (transcript or "")[:150])

    tasks = nlp_mod.extract_tasks(transcript)
    logger.info("E2E DEBUG /process-audio tasks_count=%d example=%r", len(tasks), tasks[0] if tasks else None)

    summary = nlp_mod.summarize(transcript)
    logger.info("E2E DEBUG /process-audio summary_preview=%r", (summary or "")[:150])

    # persist to Postgres (uses JSON column `tasks_json`)
    db = SessionLocal()
    try:
        meeting = Meeting(
            transcript=transcript,
            summary=summary,
            tasks_json=tasks
        )
        db.add(meeting)
        db.commit()
        db.refresh(meeting)
        logger.info("E2E DEBUG /process-audio saved id=%s", meeting.id)
        meeting_id = meeting.id
    finally:
        db.close()

    return {
        "id": meeting_id,
        "transcript": transcript,
        "summary": summary,
        "tasks": tasks
    }
