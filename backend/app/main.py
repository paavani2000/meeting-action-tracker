import logging
import os
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from typing import List
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
import requests

# ---- Your NLP & DB imports ----
from app.stt import transcribe_audio
from app import nlp as nlp_mod
from sqlalchemy.orm import Session
from app.db import SessionLocal
from app.models import Meeting

from fastapi.middleware.cors import CORSMiddleware

# ---- Load environment variables ----
load_dotenv()
JIRA_BASE = os.getenv("JIRA_BASE")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")

# ---- Logging setup ----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("meeting_tracker")

# ---- FastAPI app ----
app = FastAPI(title="Meeting Action Tracker")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- DB dependency ----
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---- Pydantic models ----
class TranscriptIn(BaseModel):
    text: str

class ExtractIn(BaseModel):
    id: int
    text: str

class JiraTicketIn(BaseModel):
    meeting_id: int
    assignee_email: EmailStr

# ---- Helper functions for Jira/email ----
def get_jira_account_id(email: str):
    url = f"{JIRA_BASE}/rest/api/3/user/search?query={email}"
    resp = requests.get(url, auth=(JIRA_EMAIL, JIRA_API_TOKEN))
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail="Failed to fetch Jira user")
    users = resp.json()
    if not users:
        raise HTTPException(status_code=404, detail=f"No Jira user found for {email}")
    return users[0]["accountId"]

def create_jira_issue(summary, description, account_id):
    url = f"{JIRA_BASE}/rest/api/3/issue"
    payload = {
        "fields": {
            "project": {"key": "KAN"},  # Replace with your Jira project key
            "summary": summary,
            "description": description,
            "issuetype": {"name": "Task"},
            "assignee": {"id": account_id}
        }
    }
    resp = requests.post(url, json=payload, auth=(JIRA_EMAIL, JIRA_API_TOKEN))
    if resp.status_code != 201:
        raise HTTPException(status_code=resp.status_code, detail="Failed to create Jira issue")
    return resp.json()

def send_email(to_email, subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SMTP_USER
    msg["To"] = to_email

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.sendmail(msg["From"], [to_email], msg.as_string())

# ---- API Endpoints ----

@app.get("/meetings")
def list_meetings(db: Session = Depends(get_db)) -> List[dict]:
    rows = db.query(Meeting).order_by(Meeting.created_at.desc()).limit(50).all()
    return [
        {
            "id": m.id,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "summary": m.summary,
            "tasks_json": m.tasks_json,
        }
        for m in rows
    ]

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

    db = SessionLocal()
    try:
        meeting = Meeting(
            transcript=text,
            summary="",
            tasks_json=[]
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

    tasks = nlp_mod.extract_tasks(body.text)
    summary = nlp_mod.summarize(body.text)

    logger.info("E2E DEBUG /extract tasks_count=%d example=%r", len(tasks), tasks[0] if tasks else None)
    logger.info("E2E DEBUG /extract summary_preview=%r", (summary or "")[:150])

    db = SessionLocal()
    try:
        meeting = db.get(Meeting, body.id)
        if not meeting:
            logger.error("E2E ERROR /extract Meeting id=%s not found", body.id)
            raise HTTPException(status_code=404, detail=f"Meeting id {body.id} not found")

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

@app.post("/create-jira-tickets")
def create_jira_tickets(data: JiraTicketIn, db: Session = Depends(get_db)):
    """
    Create Jira tickets for all tasks in a meeting and email the assignee.
    """
    meeting = db.get(Meeting, data.meeting_id)
    if not meeting:
        raise HTTPException(status_code=404, detail=f"Meeting id {data.meeting_id} not found")

    if not meeting.tasks_json:
        raise HTTPException(status_code=400, detail="No tasks found for this meeting")

    account_id = get_jira_account_id(data.assignee_email)

    created_tickets = []
    for task in meeting.tasks_json:
        issue = create_jira_issue(task, f"Auto-generated from Meeting ID {meeting.id}", account_id)
        send_email(
            data.assignee_email,
            f"New Jira Ticket Assigned: {issue['key']}",
            f"You have been assigned a new Jira ticket.\n\n"
            f"Task: {task}\n"
            f"Ticket Link: {JIRA_BASE}/browse/{issue['key']}"
        )
        created_tickets.append(issue["key"])

    return {"status": "success", "tickets": created_tickets}
