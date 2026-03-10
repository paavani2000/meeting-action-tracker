import logging
import os
import uuid
import subprocess
import difflib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
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
from app.models import Meeting, TeamMember

from fastapi.middleware.cors import CORSMiddleware

# ---- Load environment variables ----
load_dotenv()
JIRA_BASE = os.getenv("JIRA_BASE")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
JIRA_PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY", "KAN")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", 587))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL")

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

class TeamMemberIn(BaseModel):
    name: str
    email: EmailStr

# ---- Helper functions for Jira/email ----
def get_jira_account_id(email: str):
    url = f"{JIRA_BASE}/rest/api/3/user/search?query={email}"
    resp = requests.get(url, auth=(JIRA_EMAIL, JIRA_API_TOKEN))
    if resp.status_code != 200:
        raise HTTPException(status_code=resp.status_code, detail=f"Jira user lookup failed {resp.status_code}: {resp.text[:300]}")
    users = resp.json()
    if not users:
        raise HTTPException(status_code=404, detail=f"No Jira user found for {email}")
    return users[0]["accountId"]

def create_jira_issue(summary: str, description: str, account_id: str):
    url = f"{JIRA_BASE}/rest/api/3/issue"
    payload = {
        "fields": {
            "project": {"key": JIRA_PROJECT_KEY},
            "summary": summary[:255],
            "description": {
                "type": "doc",
                "version": 1,
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}]
            },
            "issuetype": {"name": "Task"},
            "assignee": {"id": account_id}
        }
    }
    resp = requests.post(url, json=payload, auth=(JIRA_EMAIL, JIRA_API_TOKEN))
    if resp.status_code != 201:
        raise HTTPException(status_code=resp.status_code, detail=f"Jira error {resp.status_code}: {resp.text[:300]}")
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

# ---- File handling ----
VIDEO_EXTENSIONS = {'.mp4', '.webm', '.mov', '.avi', '.mkv', '.m4v'}

async def save_and_prepare(file: UploadFile):
    """
    Save the uploaded file, extract audio via ffmpeg if it's a video.
    Returns (audio_path, [temp_paths_to_cleanup]).
    """
    ext = os.path.splitext(file.filename or '')[1].lower() or '.wav'
    uid = uuid.uuid4().hex
    raw_path = f"tmp_{uid}{ext}"

    with open(raw_path, "wb") as f:
        f.write(await file.read())

    if ext in VIDEO_EXTENSIONS:
        wav_path = f"tmp_{uid}.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path,
                 "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                 wav_path],
                check=True, capture_output=True
            )
        except subprocess.CalledProcessError as e:
            os.remove(raw_path)
            raise HTTPException(status_code=422, detail="ffmpeg failed to extract audio from video")
        return wav_path, [raw_path, wav_path]

    return raw_path, [raw_path]

# ---- API Endpoints ----

@app.get("/meetings")
def list_meetings(db: Session = Depends(get_db)) -> List[dict]:
    rows = db.query(Meeting).order_by(Meeting.created_at.desc()).limit(50).all()
    return [
        {
            "id": m.id,
            "created_at": m.created_at.isoformat() if m.created_at else None,
            "name": m.name or "",
            "summary": m.summary,
            "tasks_json": m.tasks_json,
        }
        for m in rows
    ]

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), name: str = Form("")):
    """
    Step 1 (draft): Transcribe audio/video, create a draft Meeting row
    with transcript only, and return its id.
    """
    audio_path, tmp_files = await save_and_prepare(file)
    try:
        text = transcribe_audio(audio_path)
    finally:
        for p in tmp_files:
            if os.path.exists(p):
                os.remove(p)
    logger.info("E2E DEBUG /transcribe transcript_preview=%r", (text or "")[:150])

    db = SessionLocal()
    try:
        meeting = Meeting(
            name=name.strip() or "",
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

    return {"id": meeting_id, "transcript": text, "name": name.strip() or ""}

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

    # Auto-create Jira tickets for tasks whose owner matches a team member
    if JIRA_BASE and JIRA_EMAIL and JIRA_API_TOKEN:
        try:
            db2 = SessionLocal()
            team = db2.query(TeamMember).all()
            db2.close()
            for t in tasks:
                owner = (t.get("owner") or "").strip().lower()
                if not owner:
                    continue
                team_names = [m.name.strip().lower() for m in team]
                close = difflib.get_close_matches(owner, team_names, n=1, cutoff=0.75)
                match = team[team_names.index(close[0])] if close else None
                if match:
                    task_text = t.get("task") or t.get("sentence", "")
                    try:
                        account_id = get_jira_account_id(match.email)
                        issue = create_jira_issue(task_text, f"Auto-assigned from meeting #{body.id}", account_id)
                        send_email(
                            match.email,
                            f"New Jira Ticket Assigned: {issue['key']}",
                            f"Hi {match.name},\n\nYou have been auto-assigned a task from a meeting.\n\n"
                            f"Task: {task_text}\n"
                            f"Ticket: {JIRA_BASE}/browse/{issue['key']}"
                        )
                    except Exception:
                        pass  # best-effort per task
        except Exception:
            pass  # don't fail the whole request

    if SLACK_WEBHOOK_URL:
        try:
            meeting_label = meeting.name if meeting.name else f"Meeting on {meeting.created_at.strftime('%b %d') if meeting.created_at else f'#{body.id}'}"
            lines = [f"*{meeting_label}: Action Items*"]
            for t in tasks:
                owner = t.get("owner") or "Unassigned"
                task = t.get("task") or t.get("sentence", "")
                deadline = t.get("deadline", "")
                deadline_str = f", due {deadline[:10]}" if deadline else ""
                lines.append(f"• *{owner}*: {task}{deadline_str}")
            requests.post(SLACK_WEBHOOK_URL, json={"text": "\n".join(lines)}, timeout=5)
        except Exception:
            pass  # Slack notification is best-effort

    return {"id": body.id, "summary": summary, "tasks": tasks}

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    One-shot path: transcribe -> NLP -> save full record.
    """
    audio_path, tmp_files = await save_and_prepare(file)
    try:
        transcript = transcribe_audio(audio_path)
    finally:
        for p in tmp_files:
            if os.path.exists(p):
                os.remove(p)
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

@app.get("/team")
def list_team(db: Session = Depends(get_db)):
    members = db.query(TeamMember).order_by(TeamMember.name).all()
    return [{"id": m.id, "name": m.name, "email": m.email} for m in members]

@app.post("/team")
def add_team_member(data: TeamMemberIn, db: Session = Depends(get_db)):
    existing = db.query(TeamMember).filter(TeamMember.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="A team member with that email already exists")
    member = TeamMember(name=data.name.strip(), email=data.email.strip())
    db.add(member)
    db.commit()
    db.refresh(member)
    return {"id": member.id, "name": member.name, "email": member.email}

@app.delete("/team/{member_id}")
def delete_team_member(member_id: int, db: Session = Depends(get_db)):
    member = db.get(TeamMember, member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Team member not found")
    db.delete(member)
    db.commit()
    return {"status": "deleted"}

