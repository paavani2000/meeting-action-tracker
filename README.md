# 🎤 Meeting Action Tracker

A team tool that turns meeting recordings into structured action items automatically.

Upload a recording, get a transcript, extract who needs to do what by when, and push tasks directly to Jira. No more manual notes. No more dropped action items.

---

## 🧩 Overview

This project is structured into **two connected applications**:

* ⚙️ **Backend (FastAPI + NLP + PostgreSQL)**

  Processes audio and video with Whisper, extracts action items and summaries using NLP (spaCy + Transformers), persists results to PostgreSQL, and integrates with Jira and email.

* 🌐 **Frontend (React + Vite + Tailwind)**

  A team-facing web UI to upload recordings, view transcripts, summaries, and action item cards, browse past meetings, and trigger Jira ticket creation.

Together, they provide a seamless pipeline for **meeting intelligence** from raw audio or video to structured, assigned tasks.

---

## 🔍 Core Features

✅ **Speech-to-Text (STT)** - Transcribe meetings using OpenAI Whisper

✅ **Video Support** - .mp4, .mov, .webm and more, auto-converted via ffmpeg

✅ **Task Extraction** - Identify commitments and requests with owners, deadlines, and intent type

✅ **Summarization** - Generate concise meeting summaries

✅ **Jira Integration** - Create Jira tickets from action items, assigned by email

✅ **Email Notifications** - Notify assignees when a ticket is created

✅ **Persistence** - Save transcripts, summaries, and tasks to PostgreSQL

✅ **Frontend Upload** - Upload audio or video files directly from the browser

✅ **Meeting History** - Browse saved meetings and extracted tasks

---

## 🛠 Tech Stack

| Layer | Tech Details |
| --- | --- |
| 🎧 **STT** | OpenAI Whisper |
| 🧠 **NLP** | spaCy, Hugging Face Transformers, dateparser |
| ⚙️ **Backend** | FastAPI, SQLAlchemy, PostgreSQL |
| 🌐 **Frontend** | React, Vite, Tailwind CSS |
| 🎬 **Video** | ffmpeg |
| 🔗 **Integrations** | Jira REST API, SMTP email |

---

## 📦 Project Structure

```
MEETING-ACTION-TRACKER/
├─ backend/
│  └─ app/
│     ├─ main.py         # FastAPI entrypoint (transcribe, extract, Jira)
│     ├─ stt.py          # Speech-to-text + ffmpeg video extraction
│     ├─ nlp.py          # NLP pipeline (tasks + summary)
│     ├─ models.py       # SQLAlchemy models
│     ├─ db.py           # DB session + engine
│     └─ init_db.py      # Create tables
├─ frontend/
│  ├─ src/App.jsx        # React UI (upload, progress, results, history)
│  └─ package.json
├─ data/
│  └─ samples/           # Example audio files
├─ .env.example          # Environment variable template
├─ README.md
└─ requirements.txt
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.9+, Node.js 18+, PostgreSQL
- ffmpeg: `brew install ffmpeg` (Mac) or `apt install ffmpeg` (Linux)

### 1. Backend

```bash
cd backend
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Copy and fill in environment variables
cp ../.env.example .env

# Init database
python app/init_db.py

# Run server
uvicorn app.main:app --reload
```

API docs: http://127.0.0.1:8000/docs

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at http://127.0.0.1:5173

---

## 🛣 Roadmap

* [x] Phase 1 - Speech-to-Text (Whisper)
* [x] Phase 2 - Video support (ffmpeg)
* [x] Phase 3 - NLP (Summarization + Task Extraction)
* [x] Phase 4 - Persistence (PostgreSQL)
* [x] Phase 5 - Frontend (React upload, progress, results, history)
* [x] Phase 6 - Jira integration (ticket creation + email notifications)
* [ ] Phase 7 - Slack notifications (post action items to team channel)
* [ ] Phase 8 - Live browser recording (MediaRecorder API)

---

## 📡 External Services

* **OpenAI Whisper** (speech-to-text)
* **Hugging Face Transformers** (summarization, intent classification)
* **spaCy** (NER, sentence splitting)
* **Jira REST API** (ticket creation and assignment)
* **SMTP** (email notifications to assignees)
