🎤 Meeting Action Tracker

An end-to-end solution for capturing, summarizing, and tracking meeting action items.
Upload meeting recordings, automatically generate transcripts, extract summaries and actionable tasks using NLP, and persist them to a database.
Includes a simple React frontend for uploading audio and browsing past meetings.

🧩 Overview

This project is structured into two connected applications:

⚙️ Backend (FastAPI + NLP + PostgreSQL/SQLite)
Processes audio with Whisper, extracts tasks with NLP (spaCy + transformers), summarizes text, and saves everything to the database.

🌐 Frontend (React + Vite + Tailwind)
A lightweight web UI to upload audio files, view transcripts, summaries, and tasks, and browse past meetings.

Together, they provide a seamless pipeline for meeting intelligence—from raw audio to structured tasks.

🔍 Core Features

✅ Speech-to-Text (STT) – Transcribe meetings using OpenAI Whisper
✅ Task Extraction – Identify commitments/requests with owners and deadlines
✅ Summarization – Generate concise meeting summaries
✅ Persistence – Save transcripts, summaries, and tasks to Postgres (or SQLite)
✅ Frontend Upload – Upload .wav or .m4a files from browser
✅ Meeting History – Browse saved meetings and extracted tasks
✅ Extensible – Ready for Slack/Jira integrations (Phase 5)

🛠 Tech Stack
Layer	Tech Details
🎧 STT	OpenAI Whisper
🧠 NLP	spaCy, Hugging Face Transformers, dateparser
⚙️ Backend	FastAPI, SQLAlchemy, PostgreSQL (or SQLite)
🌐 Frontend	React, Vite, Tailwind CSS
🔗 Integrations (planned)	Slack, Jira, email reminders
📦 Project Structure
MEETING-ACTION-TRACKER/
├─ backend/
│  └─ app/
│     ├─ main.py         # FastAPI entrypoint
│     ├─ stt.py          # Speech-to-text logic
│     ├─ nlp.py          # NLP pipeline (tasks + summary)
│     ├─ models.py       # SQLAlchemy models
│     ├─ db.py           # DB session + engine
│     └─ init_db.py      # Create tables
├─ frontend/
│  ├─ src/App.jsx        # React UI
│  └─ package.json
├─ data/
│  └─ samples/           # Example audio files
├─ README.md
└─ requirements.txt

🚀 Getting Started
1. Backend
cd backend
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
pip install -r requirements.txt

# init database
python app/init_db.py

# run server
uvicorn app.main:app --reload


API docs: http://127.0.0.1:8000/docs

2. Frontend
cd frontend
npm install
npm run dev


Frontend runs at http://127.0.0.1:5173

🛣 Roadmap

 Phase 1 – Speech-to-Text (Whisper)

 Phase 2 – NLP (Summarization + Task Extraction)

 Phase 3 – Persistence (Postgres/SQLite)

 Phase 4 – Frontend (React upload + history)

 Phase 5 – Integrations (Slack, Jira, reminders)

📡 External Services

OpenAI Whisper (speech-to-text)

Hugging Face Transformers (summarization, intent classification)

spaCy (NER, sentence splitting)

📖 License

MIT License – free to use, modify, and distribute.
