# app/nlp.py
from __future__ import annotations
from transformers import pipeline
import spacy
import dateparser
from typing import List, Dict, Any, Optional
import re
from datetime import datetime
from functools import lru_cache

# ---- Load models once (fast for API requests) ----
@lru_cache(maxsize=1)
def get_classifier():
    # Zero-shot classifier for intent
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@lru_cache(maxsize=1)
def get_nlp():
    # spaCy English model for NER + sentence split
    return spacy.load("en_core_web_sm")

INTENT_LABELS = ["commitment", "request", "information"]

# Basic verb list to help summarize the "task"
ACTION_VERB_HINTS = {
    "send","share","submit","finish","complete","review","deploy","fix",
    "update","create","schedule","prepare","write","summarize","follow","follow-up","followup",
    "email","call","meet","implement","investigate","test","ship","publish","push"
}

PRONOUNS = {"I","We","You","He","She","They","i","we","you","he","she","they"}

def _normalize_dt(text: str, ref: Optional[datetime] = None) -> Optional[str]:
    """Return ISO date string if parsed; else None."""
    dt = dateparser.parse(text, settings={"RELATIVE_BASE": ref or datetime.now()})
    return dt.isoformat() if dt else None

def _owner_from_sentence(doc: "spacy.tokens.Doc") -> Optional[str]:
    # Prefer PERSON entities
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    if persons:
        return persons[0]

    # Fall back to first pronoun subject-ish token
    for token in doc:
        if token.text in PRONOUNS:
            return token.text

    return None

def _task_phrase(doc: "spacy.tokens.Doc") -> Optional[str]:
    # Try to build "verb + object" phrase (simple, good enough for MVP)
    # e.g., "send the draft", "finish the report"
    verbs = [t for t in doc if t.pos_ == "VERB" or t.lemma_ in ACTION_VERB_HINTS]
    if not verbs:
        return None
    v = verbs[0]
    # Collect direct object + compound + modifiers
    obj = None
    for child in v.children:
        if child.dep_ in {"dobj","obj"}:
            obj = child
            break

    phrase = v.lemma_
    if obj:
        # include determiners/compounds around the object
        left = " ".join(w.text for w in obj.lefts if w.dep_ in {"det","amod","compound"})
        right = " ".join(w.text for w in obj.rights if w.dep_ in {"pobj","dobj","attr","amod","compound"})
        obj_chunk = " ".join(x for x in [left, obj.text, right] if x)
        phrase = f"{v.lemma_} {obj_chunk}"

    return phrase

def _deadline_candidates(doc: "spacy.tokens.Doc") -> List[str]:
    cands = [ent.text for ent in doc.ents if ent.label_ in {"DATE","TIME"}]
    # also grab simple patterns like "EOD", "end of day", "next week", "by Friday"
    text = doc.text
    extra = re.findall(r"\b(EOD|end of day|by\s+\w+day|next week|next month|in \d+ (days|weeks))\b", text, re.IGNORECASE)
    cands.extend([" ".join(t) if isinstance(t, tuple) else t for t in extra])
    return list(dict.fromkeys(cands))  # unique order-preserving

def classify_intent(sentence: str) -> str:
    clf = get_classifier()
    res = clf(sentence, INTENT_LABELS)
    # top label
    return res["labels"][0]

def extract_tasks(transcript: str, reference_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
    nlp = get_nlp()
    tasks: List[Dict[str, Any]] = []

    # Segment to sentences
    doc = nlp(transcript)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

    for sent in sentences:
        intent = classify_intent(sent)
        if intent not in ("commitment","request"):
            # ignore purely informational lines
            continue

        sdoc = nlp(sent)

        owner = _owner_from_sentence(sdoc)
        task = _task_phrase(sdoc)

        # deadlines
        deadline_iso = None
        for cand in _deadline_candidates(sdoc):
            normalized = _normalize_dt(cand, ref=reference_time)
            if normalized:
                deadline_iso = normalized
                break

        tasks.append({
            "sentence": sent,
            "type": intent,          # "commitment" or "request"
            "owner": owner,          # "I", "Alice", etc.
            "task": task,            # "send the draft"
            "deadline": deadline_iso # ISO8601 string or None
        })

    return tasks

def summarize(transcript: str, max_len: int = 130, min_len: int = 50) -> str:
    # Light-weight summarizer; can swap models later
    from transformers import pipeline
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Chunk long text to avoid token limits
    chunks = []
    nlp = get_nlp()
    buffer = []
    count = 0
    for s in nlp(transcript).sents:
        buffer.append(s.text.strip())
        count += len(s.text.split())
        if count > 350:  # rough chunk size
            chunks.append(" ".join(buffer))
            buffer, count = [], 0
    if buffer:
        chunks.append(" ".join(buffer))

    summaries = []
    for ch in chunks:
        out = summarizer(ch, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
        summaries.append(out)
    return " ".join(summaries) if summaries else ""
