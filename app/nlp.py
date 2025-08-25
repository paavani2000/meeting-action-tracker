# app/nlp.py
from __future__ import annotations
import logging
from transformers import pipeline
import spacy
import dateparser
from typing import List, Dict, Any, Optional
import re
from datetime import datetime, timedelta
from functools import lru_cache

logger = logging.getLogger("meeting_tracker.nlp")

# ---- Load models once (fast for API requests) ----
@lru_cache(maxsize=1)
def get_classifier():
    # Zero-shot classifier for intent
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

@lru_cache(maxsize=1)
def get_nlp():
    # Prefer transformer NER if available; fallback to sm
    try:
        return spacy.load("en_core_web_trf")
    except Exception:
        return spacy.load("en_core_web_sm")

INTENT_LABELS = ["commitment", "request", "information"]

# Basic verb list to help summarize the "task"
ACTION_VERB_HINTS = {
    "send","share","submit","finish","complete","review","deploy","fix",
    "update","create","schedule","prepare","write","summarize","follow","follow-up","followup",
    "email","call","meet","implement","investigate","test","ship","publish","push","book","notify"
}

PRONOUNS = {"I","We","You","He","She","They","i","we","you","he","she","they"}

def _normalize_dt(text: str, ref: Optional[datetime] = None) -> Optional[str]:
    """
    Parse a natural-language date and return ISO8601.
    If parsed date/time is in the past, roll forward (7d) so 'Friday'
    means upcoming Friday, not the previous one.
    """
    base = ref or datetime.now()
    dt = dateparser.parse(
        text,
        settings={
            "RELATIVE_BASE": base,
            "PREFER_DATES_FROM": "future",
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )
    if not dt:
        return None
    if dt.date() < base.date():
        dt = dt + timedelta(days=7)
    return dt.isoformat()

def _owner_from_sentence(doc: "spacy.tokens.Doc") -> Optional[str]:
    # Prefer PERSON entities
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    logger.info("NLP DEBUG ents=%s", [(e.text, e.label_) for e in doc.ents])
    if persons:
        logger.info("NLP DEBUG owner_by_PERSON=%r", persons[0])
        return persons[0]

    # NEW: fallback to first proper noun (good for short sentences)
    for token in doc:
        if token.pos_ == "PROPN":
            logger.info("NLP DEBUG owner_by_PROPN=%r", token.text)
            return token.text

    # Fall back to first pronoun subject-ish token
    for token in doc:
        if token.text in PRONOUNS:
            logger.info("NLP DEBUG owner_by_PRONOUN=%r", token.text)
            return token.text

    logger.info("NLP DEBUG owner=None")
    return None

def _task_phrase(doc: "spacy.tokens.Doc") -> Optional[str]:
    # Try to build "verb + object" phrase (simple, good enough for MVP)
    # e.g., "send the draft", "finish the report"
    verbs = [t for t in doc if t.pos_ == "VERB" or t.lemma_ in ACTION_VERB_HINTS]
    logger.info("NLP DEBUG tokens=%s", [(t.text, t.lemma_, t.pos_, t.dep_) for t in doc])
    logger.info("NLP DEBUG candidate_verbs=%s", [(t.text, t.lemma_, t.pos_, t.dep_) for t in verbs])
    if not verbs:
        logger.info("NLP DEBUG task=None (no verbs)")
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

    logger.info("NLP DEBUG task_phrase=%r", phrase)
    return phrase

def _deadline_candidates(doc: "spacy.tokens.Doc") -> List[str]:
    cands = [ent.text for ent in doc.ents if ent.label_ in {"DATE","TIME"}]
    # also grab simple patterns like "EOD", "end of day", "next week", "by Friday"
    text = doc.text
    extra = re.findall(r"\b(EOD|end of day|by\s+\w+day|next week|next month|in \d+ (days|weeks))\b", text, re.IGNORECASE)
    cands.extend([" ".join(t) if isinstance(t, tuple) else t for t in extra])
    uniq = list(dict.fromkeys(cands))  # unique order-preserving
    logger.info("NLP DEBUG deadline_candidates=%s", uniq)
    return uniq

def classify_intent(sentence: str) -> str:
    clf = get_classifier()
    res = clf(sentence, INTENT_LABELS)
    top = res["labels"][0]

    # NEW: rule override — treat “will/shall + action verb” as a commitment
    if re.search(r"\b(will|shall)\b", sentence, re.IGNORECASE):
        lower = sentence.lower()
        if any(v in lower for v in ACTION_VERB_HINTS):
            logger.info("NLP DEBUG intent override -> commitment (rule hit)")
            return "commitment"

    logger.info("NLP DEBUG intent sentence=%r -> %s", sentence, top)
    return top

def extract_tasks(transcript: str, reference_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
    nlp = get_nlp()
    tasks: List[Dict[str, Any]] = []

    # Segment to sentences
    doc = nlp(transcript)
    sentences = [s.text.strip() for s in doc.sents if s.text.strip()]
    logger.info("NLP DEBUG sentences_count=%d", len(sentences))

    for i, sent in enumerate(sentences, 1):
        logger.info("NLP DEBUG [%d] sentence=%r", i, sent)
        intent = classify_intent(sent)
        logger.info("NLP DEBUG [%d] intent=%s", i, intent)

        # Keep the softened gate: we still try extraction even if 'information'
        sdoc = nlp(sent)

        owner = _owner_from_sentence(sdoc)
        task = _task_phrase(sdoc)

        # deadlines
        deadline_iso = None
        for cand in _deadline_candidates(sdoc):
            normalized = _normalize_dt(cand, ref=reference_time)
            logger.info("NLP DEBUG [%d] deadline cand=%r -> %r", i, cand, normalized)
            if normalized:
                deadline_iso = normalized
                break

        # Append if actionable
        if intent in ("commitment", "request") or owner or task or deadline_iso:
            item = {
                "sentence": sent,
                "type": intent,
                "owner": owner,
                "task": task,
                "deadline": deadline_iso
            }
            logger.info("NLP DEBUG [%d] extracted=%s", i, item)
            tasks.append(item)
        else:
            logger.info("NLP DEBUG [%d] no actionable info; not appending", i)

    logger.info("NLP DEBUG total_tasks=%d", len(tasks))
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
