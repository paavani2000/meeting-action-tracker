# # app/nlp.py
# from __future__ import annotations
# from transformers import pipeline
# import spacy
# import dateparser
# from typing import List, Dict, Any, Optional
# import re
# from datetime import datetime
# from functools import lru_cache

# # ---- Load models once (fast for API requests) ----
# @lru_cache(maxsize=1)
# def get_classifier():
#     # Zero-shot classifier for intent
#     return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# @lru_cache(maxsize=1)
# def get_nlp():
#     # spaCy English model for NER + sentence split
#     return spacy.load("en_core_web_sm")

# INTENT_LABELS = ["commitment", "request", "information"]

# # Basic verb list to help summarize the "task"
# ACTION_VERB_HINTS = {
#     "send","share","submit","finish","complete","review","deploy","fix",
#     "update","create","schedule","prepare","write","summarize","follow","follow-up","followup",
#     "email","call","meet","implement","investigate","test","ship","publish","push"
# }

# PRONOUNS = {"I","We","You","He","She","They","i","we","you","he","she","they"}

# def _normalize_dt(text: str, ref: Optional[datetime] = None) -> Optional[str]:
#     """Return ISO date string if parsed; else None."""
#     dt = dateparser.parse(text, settings={"RELATIVE_BASE": ref or datetime.now()})
#     return dt.isoformat() if dt else None

# def _owner_from_sentence(doc: "spacy.tokens.Doc") -> Optional[str]:
#     # Prefer PERSON entities
#     persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
#     if persons:
#         return persons[0]

#     # Fall back to first pronoun subject-ish token
#     for token in doc:
#         if token.text in PRONOUNS:
#             return token.text

#     return None

# def _task_phrase(doc: "spacy.tokens.Doc") -> Optional[str]:
#     # Try to build "verb + object" phrase (simple, good enough for MVP)
#     # e.g., "send the draft", "finish the report"
#     verbs = [t for t in doc if t.pos_ == "VERB" or t.lemma_ in ACTION_VERB_HINTS]
#     if not verbs:
#         return None
#     v = verbs[0]
#     # Collect direct object + compound + modifiers
#     obj = None
#     for child in v.children:
#         if child.dep_ in {"dobj","obj"}:
#             obj = child
#             break

#     phrase = v.lemma_
#     if obj:
#         # include determiners/compounds around the object
#         left = " ".join(w.text for w in obj.lefts if w.dep_ in {"det","amod","compound"})
#         right = " ".join(w.text for w in obj.rights if w.dep_ in {"pobj","dobj","attr","amod","compound"})
#         obj_chunk = " ".join(x for x in [left, obj.text, right] if x)
#         phrase = f"{v.lemma_} {obj_chunk}"

#     return phrase

# def _deadline_candidates(doc: "spacy.tokens.Doc") -> List[str]:
#     cands = [ent.text for ent in doc.ents if ent.label_ in {"DATE","TIME"}]
#     # also grab simple patterns like "EOD", "end of day", "next week", "by Friday"
#     text = doc.text
#     extra = re.findall(r"\b(EOD|end of day|by\s+\w+day|next week|next month|in \d+ (days|weeks))\b", text, re.IGNORECASE)
#     cands.extend([" ".join(t) if isinstance(t, tuple) else t for t in extra])
#     return list(dict.fromkeys(cands))  # unique order-preserving

# def classify_intent(sentence: str) -> str:
#     clf = get_classifier()
#     res = clf(sentence, INTENT_LABELS)
#     # top label
#     return res["labels"][0]

# def extract_tasks(transcript: str, reference_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
#     nlp = get_nlp()
#     tasks: List[Dict[str, Any]] = []

#     # Segment to sentences
#     doc = nlp(transcript)
#     sentences = [s.text.strip() for s in doc.sents if s.text.strip()]

#     for sent in sentences:
#         intent = classify_intent(sent)
#         if intent not in ("commitment","request"):
#             # ignore purely informational lines
#             continue

#         sdoc = nlp(sent)

#         owner = _owner_from_sentence(sdoc)
#         task = _task_phrase(sdoc)

#         # deadlines
#         deadline_iso = None
#         for cand in _deadline_candidates(sdoc):
#             normalized = _normalize_dt(cand, ref=reference_time)
#             if normalized:
#                 deadline_iso = normalized
#                 break

#         tasks.append({
#             "sentence": sent,
#             "type": intent,          # "commitment" or "request"
#             "owner": owner,          # "I", "Alice", etc.
#             "task": task,            # "send the draft"
#             "deadline": deadline_iso # ISO8601 string or None
#         })

#     return tasks

# def summarize(transcript: str, max_len: int = 130, min_len: int = 50) -> str:
#     # Light-weight summarizer; can swap models later
#     from transformers import pipeline
#     summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
#     # Chunk long text to avoid token limits
#     chunks = []
#     nlp = get_nlp()
#     buffer = []
#     count = 0
#     for s in nlp(transcript).sents:
#         buffer.append(s.text.strip())
#         count += len(s.text.split())
#         if count > 350:  # rough chunk size
#             chunks.append(" ".join(buffer))
#             buffer, count = [], 0
#     if buffer:
#         chunks.append(" ".join(buffer))

#     summaries = []
#     for ch in chunks:
#         out = summarizer(ch, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]
#         summaries.append(out)
#     return " ".join(summaries) if summaries else ""

# app/nlp.py
from __future__ import annotations
from transformers import pipeline
import spacy
import dateparser
from typing import List, Dict, Any, Optional
import re
from datetime import datetime
from functools import lru_cache
import logging
from datetime import datetime, timedelta


logger = logging.getLogger("meeting_tracker.nlp")

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
    """
    Parse a natural-language date and return ISO8601.
    If the parsed date/time is in the past (relative to `ref` or now),
    bump it forward to the next occurrence (7 days) so 'Friday' means
    the upcoming Friday, not the previous one.
    """
    base = ref or datetime.now()
    dt = dateparser.parse(
        text,
        settings={
            "RELATIVE_BASE": base,
            "PREFER_DATES_FROM": "future",  # bias toward future
            "RETURN_AS_TIMEZONE_AWARE": False,
        },
    )
    if not dt:
        return None

    # If we still landed in the past (dateparser can do that sometimes), roll forward a week.
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

        # --- MINIMAL CHANGE: don't skip on 'information'; still try to extract ---
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

        # Only append if we actually found something actionable
        if intent in ("commitment", "request") or owner or task or deadline_iso:
            item = {
                "sentence": sent,
                "type": intent,          # "commitment" / "request" / "information"
                "owner": owner,          # "I", "Alice", etc.
                "task": task,            # "send the draft"
                "deadline": deadline_iso # ISO8601 string or None
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
