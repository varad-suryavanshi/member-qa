# app/utils.py
import re
from typing import List, Optional, Tuple
from rapidfuzz import fuzz, process


# -----------------------------
# Basic text helpers
# -----------------------------

def normalize_text(s: str) -> str:
    """Collapse whitespace and strip ends."""
    return " ".join(s.strip().split())

# -----------------------------
# Name extraction (fuzzy)
# -----------------------------

def extract_candidate_name(question: str, all_names: List[str]) -> Tuple[Optional[str], float]:
    """
    Fuzzy match the person's name from the question against known user names.
    Returns (best_name, score). Score ~0..100. Returns (None, 0.0) if nothing reasonable.
    """
    # Try longest capitalized span first; fallback to full question
    caps = re.findall(r"[A-Z][a-zA-Z'’\-]+(?:\s+[A-Z][a-zA-Z'’\-]+)*", question)
    queries = []
    if caps:
        # Use the longest capitalized span (likely the person's name)
        queries.append(max(caps, key=len))
    queries.append(question)

    best_name, best_score = None, 0.0
    for q in queries:
        match = process.extractOne(q, all_names, scorer=fuzz.token_set_ratio)
        if match:
            cand, score, _ = match
            if score > best_score:
                best_name, best_score = cand, float(score)
    # Only accept if reasonably confident
    if best_score >= 70.0:
        return best_name, best_score
    return None, 0.0

# -----------------------------
# Topic heuristic (optional)
# -----------------------------

def detect_topic(question: str) -> str:
    """
    Very light topic classifier for optional query decoration.
    """
    q = question.lower()
    if any(k in q for k in ["book", "flight", "hotel", "suite", "room", "villa", "check-in", "itinerary", "trip", "travel"]):
        return "travel"
    if any(k in q for k in ["restaurant", "dinner", "table", "reservation", "chef’s table", "chef's table"]):
        return "dining"
    if any(k in q for k in ["invoice", "billing", "charge", "payment", "renewal", "transaction", "points", "loyalty"]):
        return "billing"
    return "general"

# -----------------------------
# Focus terms (general coverage gate)
# -----------------------------



_GENERIC = {
    "what","which","who","when","where","why","how","much","many",
    "plan","planning","trip","travel","vacation","book","booking","secure","arrange","need","want",
    "restaurant","restaurants","dinner","reservation","reservations","table","tables","seat","seats",
    "family","please","thanks","thank","you","my","his","her","their",
    "the","a","an","at","for","to","in","on","with","of","and","or",
    "is","are","was","were","do","does","did"
}
_WORD = re.compile(r"[a-zA-Z][a-zA-Z\-']+")

def extract_focus_terms(question: str, person_name: Optional[str]) -> List[str]:
    """
    Extract specific 'focus' terms that should appear in the evidence for us to answer.
    Removes the person's name and generic utility words. Keeps quoted phrases and up to 5 items.
    """
    q = question.strip()

    # Keep quoted phrases verbatim
    quoted = re.findall(r'"([^"]+)"|“([^”]+)”|\'([^\']+)\'', q)
    phrases = [t for tup in quoted for t in tup if t]

    words = [w.lower() for w in _WORD.findall(q)]
    name_tokens = set(person_name.lower().split()) if person_name else set()

    focus_words: List[str] = []
    for w in words:
        if len(w) <= 2:
            continue
        if w in name_tokens:
            continue
        if w in _GENERIC:
            continue
        focus_words.append(w)

    # Prefer phrases first, then words; dedupe; cap length
    out: List[str] = []
    seen = set()
    for t in phrases + focus_words:
        t = t.strip().lower()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out[:5]


# -----------------------------
# Evidence-type detectors
# -----------------------------



# Compile once at import
_DATEISH = re.compile(
    r"""
    (?:\b\d{4}-\d{2}-\d{2}\b)                              # 2025-11-10
  | (?:\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b)            # 11/10, 11-10-2025
  | (?:\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)
        (?:uary|ch|ril|e|y|e|y|ust|tember|ober|ember)?
        \s+\d{1,2}(?:,\s*\d{4})?\b)                       # Nov 10, 2025 / November 10
  | (?:\b(?:today|tomorrow|tonight|tonite|yesterday)\b)   # relative day words
  | (?:\b(?:this|next|coming)?\s*
        (?:mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun
        |monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b)  # (this) Friday
  | (?:\b(?:this|next)\s+(?:week|weekend|month|quarter|year)\b)        # this weekend / next month
  | (?:\b(?:first|second|third|fourth)\s+week\s+of\s+
        (?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec
        |january|february|march|april|may|june|july|august|september|october|november|december)\b)  # first week of December
    """,
    re.IGNORECASE | re.VERBOSE,
)

def has_dateish(text: str) -> bool:
    return bool(_DATEISH.search(text or ""))


_NUMBER_WORDS = {
    "zero","one","two","three","four","five","six","seven","eight","nine","ten",
    "eleven","twelve","thirteen","fourteen","fifteen","sixteen","seventeen",
    "eighteen","nineteen","twenty","thirty","forty","fifty","sixty","seventy",
    "eighty","ninety","hundred","thousand","million"
}

def has_quantityish(text: str) -> bool:
    """True if the text contains a digit or a spelled-out number word."""
    t = text.lower()
    if re.search(r"\b\d+\b", t):
        return True
    return any(w in t for w in _NUMBER_WORDS)





