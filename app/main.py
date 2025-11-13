# app/main.py
import re
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AskResponse
from .retrieval import store
from .utils import (
    detect_topic,
    extract_candidate_name,
    normalize_text,
    has_dateish,
    has_quantityish,
    extract_focus_terms,  
)
from .llm import synthesize_answer

app = FastAPI(title="Member QA Service")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or restrict to your UI domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    store._warm_background()

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/health", "/ask?question=..."]}

@app.get("/ask")
def ask(question: str = Query(..., description="Natural language question"), debug: bool = Query(False, description="Return debugging info")):
    q = normalize_text(question)

    # Make upstream fetch failures graceful
    try:
        store.ensure_fresh()
    except Exception:
        return JSONResponse(
            content={"answer": "I couldn’t reach the messages API right now. Please try again in a moment."}
        )

    # Person + topic
    name, _ = extract_candidate_name(q, store.user_names)
    topic = detect_topic(q)

    # Retrieval
    query = q if topic == "general" else f"{q} topic:{topic}"
    snippets = store.search(query, user_name=name, top_k=10)

    # -------- General coverage gate (focus terms) --------

    focus = extract_focus_terms(q, name)  # e.g., ["london"]
    if focus:
        def _has_focus_in_same_snippet(s):
            text = f"{s['user_name']} {s['timestamp']} {s['message']}".lower()
            same_user = (name is None) or (s["user_name"] == name)
            has_focus = any(f in text for f in focus)
            return same_user and has_focus

        if not any(_has_focus_in_same_snippet(s) for s in snippets):
            return JSONResponse(
                content={"answer": "I don’t have enough information to answer from the messages.(no message with focus terms)"}
            )

    # -------- Type guards (keep these) --------
    low = q.lower()

    # "when" questions must have a date-like token in evidence
    if "when" in low and not any(has_dateish(s["message"]) for s in snippets):
        return JSONResponse(
            content={"answer": "I don’t have enough information to answer when. I couldn’t find a date in the messages.(type guard when)"}
        )

    # "how many/how much" must have numeric evidence (digits or number-words)
    if ("how many" in low or "how much" in low):
        evidence_text = " ".join(s["message"] for s in snippets)
        if not has_quantityish(evidence_text):
            return JSONResponse(
                content={"answer": "I don’t have enough information to answer the quantity from the messages.(type guard how many/much)"}
            )

    # No evidence at all
    if not snippets:
        return JSONResponse(content={"answer": "I don’t have enough information to answer from the messages.(no evidence at all)"})

    # Synthesize with LLM (grounded; SYSTEM prompt already forbids guessing)
    answer = synthesize_answer(q, snippets)
    if debug:
        # answer is a dict when debug=True (per code above)
        return {
            "question": question,
            "person_detected": name,
            "topic": topic,
            "focus_terms": focus,
            "snippets_checked": [
                {"user": s["user_name"], "ts": s["timestamp"], "msg": s["message"]} for s in snippets
            ],
            "llm_debug": answer,               # contains candidate, verifier_reason, verifier_raw, final
            "answer": answer.get("final") if isinstance(answer, dict) else answer,
        }

    return JSONResponse(content={"answer": answer})
