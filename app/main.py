# app/main.py
import re
from functools import lru_cache
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from .schemas import AskResponse
from . import retrieval  # ← import the module, not `store`
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@lru_cache(maxsize=1)
def get_store():
    return retrieval.get_store()   # lazy singleton from retrieval.py

@app.on_event("startup")
async def startup_event():
    # Don’t block startup; kick a tiny background warmup after bind
    import asyncio
    async def _warm():
        await asyncio.sleep(0.2)
        try:
            get_store()._warm_background()
        except Exception as e:
            print("[startup warm] non-fatal:", e, flush=True)
    asyncio.create_task(_warm())

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/")
def root():
    return {"ok": True, "endpoints": ["/health", "/ask?question=..."]}

@app.get("/ask", response_model=AskResponse)
def ask(
    question: str = Query(..., description="Natural language question"),
    debug: bool = Query(False, description="Return debugging info"),
):
    q = normalize_text(question)

    store = get_store()
    try:
        store.ensure_fresh()  # may fetch + first-time load, but only after bind
    except Exception:
        return JSONResponse(
            content={"answer": "I couldn’t reach the messages API right now. Please try again in a moment."}
        )

    name, _ = extract_candidate_name(q, store.user_names)
    topic = detect_topic(q)

    query = q if topic == "general" else f"{q} topic:{topic}"
    snippets = store.search(query, user_name=name, top_k=10)

    # focus/type guards unchanged…
    low = q.lower()
    focus = extract_focus_terms(q, name)
    if focus:
        def _has_focus_in_same_snippet(s):
            text = f"{s['user_name']} {s['timestamp']} {s['message']}".lower()
            same_user = (name is None) or (s["user_name"] == name)
            return same_user and any(f in text for f in focus)
        if not any(_has_focus_in_same_snippet(s) for s in snippets):
            return JSONResponse(content={"answer": "I don’t have enough information to answer from the messages.(no message with focus terms)"})

    if "when" in low and not any(has_dateish(s["message"]) for s in snippets):
        return JSONResponse(content={"answer": "I don’t have enough information to answer when. I couldn’t find a date in the messages.(type guard when)"})

    if ("how many" in low or "how much" in low):
        evidence_text = " ".join(s["message"] for s in snippets)
        if not has_quantityish(evidence_text):
            return JSONResponse(content={"answer": "I don’t have enough information to answer the quantity from the messages.(type guard how many/much)"})

    if not snippets:
        return JSONResponse(content={"answer": "I don’t have enough information to answer from the messages.(no evidence at all)"})

    answer = synthesize_answer(q, snippets)
    if debug and isinstance(answer, dict):
        return {
            "question": question,
            "person_detected": name,
            "topic": topic,
            "focus_terms": focus,
            "snippets_checked": [
                {"user": s["user_name"], "ts": s["timestamp"], "msg": s["message"]} for s in snippets
            ],
            "llm_debug": answer,
            "answer": answer.get("final"),
        }

    return JSONResponse(content={"answer": answer if not isinstance(answer, dict) else answer.get("final")})
