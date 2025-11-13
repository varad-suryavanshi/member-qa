# app/llm.py
import os
import re
from typing import List, Dict
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

FALLBACK = "I don’t have enough information to answer."

# ------------------------------
# Provider/model config
# ------------------------------
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
_groq = Groq(api_key=os.getenv("GROQ_API_KEY", ""))



# ------------------------------
# Prompt builders (minimal, grounded)
# ------------------------------
SYSTEM = (
    "You are a precise assistant. Answer the QUESTION using ONLY the EVIDENCE. "
    f"If the answer is not clearly supported by the EVIDENCE, reply exactly: {FALLBACK}"
)

def build_user_prompt(question: str, snippets: List[Dict]) -> str:
    # Keep evidence short and plain. Enable redaction by swapping to _sanitize below if desired.
    lines = [f"- [{s['user_name']} at {s['timestamp']}] {s['message']}" for s in snippets]
    # lines = [f"- [{s['user_name']} at {s['timestamp']}] {_sanitize(s['message'])}" for s in snippets]  # <— enable to redact
    evidence = "\n".join(lines) if lines else "(none)"
    return (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE (relevant member messages):\n{evidence}\n\n"
        "Instructions:\n"
        "- Answer using only the EVIDENCE above, as ONE short sentence in natural third person.\n"
        "- Prefer starting with the person’s name exactly as it appears in the QUESTION when helpful.\n"
        f"- If the EVIDENCE does not clearly contain the answer, reply exactly: {FALLBACK}"
    )

# ------------------------------
# Small helpers
# ------------------------------
def _fb(reason: str) -> str:
    """Always-on reason-tagged fallback."""
    return f"{FALLBACK} ({reason})"

def _postprocess(text: str) -> str:
    ans = (text or "").strip()
    # strip enclosing quotes
    if (ans.startswith('"') and ans.endswith('"')) or (ans.startswith('“') and ans.endswith('”')):
        ans = ans[1:-1].strip()
    # keep short
    if len(ans) > 500:
        ans = ans[:500].rstrip()
    return ans



# ------------------------------
# Core function (always-on reasoned fallbacks)
# ------------------------------
def synthesize_answer(question: str, snippets: List[Dict]) -> str:
    # If no key, strict fallback with reason
    if not _groq.api_key:
        return _fb("no API/model key")

    prompt = build_user_prompt(question, snippets)

    try:
        chat = _groq.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=50000,
        )
        # print("GROQ full response:", chat)
        choice = chat.choices[0]
        # print("finish_reason:", getattr(choice, "finish_reason", None))
        # print("message:", getattr(choice, "message", None))
        raw = (chat.choices[0].message.content or "").strip()
        ans = _postprocess(raw)
        # print("GROQ raw answer:", raw)

        if not ans:
            return _fb("empty model output")

        return ans

    except Exception:
        return _fb("model-exception")
