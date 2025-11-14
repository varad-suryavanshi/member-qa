# Member QA — Question Answering API

### **Live API:** [https://member-qa-933867666784.us-central1.run.app](https://member-qa-933867666784.us-central1.run.app)

### **Simple UI:** [https://varad-suryavanshi.github.io/member-qa/](https://varad-suryavanshi.github.io/member-qa/)

This service answers natural-language questions about member data exposed by the public `https://november7-730026606190.europe-west1.run.app/messages` API the assignment provides.

Examples:

* “When is Layla planning her trip to London?”
* “How many cars does Vikram Desai have?”
* “What are Amira’s favorite restaurants?”

---

## Tech Stack

* **Language / Framework:** Python 3.12, FastAPI, Uvicorn
* **Deploy:** Google Cloud Run (public, stateless)
* **Retrieval / Ranking:**

  * BM25 (rank-bm25) + MiniLM embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
  * Reciprocal Rank Fusion (RRF) to merge lexical + semantic recall
  * Cross-Encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) for final ordering
* **LLM (answer synthesis):**
  * **Groq** API with **Llama 3.1 8B Instant** (`GROQ_MODEL=llama-3.1-8b-instant`)
  * Used to *format* the final answer with strict grounding rules (no external knowledge)
* **Heuristics:** light name detection, topic hints, and type guards (“when”, “how many/how much”)
* **Upstream data:** `GET /messages` from the assignment’s API (see Swagger)

---

## How to Use

### Health check

```bash
GET https://member-qa-933867666784.us-central1.run.app/health
# -> {"ok": true}
```

### Ask a question

```bash
GET https://member-qa-933867666784.us-central1.run.app/ask?question=When%20is%20Layla%20planning%20her%20trip%20to%20London%3F
# -> {"answer": "..."}
```

* URL-encode your question (spaces → `%20`, etc.).
* The service returns exactly `{ "answer": "..." }`.
* If there isn’t enough grounded evidence, it returns:

  ```
  {"answer":"I don’t have enough information to answer from the messages.(...)"} 
  ```

### Debug mode (for reviewers)

Add `&debug=true` to see how the answer was produced (retrieved snippets, detected person/topic, and LLM debug if applicable):

```bash
GET https://member-qa-933867666784.us-central1.run.app/ask?question=How%20many%20cars%20does%20Vikram%20Desai%20have%3F&debug=true
```

### Optional web UI

Use the hosted static page (no setup needed):
**[https://varad-suryavanshi.github.io/member-qa/](https://varad-suryavanshi.github.io/member-qa/)**
Type a question, toggle **Debug**, and it calls the API above.

---

## What the Service Does (Overview)

1. **Fetch member messages** from the assignment’s public API.
2. **Recall candidates** with two signals:

   * **BM25** over tokenized text
   * **MiniLM embeddings** cosine similarity (boosting the named user if detected)
3. **Fuse** both rankings using **RRF** and take top candidates.
4. **Rerank** with a **Cross-Encoder** for final ordering.
5. **Answering logic**:

   * If the question is “when”, require a date-ish string in the evidence.
   * If “how many / how much”, require a numeric in the evidence.
   * Otherwise return the best supported span or say “not enough information.”
6. **Debug mode** shows the retrieved snippets and internal detections to make review easy.

---

## Endpoints We Consume

* Assignment API Swagger:
  `GET /messages` — [https://november7-730026606190.europe-west1.run.app/docs#/default/get_messages_messages__get](https://november7-730026606190.europe-west1.run.app/docs#/default/get_messages_messages__get)

The service is resilient to redirects and minor content-type hiccups and retries briefly on transient failures.

---

## Bonus 1 — Design Notes

I explored several approaches; here’s what worked (and what didn’t):

### A) Plain semantic search → LLM “RAG” answer (didn’t perform well)

* I tried embedding-only retrieval and fed the top hits into a strict RAG prompt that **forbids guessing** and enforces type-precise output.
* Prompt (system) I used:

```
You are an excellent, precise assistant. You answer QUESTIONS using ONLY the EVIDENCE provided
in the prompt. You must never guess or rely on outside knowledge.

Your job:
- Read the retrieved member messages (EVIDENCE).
- If the evidence explicitly contains the answer, return it.
- If the evidence does not clearly contain the answer, reply exactly:
  "I don’t have enough information to answer."

Answering policy (follow ALL):
1) Grounding only:
   - Use ONLY the provided EVIDENCE. Do not infer, assume, or hallucinate.
   - If the question names a specific entity/term (e.g., a place, venue, phone number) that does not appear
     in the evidence, return the fallback.

2) Fidelity to the text:
   - Prefer exact strings, numbers, and dates that appear in the EVIDENCE.
   - Do not reformat phone numbers or alter proper nouns unless the question requires normalization.

3) Recency on conflicts:
   - If multiple messages conflict, select the information from the message with the LATEST timestamp.
   - If conflict cannot be resolved, use the fallback.

4) Type-precise answers:
   - If the question asks “when”, you must output a date-like string that appears in the EVIDENCE
     (weekday/month name/ISO date). If none exists, use the fallback.
   - If the question asks “how many / how much”, you must output a numeric value found in the EVIDENCE.
     If none exists, use the fallback.
   - If the question asks “which/what (entity)”, return the exact entity string from the EVIDENCE.
     If no single clear entity is present, use the fallback.

5) Partial/ambiguous evidence:
   - If the EVIDENCE is partially related but does not directly answer the question, use the fallback.
   - If multiple candidates exist and the EVIDENCE does not make one answer unambiguously correct, use the fallback.

6) No meta commentary:
   - Do NOT mention that you were given evidence.
   - Do NOT explain your steps or cite messages.
   - Do NOT output bullet points, lists, or extra sentences.

7) Answer validation (final check):
   - Before returning, verify that your draft answer directly and fully addresses the user’s question (scope, entity, and type).
   - If it does not perfectly align with the question, replace it with:
     "I don’t have enough information to answer."

Formatting (strict):
- Output ONLY the final answer string on a single line.
- No preamble, no citations, no explanations.
- If unknown or unsupported by evidence, reply exactly:
  "I don’t have enough information to answer."
```

User prompt builder (excerpt):

```python
def build_user_prompt(question: str, snippets: list[dict]) -> str:
    lines = [f"- [{s['user_name']} at {s['timestamp']}] {s['message']}" for s in snippets]
    evidence = "\n".join(lines) if lines else "(none)"
    return (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE (each bullet is a retrieved message):\n{evidence}\n"
        "  - Example format per line:\n"
        "    - [<User Name> at <ISO Timestamp>] <Message text>\n\n"
        "CONSTRAINTS & RULES:\n"
        "1) Use ONLY the EVIDENCE above. Do not rely on external world knowledge.\n"
        "2) If the answer is not explicitly present, reply exactly:\n"
        "   \"I don’t have enough information to answer.\"\n"
        "3) If multiple messages conflict, prefer the information from the message with the latest timestamp.\n"
        "4) If the question asks “when”, the answer must include a date-like phrase present in the EVIDENCE "
        "(weekday/month/ISO date). If none exists, say you don’t have enough information.\n"
        "5) If the question asks “how many / how much”, return a numeric value found in the EVIDENCE. "
        "If none exists, say you don’t have enough information.\n\n"
        "INSTRUCTIONS FOR FORMATTING:\n"
        "- Return ONLY the final answer string. No extra words, no explanations, no quotes.\n"
        "- If unknown, reply exactly: \"I don’t have enough information to answer.\""
    )
```

**Why it underperformed:** the dataset is short messages with sparse signals; strict formatting + strict fallback caused many “not enough info” responses even when weak signals existed.

### B) ChromaDB / vector-DB with semantic top-k (also not ideal here)

* I tried dropping the dataset into ChromaDB. With small, noisy text, a pure vector search wasn’t consistently precise for questions like “how many”/“when”.

### C) Final approach (what I shipped)

* **Hybrid retrieval**: BM25 + MiniLM; fuse with **RRF** for better recall.
* **Rerank** with a **Cross-Encoder** to focus on the actual question.
* **Type guards**:

  * “when” → require a date-ish match in the evidence
  * “how many/how much” → require a numeric in the evidence
* **Name detection**: fuzzy match the person in the question; lightly boost that user’s messages.
* **Fallback**: if the evidence doesn’t support the asked type exactly, return the not-enough-info string.

This balanced precision/recall and behaved predictably under the assignment’s constraints.

---

## Bonus 2 — Data Insights (anomalies & inconsistencies)


Below are the main data quality patterns I observed, with concrete examples from the provided messages (UTC timestamps shown). The system’s type-guards and “latest-wins” policy are designed to handle these safely.

### 1) Relative dates (need normalization)

Many messages use relative time (“tonight”, “this Friday”, “next Tuesday”, “next month”), which are **not explicit dates** and must be resolved **relative to the message timestamp**—or else answered with the fallback.

* “Please book a private jet to Paris **for this Friday**.” — *Sophia Al-Farsi*, `2025-05-05T07:47:20Z`
* “Confirm my dinner … **tonight**.” — *Fatima El-Tahir*, `2024-11-14T20:03:44Z`
* “I need two tickets … **this Saturday**.” — *Armand Dupont*, `2025-03-09T02:25:23Z`
* “limo … **next Tuesday**.” — *Layla Kawaguchi*, `2025-04-03T03:42:01Z`
  **Mitigation:** Only answer “when” if an explicit date string exists; otherwise return “I don’t have enough information to answer.”

### 2) Temporal oddities (past date references)

Some booking requests reference dates that appear **earlier** than the message timestamp, which is inconsistent for literal scheduling.

* “Book … **on November 10**.” — *Hans Müller*, timestamp `2024-12-24T08:13:22Z`
  **Mitigation:** Prefer the latest timestamped fact; if the timeline is inconsistent, use the fallback.

### 3) PII/identifier numbers ≠ quantities

Numbers like phone, fax, membership IDs, card endings, and frequent-flyer numbers can be mistaken for counts:

* Phone/fax: `555-349-7841`, `987-654-3210`, `212-555-6051`, `431-555-2363`
* IDs: “HNW membership number: `001-235-789`”, “frequent flyer number `123456789`”
  **Mitigation:** For “how many/how much” questions, require a number **paired with a count noun** (“tickets”, “seats”, “cars”, “people”, etc.) and ignore PII-style patterns.

### 4) Unicode & apostrophes in names (tokenization risk)

Names include diacritics and apostrophes (e.g., **Müller**, **O’Sullivan**). Simple ASCII-only tokenization harms lexical recall and name matching.
**Mitigation:** Use Unicode-aware tokenization and/or ASCII folding for indexing (keep original text for display); keep fuzzy matching for names.

### 5) Evolving facts and preferences

Preferences and contact details change over time (seat preferences, amenities, phone numbers).
**Mitigation:** When multiple values exist, **pick the message with the latest timestamp**. If conflicting and not resolvable, return the fallback.

### 6) Vague periods vs. explicit dates

Phrases like “**first week of December**”, “**next month**”, “**on November 25/22/9**” appear with earlier timestamps (advance planning).
**Mitigation:** If an explicit date is present (e.g., “November 25”), return that exact string; if only a vague period exists, use the fallback for “when”.

### 7) Proper-noun entities (preserve exact strings)

Venue/restaurant names (Alinea, Le Bernardin, Nobu, The Fat Duck, The French Laundry, The Ivy) must be returned **verbatim**.
**Mitigation:** Answer “what/which” with the **exact surface form** from the message—no normalization or paraphrasing.

---

**Guardrails implemented in the service**

* **Type guards:**

  * “when” → require a date-like string in evidence; else fallback.
  * “how many/how much” → require a numeric tied to a count noun; else fallback.
* **Latest-wins rule:** prefer facts from the **most recent** message when conflicts exist.
* **Unicode-aware retrieval:** fuzzy name matching and indexing that tolerates diacritics/apostrophes.
* **No guessing:** strict fallback (“I don’t have enough information to answer.”) when evidence is partial, ambiguous, or inconsistent.


> If you want to inspect what the model saw before answering, hit the API with `&debug=true` to see the retrieved snippets and internal detections.

---

## Deploy / Run (notes)

* **Public deploy:** Cloud Run (1 min instance to reduce cold-start; request timeout 300s).
* **CORS:** open for demo.
* **Config:** the upstream API base is provided via `MESSAGES_URL_BASE` env var.
* **No secrets** are embedded; all runtime config is via Cloud Run environment variables.

Local dev (optional):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
# open http://localhost:8000/ask?question=...
```
