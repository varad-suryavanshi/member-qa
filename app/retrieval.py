# app/retrieval.py
import os, time, threading, requests, numpy as np, heapq, re
from typing import Optional
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

MESSAGES_URL_BASE = os.getenv(
    "MESSAGES_URL_BASE",
    "https://november7-730026606190.europe-west1.run.app"
).rstrip("/")

_PATHS = ["/messages", "/messages/"]
_REFRESH_SEC = 600  # 10 min
_TIMEOUT = 30
_token_re = re.compile(r"[a-z0-9']+")

def _authed_get(url: str):
    # Target service is public — do a simple GET
    headers = {"User-Agent": "member-qa/1.0"}
    r = requests.get(url, timeout=_TIMEOUT, headers=headers, allow_redirects=False)
    if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
        loc = r.headers.get("location")
        if loc and loc.startswith("/"):
            loc = MESSAGES_URL_BASE + loc
        if loc:
            r = requests.get(loc, timeout=_TIMEOUT, headers=headers, allow_redirects=False)
    r.raise_for_status()
    return r

def _get_messages_once(url: str):
    r = _authed_get(url)
    return r.json()

def _safe_fetch_messages():
    last_err = None
    for p in _PATHS:
        url = MESSAGES_URL_BASE + p
        try:
            return _get_messages_once(url)
        except Exception as e:
            # Log why it failed (shows up in Cloud Run logs)
            try:
                body = getattr(e, 'response', None).text[:500] if getattr(e, 'response', None) else str(e)
            except Exception:
                body = str(e)
            print(f"[messages fetch failed] url={url} err={type(e).__name__}: {body}", flush=True)
            last_err = e
    raise last_err

# def _get_messages_once(url: str):
#     r = requests.get(url, timeout=30, allow_redirects=False, headers={"User-Agent": "member-qa/1.0"})
#     if r.is_redirect or r.status_code in (301, 302, 303, 307, 308):
#         loc = r.headers.get("location")
#         if loc:
#             if loc.startswith("/"):
#                 loc = MESSAGES_URL_BASE.rstrip("/") + loc
#             r = requests.get(loc, timeout=30, allow_redirects=False, headers={"User-Agent": "member-qa/1.0"})
#     r.raise_for_status()
#     return r.json()

# def _safe_fetch_messages():
#     last_err = None
#     for p in _PATHS:
#         try:
#             return _get_messages_once(MESSAGES_URL_BASE.rstrip("/") + p)
#         except Exception as e:
#             last_err = e
#     raise last_err

def _tokenize(text: str):
    return _token_re.findall(text.lower())

def _rrf(ranks, k=60):
    scores = {}
    for doc_id, pos_list in ranks.items():
        s = 0.0
        for pos in pos_list:
            s += 1.0 / (k + pos)
        scores[doc_id] = s
    return scores

class MessageStore:
    def __init__(self):
        # Lazy models
        self.embedder = None
        self.reranker = None
        self._models_ready = False

        # Data
        self._last_fetch = 0.0
        self.items = []
        self.texts = []
        self.embeddings = None   # np.ndarray [N, D]
        self.user_names = []
        self.bm25 = None
        self.bm25_corpus = []

        # Sync
        self._lock = threading.Lock()

    def _ensure_models(self):
        if self._models_ready:
            return
        with self._lock:
            if self._models_ready:
                return
            # Prefer HF_HOME over TRANSFORMERS_CACHE (TRANSFORMERS_CACHE is deprecated)
            os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "/var/tmp/hf-cache"))
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.getenv("SENTENCE_TRANSFORMERS_HOME", "/var/tmp/hf-cache"))
            # HUGGINGFACE_HUB_TOKEN is injected by Cloud Run secret env var if present
            # Load once
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._models_ready = True

    def _fetch(self):
        data = _safe_fetch_messages()
        items = data.get("items", [])
        texts = [f"{it['user_name']} | {it['timestamp']} | {it['message']}" for it in items]

        embs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True) if texts else np.zeros((0, 384), dtype=np.float32)
        corpus_tok = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(corpus_tok) if texts else None
        user_names = sorted(list({it["user_name"] for it in items}))

        with self._lock:
            self.items = items
            self.texts = texts
            self.embeddings = embs
            self.bm25 = bm25
            self.bm25_corpus = corpus_tok
            self.user_names = user_names
            self._last_fetch = time.time()

    def _warm_background(self):
        # Call from app startup; don't block boot
        def _bg():
            try:
                self._ensure_models()
                self._fetch()
            except Exception as e:
                print(f"[warmup] non-fatal: {e}", flush=True)
        threading.Thread(target=_bg, daemon=True).start()

    def ensure_fresh(self):
        self._ensure_models()
        if self.embeddings is None or (time.time() - self._last_fetch) > _REFRESH_SEC:
            self._fetch()

    def _embedding_topn(self, query: str, user_name: str | None, N: int = 100):
        qvec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = (self.embeddings @ qvec).ravel() if self.embeddings is not None and len(self.embeddings) else np.array([], dtype=np.float32)
        if user_name and len(sims):
            mask = np.array([1.15 if it["user_name"] == user_name else 1.0 for it in self.items], dtype=np.float32)
            sims = sims * mask
        idx = np.argsort(-sims)[:N] if len(sims) else np.array([], dtype=int)
        return idx.tolist()

    def _bm25_topn(self, query: str, N: int = 100):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        idx = np.argsort(-scores)[:N]
        return idx.tolist()

    def search(self, query: str, user_name: str | None = None, top_k: int = 10):
        self.ensure_fresh()
        bm_idx = self._bm25_topn(query, N=100)
        em_idx = self._embedding_topn(query, user_name, N=100)

        ranks = {}
        for r, i in enumerate(bm_idx): ranks.setdefault(i, []).append(r+1)
        for r, i in enumerate(em_idx): ranks.setdefault(i, []).append(r+1)
        rrf_scores = _rrf(ranks, k=60)

        topM = heapq.nlargest(60, rrf_scores.items(), key=lambda kv: kv[1])
        cand_idx = [i for (i, _) in topM]

        if not cand_idx:
            return []

        pairs = [(query, self.texts[i]) for i in cand_idx]
        ce_scores = self.reranker.predict(pairs)
        order = np.argsort(-np.asarray(ce_scores))[:top_k]

        results = []
        for j in order:
            i = cand_idx[j]
            it = self.items[i]
            results.append({
                "user_name": it["user_name"],
                "timestamp": it["timestamp"],
                "message": it["message"],
                "score": float(ce_scores[j]),
            })
        return results

# Create store without loading models
store = MessageStore()
