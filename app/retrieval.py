# app/retrieval.py
import os, time, threading, requests, numpy as np, heapq, re
from typing import Optional
from rank_bm25 import BM25Okapi

MESSAGES_URL_BASE = os.getenv(
    "MESSAGES_URL_BASE",
    "https://november7-730026606190.europe-west1.run.app"
).rstrip("/")

_PATHS = ["/messages", "/messages/"]
_REFRESH_SEC = 600
_TIMEOUT = 30
_token_re = re.compile(r"[a-z0-9']+")

def _authed_get(url: str):
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
            try:
                body = getattr(e, 'response', None).text[:500] if getattr(e, 'response', None) else str(e)
            except Exception:
                body = str(e)
            print(f"[messages fetch failed] url={url} err={type(e).__name__}: {body}", flush=True)
            last_err = e
    raise last_err

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
        self.embedder = None
        self.reranker = None
        self._models_ready = False

        self._last_fetch = 0.0
        self.items = []
        self.texts = []
        self.embeddings = None
        self.user_names = []
        self.bm25 = None
        self.bm25_corpus = []

        self._lock = threading.Lock()

    def _ensure_models(self):
        """Load heavy models lazily, once."""
        if self._models_ready:
            return
        with self._lock:
            if self._models_ready:
                return
            os.environ.setdefault("HF_HOME", os.getenv("HF_HOME", "/var/tmp/hf-cache"))
            os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", os.getenv("SENTENCE_TRANSFORMERS_HOME", "/var/tmp/hf-cache"))
            from sentence_transformers import SentenceTransformer, CrossEncoder
            self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            self._models_ready = True

    def _fetch(self):
        """Refresh messages and indexes."""
        data = _safe_fetch_messages()
        items = data.get("items", [])
        texts = [f"{it['user_name']} | {it['timestamp']} | {it['message']}" for it in items]

        embs = self.embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True) if (self.embedder and texts) else np.zeros((0, 384), dtype=np.float32)
        corpus_tok = [_tokenize(t) for t in texts]
        bm25 = BM25Okapi(corpus_tok) if texts else None
        user_names = sorted({it["user_name"] for it in items})

        with self._lock:
            self.items = items
            self.texts = texts
            self.embeddings = embs
            self.bm25 = bm25
            self.bm25_corpus = corpus_tok
            self.user_names = list(user_names)
            self._last_fetch = time.time()

    def _warm_background(self):
        """Kick off a non-blocking warmup."""
        def _bg():
            try:
                self._ensure_models()
                self._fetch()
            except Exception as e:
                print(f"[warmup] non-fatal: {e}", flush=True)
        threading.Thread(target=_bg, daemon=True).start()

    def ensure_fresh(self):
        """Ensure models are loaded and data is recent."""
        self._ensure_models()
        if self.embeddings is None or (time.time() - self._last_fetch) > _REFRESH_SEC:
            self._fetch()

    def _embedding_topn(self, query: str, user_name: Optional[str], N: int = 100):
        if not self.embedder or self.embeddings is None or len(self.embeddings) == 0:
            return []
        qvec = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = (self.embeddings @ qvec).ravel()
        if user_name:
            mask = np.array([1.15 if it["user_name"] == user_name else 1.0 for it in self.items], dtype=np.float32)
            sims = sims * mask
        idx = np.argsort(-sims)[:N]
        return idx.tolist()

    def _bm25_topn(self, query: str, N: int = 100):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        idx = np.argsort(-scores)[:N]
        return idx.tolist()

    def search(self, query: str, user_name: Optional[str] = None, top_k: int = 10):
        """BM25 + embedding recall, RRF fuse, optional CrossEncoder rerank."""
        self.ensure_fresh()
        bm_idx = self._bm25_topn(query, N=100)
        em_idx = self._embedding_topn(query, user_name, N=100)

        ranks = {}
        for r, i in enumerate(bm_idx): ranks.setdefault(i, []).append(r + 1)
        for r, i in enumerate(em_idx): ranks.setdefault(i, []).append(r + 1)
        rrf_scores = _rrf(ranks, k=60)

        topM = heapq.nlargest(60, rrf_scores.items(), key=lambda kv: kv[1])
        cand_idx = [i for (i, _) in topM]
        if not cand_idx:
            return []

        if self._models_ready:
            from numpy import asarray
            pairs = [(query, self.texts[i]) for i in cand_idx]
            ce_scores = self.reranker.predict(pairs)
            order = np.argsort(-asarray(ce_scores))[:top_k]
            chosen = [cand_idx[j] for j in order]
        else:
            chosen = cand_idx[:top_k]

        results = []
        for i in chosen:
            it = self.items[i]
            results.append({
                "user_name": it["user_name"],
                "timestamp": it["timestamp"],
                "message": it["message"],
                "score": float(rrf_scores[i]),
            })
        return results

# Lazy singleton
_store_singleton = None
def get_store() -> MessageStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = MessageStore()
    return _store_singleton
