"""Part 1 - RAG core: two chunking strategies, two ChromaDB collections, calibrated
grounded generation (MOCK_LLM: extractive, answers ONLY from retrieved context)."""
import json
import os
import re
import config
import chromadb
from sentence_transformers import SentenceTransformer
from kb import DOCUMENTS
from textutil import FALLBACK, content_words, split_sentences

_model = None
_client = None
GEN_CALLS = {"count": 0}          # incremented every time the (mock) LLM generation step runs


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBED_MODEL)
    return _model


def embed(texts):
    return get_model().encode(list(texts), normalize_embeddings=True).tolist()


# ---------------- Task 3: two chunking strategies ----------------
def chunk_fixed(text, size=config.FIXED_SIZE, overlap=config.FIXED_OVERLAP):
    chunks, step, i = [], size - overlap, 0
    while i < len(text):
        chunks.append(text[i:i + size].strip())
        if i + size >= len(text):
            break
        i += step
    return [c for c in chunks if c]


def chunk_sentences(text):
    return split_sentences(text)


STRATEGIES = {"fixed": chunk_fixed, "sentence": chunk_sentences}


def get_collection(strategy):
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=config.CHROMA_DIR)
    return _client.get_or_create_collection(f"kb_{strategy}", metadata={"hnsw:space": "cosine"})


def index_documents(docs):
    """Upsert docs into BOTH collections (each strategy in its own collection)."""
    counts = {}
    for strat, fn in STRATEGIES.items():
        col = get_collection(strat)
        ids, texts, metas = [], [], []
        for d in docs:
            col.delete(where={"doc_id": d["doc_id"]})            # drop stale chunks of an updated doc
            for j, c in enumerate(fn(d["text"])):
                ids.append(f"{d['doc_id']}::{strat}::{j}")
                texts.append(c)
                metas.append({"doc_id": d["doc_id"], "title": d["title"], "chunk_index": j})
        if ids:
            col.upsert(ids=ids, documents=texts, embeddings=embed(texts), metadatas=metas)
        counts[strat] = col.count()
    return counts


def build_index():
    return index_documents(DOCUMENTS)


def add_document(doc):
    """Used by POST /add-document. Replaces an existing doc with the same id."""
    for i, d in enumerate(DOCUMENTS):
        if d["doc_id"] == doc["doc_id"]:
            DOCUMENTS[i] = doc
            break
    else:
        DOCUMENTS.append(doc)
    counts = index_documents([doc])
    from governance import CACHE
    CACHE.clear()                                                  # cached answers may now be stale
    return counts


def retrieve(query, strategy=None, k=config.TOP_K):
    col = get_collection(strategy or config.DEPLOYED_STRATEGY)
    r = col.query(query_embeddings=embed([query]), n_results=min(k, col.count()))
    return [{"text": t, "doc_id": m["doc_id"], "title": m["title"], "sim": round(1 - dist, 4)}
            for t, m, dist in zip(r["documents"][0], r["metadatas"][0], r["distances"][0])]


# ---------------- Task 4: empirical threshold calibration ----------------
IN_SCOPE_CAL = ["How many days do I have to return a lipstick?",
                "How long will a refund take for cash on delivery?",
                "Can I swap my sneakers for a bigger size?",
                "What happens if my payment failed but money was deducted?",
                "Who handles my complaint if the support agent cannot solve it?"]
OUT_SCOPE_CAL = ["Who won the cricket world cup?",
                 "Write a poem about the moon",
                 "What is the capital of France?"]


def calibrate(save=True):
    result = {}
    for strat in STRATEGIES:
        ins = {q: retrieve(q, strat, 1)[0]["sim"] for q in IN_SCOPE_CAL}
        outs = {q: retrieve(q, strat, 1)[0]["sim"] for q in OUT_SCOPE_CAL}
        lo, hi = min(ins.values()), max(outs.values())
        if lo <= hi:
            raise RuntimeError(f"[{strat}] clusters overlap (in-scope min {lo} <= out-of-scope max {hi}); "
                               "add/adjust calibration queries before choosing a threshold")
        result[strat] = {"in_scope": ins, "out_of_scope": outs,
                         "in_scope_min": lo, "out_of_scope_max": hi,
                         "threshold": round((lo + hi) / 2, 4)}
    if save:
        with open(config.CALIBRATION_PATH, "w") as f:
            json.dump(result, f, indent=2)
    return result


def get_threshold(strategy=None):
    strategy = strategy or config.DEPLOYED_STRATEGY
    if not os.path.exists(config.CALIBRATION_PATH):
        calibrate()
    with open(config.CALIBRATION_PATH) as f:
        return json.load(f)[strategy]["threshold"]


# ---------------- Task 4: grounded generation ----------------
def mock_generate(query, chunks):
    """Deterministic stand-in for an LLM: uses ONLY retrieved text, picks the (up to) two
    sentences that best overlap the question, keeps their original order."""
    GEN_CALLS["count"] += 1
    qw, seen, cands = set(content_words(query)), set(), []
    for rank, c in enumerate(chunks):
        for s in split_sentences(c["text"]):
            if s in seen:
                continue
            seen.add(s)
            cands.append((len(qw & set(content_words(s))), -rank, s, c["doc_id"]))
    best = sorted(cands, key=lambda x: (x[0], x[1]), reverse=True)[:2]
    if not best or best[0][0] == 0:
        best = cands[:1]
    order = {s: i for i, (_, _, s, _) in enumerate(cands)}
    best = sorted(best, key=lambda x: order[x[2]])
    sources = []
    for *_, d in best:
        if d not in sources:
            sources.append(d)
    return " ".join(s for _, _, s, _ in best), sources


def grounded_answer(query, strategy=None, k=config.TOP_K):
    strategy = strategy or config.DEPLOYED_STRATEGY
    chunks = retrieve(query, strategy, k)
    thr = get_threshold(strategy)
    top = chunks[0]["sim"] if chunks else 0.0
    ctx = [c["text"] for c in chunks]
    if top < thr:
        return {"grounded": False, "answer": FALLBACK, "sources": [], "top_sim": top,
                "threshold": thr, "context": ctx}
    answer, sources = mock_generate(query, chunks)
    return {"grounded": True, "answer": answer, "sources": sources, "top_sim": top,
            "threshold": thr, "context": ctx}
