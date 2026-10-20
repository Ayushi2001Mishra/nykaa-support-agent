"""Tasks 11-12 - FastAPI: POST /ask, POST /add-document, GET /health and WS /ws/chat."""
import config  # noqa: F401  (sets telemetry env vars first)
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
import chat
import rag
from governance import BudgetExceeded
from guardrails import detect_injection
from logging_utils import log_request
from schemas import SupportResponse
from textutil import split_sentences


@asynccontextmanager
async def lifespan(app):
    rag.build_index()
    yield


app = FastAPI(title="Nykaa Domain Support Agent", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=20000)   # large cap so the BUDGET guard, not 422, rejects
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    trace_id: str
    session_id: str
    response: SupportResponse
    guardrails: Dict[str, Any]
    review: Optional[Dict[str, Any]] = None
    latency_ms: float


class AddDocRequest(BaseModel):
    doc_id: str = Field(..., pattern=r"^[a-z0-9_]{3,40}$")
    title: str = Field(..., min_length=3, max_length=120)
    text: str = Field(..., min_length=20, max_length=2000)


class AddDocResponse(BaseModel):
    doc_id: str
    chunks_in_collections: Dict[str, int]


def _extras(out):
    return {"guardrails": out["guardrails"], "tokens_est": out["tokens_est"],
            "cost_usd_est": out["cost_usd_est"], **out.get("timings", {})}


@app.get("/health")
def health():
    return {"status": "ok", "mock_llm": config.MOCK_LLM, "chunking": config.DEPLOYED_STRATEGY}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):                                         # sync def -> runs in FastAPI's threadpool
    trace_id, t0 = uuid.uuid4().hex, time.perf_counter()
    sid = req.session_id or trace_id
    try:
        out = chat.chat_turn(sid, req.question, trace_id)
    except BudgetExceeded as e:
        log_request(trace_id, "/ask", req.question, 413, t0, sid, error=str(e))
        raise HTTPException(status_code=413, detail=str(e))
    log_request(trace_id, "/ask", req.question, 200, t0, sid, **_extras(out))
    return AskResponse(trace_id=trace_id, session_id=sid, response=SupportResponse(**out["response"]),
                       guardrails=out["guardrails"], review=out["review"],
                       latency_ms=round((time.perf_counter() - t0) * 1000, 2))


@app.post("/add-document", response_model=AddDocResponse)
def add_document(req: AddDocRequest):
    trace_id, t0 = uuid.uuid4().hex, time.perf_counter()
    if not 2 <= len(split_sentences(req.text)) <= 5:
        log_request(trace_id, "/add-document", req.text, 422, t0, error="2-5 sentences required")
        raise HTTPException(422, "documents must be 2-5 sentences")
    if detect_injection(req.text):                                # don't let a poisoned doc into the KB
        log_request(trace_id, "/add-document", req.text, 400, t0, error="injection in document")
        raise HTTPException(400, "document rejected: prompt-injection pattern detected")
    counts = rag.add_document(req.model_dump())
    log_request(trace_id, "/add-document", req.text, 200, t0, doc_id=req.doc_id)
    return AddDocResponse(doc_id=req.doc_id, chunks_in_collections=counts)


@app.websocket("/ws/chat")
async def ws_chat(ws: WebSocket, session_id: Optional[str] = None):
    await ws.accept()
    sid = session_id or uuid.uuid4().hex
    try:
        while True:
            text = await ws.receive_text()
            trace_id, t0 = uuid.uuid4().hex, time.perf_counter()
            try:
                out = await run_in_threadpool(chat.chat_turn, sid, text, trace_id)
                log_request(trace_id, "/ws/chat", text, 200, t0, sid, **_extras(out))
                await ws.send_json({"trace_id": trace_id, "response": out["response"],
                                    "guardrails": out["guardrails"]})
            except WebSocketDisconnect:
                raise
            except BudgetExceeded as e:
                log_request(trace_id, "/ws/chat", text, 413, t0, sid, error=str(e))
                await ws.send_json({"trace_id": trace_id, "error": str(e)})
            except Exception as e:                                # one bad message must not kill the socket
                log_request(trace_id, "/ws/chat", text, 500, t0, sid, error=repr(e))
                await ws.send_json({"trace_id": trace_id, "error": "internal error"})
    except (WebSocketDisconnect, RuntimeError):                   # client left (mid-turn or between turns)
        log_request(uuid.uuid4().hex, "/ws/chat", "[client disconnected]", 499, time.perf_counter(), sid)
