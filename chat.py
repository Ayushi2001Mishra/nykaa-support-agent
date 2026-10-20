"""Task 8 - session memory via LangChain InMemoryChatMessageHistory + RunnableWithMessageHistory.
(LangChainDeprecationWarning about LangGraph persistence is expected and left unsilenced.)"""
import json
import threading
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
import pipeline
from guardrails import mask_pii

_STORE = {}
_CTX = threading.local()


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    return _STORE.setdefault(session_id, InMemoryChatMessageHistory())


def _last_order_in_history(history):
    for m in reversed(history):                           # newest human message that named an order
        if isinstance(m, HumanMessage):
            hit = pipeline.ORDER_ID_RE.search(m.content)
            if hit:
                return hit.group(0).upper()
    return None


def _respond(inputs: dict) -> str:
    question, history = inputs["input"], inputs.get("history", [])
    own = pipeline.ORDER_ID_RE.search(question)
    order_id = own.group(0).upper() if own else _last_order_in_history(history)   # state carried over turns
    out = pipeline.ask(question, order_id=order_id, trace_id=getattr(_CTX, "trace_id", None),
                       pii_fired=getattr(_CTX, "fired", None))
    out["resolved_order_id"] = order_id
    return json.dumps(out)


_chain = RunnableWithMessageHistory(RunnableLambda(_respond), get_history,
                                    input_messages_key="input", history_messages_key="history")


def chat_turn(session_id: str, question: str, trace_id: str = None) -> dict:
    masked, fired = mask_pii(question)                    # history only ever stores MASKED text
    _CTX.trace_id, _CTX.fired = trace_id, fired
    raw = _chain.invoke({"input": masked}, config={"configurable": {"session_id": session_id}})
    return json.loads(raw if isinstance(raw, str) else raw.content)


def reset_sessions():
    _STORE.clear()
