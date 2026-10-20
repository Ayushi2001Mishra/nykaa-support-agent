"""Pure parsing helpers for MockLLM (kept separate so they are unit-testable without CrewAI).

Both silent pitfalls from the brief are handled here:
 (1) the observation is parsed ONLY from the conversation AFTER the task prompt, and only
     where it follows the model's own 'Action Input:' text - never from CrewAI's ReAct
     system-prompt template (which literally contains 'Observation: the result of the action').
 (2) tools are dispatched from their declared ARGUMENT SCHEMA (record_id / query), never by
     matching substrings of the tool name (so 'rag_lookup' is not mistaken for a lookup tool).
"""
import ast
import json
import re
from typing import List, Optional, Set, Tuple
from textutil import FALLBACK, order_sentences

_META = {"description", "type", "title", "default", "enum", "items", "anyOf", "properties", "required"}


def arg_names(arg_str: str) -> Set[str]:
    for loader in (ast.literal_eval, json.loads):
        try:
            d = loader(arg_str)
            if isinstance(d, dict):
                props = d.get("properties")
                return set(props) if isinstance(props, dict) else set(d)
        except Exception:
            pass
    return {k for k in re.findall(r"['\"](\w+)['\"]\s*:", arg_str) if k not in _META}


def declared_tools(system_text: str) -> List[Tuple[str, Set[str]]]:
    pat = re.compile(r"Tool Name:\s*(\S+)\s*\n\s*Tool Arguments:\s*(.*?)\n\s*Tool Description:", re.S)
    return [(m.group(1), arg_names(m.group(2))) for m in pat.finditer(system_text)]


def last_observation(tail: str) -> Optional[str]:
    """tail = every non-system message AFTER the first (task) message."""
    found = re.findall(r"Action Input:.*?\n\s*Observation:\s*(.*)", tail, re.S)
    return found[-1].strip() if found else None


def extract_question(text: str) -> str:
    m = re.search(r"QUESTION: \[\[(.*?)\]\]", text, re.S)
    return m.group(1).strip() if m else ""


def extract_order_id(text: str) -> Optional[str]:
    m = re.search(r"ORDER_ID: \[\[(NYK-\d+)\]\]", text)
    return m.group(1) if m else None


def extract_marker(text: str, marker: str) -> Optional[dict]:
    hits = re.findall(re.escape(marker) + r"::(\{.*\})", text)
    for raw in reversed(hits):
        try:
            return json.loads(raw)
        except Exception:
            continue
    return None


def marker_line(text: str) -> Optional[str]:
    m = re.search(r"((?:RAG_RESULT|ORDER_RESULT)::\{.*\})", text)
    return m.group(1) if m else None


def compose_response(rag: Optional[dict], order: Optional[dict]) -> dict:
    parts, sources, grounded, conf = [], [], False, 0.0
    if rag and rag.get("grounded"):
        parts.append(rag["answer"]); sources += rag["sources"]; grounded = True
        conf = float(min(1.0, max(0.0, rag.get("top_sim", 0.0))))
    order_ok = bool(order and order.get("found"))
    if order_ok:
        parts.append(order_sentences(order)); sources.append(f"order_db:{order['record_id']}")
        grounded = True; conf = max(conf, 0.9)
    elif order and order.get("record_id"):
        parts.append(f"I could not find order {order['record_id']}.")
    if not parts:
        return {"answer": FALLBACK, "grounded": False, "sources": [], "order": None,
                "escalate": False, "confidence": 0.0, "needs_clarification": False}
    return {"answer": " ".join(parts), "grounded": grounded, "sources": sources,
            "order": order if order_ok else None,
            "escalate": bool(order_ok and order.get("escalate")),
            "confidence": round(conf, 4), "needs_clarification": False}
