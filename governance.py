"""Part 4 - governance: runtime budget cap (Task 15), least autonomy (Task 15),
risk classification (Task 15) and response caching (Task 16)."""
import math
import re
import threading
import config

# ---------- Runtime layer: per-request token / cost budget ----------
class BudgetExceeded(Exception):
    pass

OUTPUT_RESERVE = 300                       # tokens reserved for the answer
CONTEXT_RESERVE = config.TOP_K * 90        # ~90 tokens per retrieved chunk
COST_PER_1K_TOKENS_USD = 0.002             # simulated price (MOCK_LLM is free)


def estimate_tokens(text):
    return math.ceil(len(text) / 4)


def estimate_request_tokens(question):
    return estimate_tokens(question) + CONTEXT_RESERVE + OUTPUT_RESERVE


def request_cost_usd(tokens):
    return round(tokens / 1000 * COST_PER_1K_TOKENS_USD, 6)


def enforce_budget(question):
    """Reject BEFORE any retrieval/LLM work if the estimated request exceeds the cap."""
    est = estimate_request_tokens(question)
    if est > config.MAX_REQUEST_TOKENS:
        raise BudgetExceeded(f"estimated {est} tokens (~${request_cost_usd(est)}) exceeds the "
                             f"per-request cap of {config.MAX_REQUEST_TOKENS}")
    return est

# ---------- Application layer: least autonomy ----------
AUTHORIZED_TOOL_HOLDERS = {"check_order_status": {"Lookup Agent"}}


def assert_tool_allowed(role, tool_name):
    allowed = AUTHORIZED_TOOL_HOLDERS.get(tool_name)
    if allowed is not None and role not in allowed:
        raise PermissionError(f"least-autonomy: '{role}' may not hold tool '{tool_name}' "
                              f"(allowed: {sorted(allowed)})")


def audit_agents(agents):
    """Second line of defence: re-verify every agent's tool list after the crew is built."""
    for a in agents:
        for t in (a.tools or []):
            assert_tool_allowed(a.role, t.name)
    return True

RISK_LEVEL = "Medium"
RISK_JUSTIFICATION = (
    "Medium risk. The system is a customer-support assistant (the scheme's Medium class: customer "
    "support tickets). It only reads policy text and order status; it never issues refunds, "
    "cancels orders or changes records, and it handles only fixed-format PII (phone, card last-4) "
    "which is masked before the model or the logs see it. It would become High if it were given "
    "authority to move money or decide claims, because that would be financial data and decisions.")

# ---------- Response cache ----------
def normalize_query(q):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s₹]", "", q.lower())).strip()


class ResponseCache:
    def __init__(self):
        self._d, self.hits, self.misses = {}, 0, 0
        self._lock = threading.Lock()

    def get_or_compute(self, key, fn):
        with self._lock:
            if key in self._d:
                self.hits += 1
                return self._d[key]
        value = fn()                       # compute outside the lock (slow path)
        with self._lock:
            self.misses += 1
            self._d[key] = value
        return value

    def clear(self):
        with self._lock:
            self._d.clear()

CACHE = ResponseCache()


def cached_grounded_answer(query, strategy=None):
    import rag                              # lazy: keeps this module importable without chroma
    strategy = strategy or config.DEPLOYED_STRATEGY
    key = f"{strategy}::{normalize_query(query)}"
    return CACHE.get_or_compute(key, lambda: rag.grounded_answer(query, strategy))
