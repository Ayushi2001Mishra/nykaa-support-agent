"""One request end to end: guardrails -> budget -> crew -> output guard -> Autogen review."""
import re
import time
import uuid
import config
from guardrails import detect_injection, groundedness_guard, mask_pii
from governance import (cached_grounded_answer, enforce_budget, request_cost_usd, BudgetExceeded)  # noqa: F401
from schemas import SupportResponse
from textutil import FALLBACK, order_sentences

ORDER_ID_RE = re.compile(r"\bNYK-\d{4}\b", re.I)
NEEDS_ORDER_RE = re.compile(r"\b(my order|order status|where is my|track my|has my order|should it be escalated)\b", re.I)
REFUSAL = SupportResponse(answer=FALLBACK, grounded=False, confidence=0.0)


def _ms(a, b):
    return round((b - a) * 1000, 2)


def ask(question, order_id=None, trace_id=None, review=True, pii_fired=None):
    t0 = time.perf_counter()
    trace_id = trace_id or uuid.uuid4().hex
    masked, f2 = mask_pii(question)                       # input guardrail 1: PII masking
    fired = sorted(set(pii_fired or []) | set(f2))
    est = enforce_budget(masked)                          # runtime layer: raises BudgetExceeded
    guard = {"pii_masked": fired, "injection": [], "groundedness_refusal": False}
    out = {"trace_id": trace_id, "masked_question": masked, "guardrails": guard, "review": None,
           "context_text": "", "tokens_est": est, "cost_usd_est": request_cost_usd(est)}

    inj = detect_injection(masked)                        # input guardrail 2: prompt injection
    if inj:
        guard["injection"] = inj
        r = SupportResponse(answer="I can't help with that request.", grounded=False)
        out.update(response=r.model_dump(), timings={"total_ms": _ms(t0, time.perf_counter())})
        return out

    oid = order_id or (ORDER_ID_RE.search(masked).group(0).upper() if ORDER_ID_RE.search(masked) else None)
    if not oid and NEEDS_ORDER_RE.search(masked):         # no order known (e.g. fresh conversation)
        r = SupportResponse(answer="Please share your order id (like NYK-1005) so I can look it up.",
                            grounded=False, needs_clarification=True)
        out.update(response=r.model_dump(), timings={"total_ms": _ms(t0, time.perf_counter())})
        return out

    from crew import run_crew
    t1 = time.perf_counter()
    resp = run_crew(masked, oid)
    t2 = time.perf_counter()

    ctx = cached_grounded_answer(masked)["context"]       # cache hit: the RAG tool just computed it
    out["context_text"] = "\n".join(ctx) + (("\n" + order_sentences(resp.order)) if resp.order else "")
    ok, reason = groundedness_guard(resp.model_dump(), out["context_text"])   # output guardrail
    if not ok:
        guard["groundedness_refusal"], guard["groundedness_reason"] = True, reason
        resp = REFUSAL

    t3 = time.perf_counter()
    if review and resp.grounded and not resp.needs_clarification:
        from review import review_draft_sync
        verdict = review_draft_sync(resp.answer, out["context_text"])
        out["review"] = verdict.model_dump()
        if not verdict.approved:
            resp = resp.model_copy(update={"answer": verdict.final_answer})
    t4 = time.perf_counter()
    out.update(response=resp.model_dump(),
               timings={"guardrail_ms": _ms(t0, t1), "crew_ms": _ms(t1, t2), "review_ms": _ms(t3, t4),
                        "total_ms": _ms(t0, t4)})
    return out
