"""Part 4 transcript: Autogen review, least autonomy, risk, budget cap, response cache."""
import config  # noqa: F401
import time
import governance as gov
import pipeline, rag
from crew import build_crew, make_agent
from mock_llm import get_llm
from review import review_draft_sync
from tools import CheckOrderStatusTool

rag.build_index()
print("== Task 14: Autogen review (RoundRobinGroupChat, max_turns=2) ==")
base = pipeline.ask("What is the return window for beauty products?", review=False)
draft, ctx = base["response"]["answer"], base["context_text"]
v = review_draft_sync(draft, ctx)
print("CASE 1 (clean draft):", v.model_dump_json())
bad = draft + " Nykaa also pays a guaranteed cashback voucher of ₹5000 on every single return."
v = review_draft_sync(bad, ctx)
print("CASE 2 (injected ungrounded claim):", v.model_dump_json())

print("\n== Task 15: least autonomy ==")
try:
    make_agent("Response Composer", "g", "b", [CheckOrderStatusTool()], get_llm())
except PermissionError as e:
    print("BLOCKED:", e)
crew = build_crew()
print("audit_agents on the real crew:", gov.audit_agents(crew.agents),
      {a.role: [t.name for t in a.tools] for a in crew.agents})
print("\nRisk level:", gov.RISK_LEVEL, "\n" + gov.RISK_JUSTIFICATION)

print("\n== Task 15: runtime budget cap ==")
print("normal request estimate:", gov.enforce_budget("What is the return window?"), "tokens (cap", config.MAX_REQUEST_TOKENS, ")")
try:
    pipeline.ask("Please explain " + "returns and refunds " * 500)
except gov.BudgetExceeded as e:
    print("REJECTED (no retrieval/LLM work done):", e)

print("\n== Task 16: response cache ==")
gov.CACHE.clear(); gov.CACHE.hits = gov.CACHE.misses = 0; rag.GEN_CALLS["count"] = 0
q = "What is the return window for beauty products?"
t = time.perf_counter(); a1 = gov.cached_grounded_answer(q); d1 = (time.perf_counter() - t) * 1000
print(f"1st call: miss  | generation calls={rag.GEN_CALLS['count']} | {d1:.2f} ms")
t = time.perf_counter(); a2 = gov.cached_grounded_answer("  what is the RETURN window for beauty products?? "); d2 = (time.perf_counter() - t) * 1000
print(f"2nd call: hit   | generation calls={rag.GEN_CALLS['count']} | {d2:.2f} ms | same answer: {a1 is a2}")
print(f"cache stats: hits={gov.CACHE.hits} misses={gov.CACHE.misses}")
