"""Part 1 transcript: index both strategies, calibrate threshold, grounded answers, precision/recall."""
import config  # noqa: F401
import rag

GOLD = {  # query -> relevant parent documents (my own labelling)
    "What is the return window for beauty products?": {"return_window"},
    "How will I get my refund for a cash on delivery order?": {"cod_refund"},
    "My shoes are the wrong size, can I swap them?": {"size_exchange"},
    "Can I cancel my order after it has shipped?": {"order_cancellation"},
    "Will you pick up my return from home?": {"reverse_pickup", "return_window"},
    "How many points do I need to use my loyalty rewards?": {"loyalty_points"},
}
OOS_QUERY = "Who won the cricket world cup?"

print("== Task 3: index both strategies ==")
print("chunks per collection:", rag.build_index())
for s in rag.STRATEGIES:
    print(f"[{s}] sample retrieval:", [(h['doc_id'], h['sim']) for h in rag.retrieve("refund for cash on delivery", s, 3)])

print("\n== Task 4: threshold calibration (top-1 cosine similarity) ==")
cal = rag.calibrate()
for s, c in cal.items():
    print(f"[{s}] in-scope:", c["in_scope"]); print(f"[{s}] out-of-scope:", c["out_of_scope"])
    print(f"[{s}] in-scope min={c['in_scope_min']}  out-of-scope max={c['out_of_scope_max']}  -> threshold={c['threshold']}")

print(f"\n== Task 4: grounded generation (strategy={config.DEPLOYED_STRATEGY}) ==")
for q in list(GOLD) + [OOS_QUERY]:
    r = rag.grounded_answer(q)
    print(f"Q: {q}\n   grounded={r['grounded']} top_sim={r['top_sim']} thr={r['threshold']} sources={r['sources']}\n   A: {r['answer']}")

print("\n== Task 5: document-level precision / recall (k=%d chunks, dedup to parent docs) ==" % config.TOP_K)
tot = {s: [0.0, 0.0] for s in rag.STRATEGIES}
for q, gold in GOLD.items():
    print(f"\nQ: {q}   gold={sorted(gold)}")
    for s in rag.STRATEGIES:
        got = []
        for h in rag.retrieve(q, s, config.TOP_K):
            if h["doc_id"] not in got:
                got.append(h["doc_id"])
        tp = len(set(got) & gold)
        p, r = tp / len(got), tp / len(gold)
        tot[s][0] += p; tot[s][1] += r
        print(f"  [{s:8}] retrieved docs={got}  precision={tp}/{len(got)}={p:.2f}  recall={tp}/{len(gold)}={r:.2f}")
n = len(GOLD)
print("\nMEAN over", n, "queries:", {s: {"precision": round(v[0] / n, 3), "recall": round(v[1] / n, 3)} for s, v in tot.items()})
print("-> Write your 2-3 sentence recommendation in README.md citing these numbers, then set CHUNKING=<winner>.")
