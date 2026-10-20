# Final Capstone - Nykaa Domain Support Agent (CrewAI)

**Track completed: Nykaa (E-commerce & Retail).** Everything runs under `MOCK_LLM=1` with zero API keys
and zero network access (after a one-time download of the SentenceTransformers model).

## Setup / run
```bash
pip install -r requirements.txt
export MOCK_LLM=1 CREWAI_DISABLE_TELEMETRY=true OTEL_SDK_DISABLED=true   # telemetry disabled (also set in config.py)
./run_all.sh                       # writes transcripts/*.txt
uvicorn app:app --reload           # optional: serve the API
```
**Telemetry:** I set `CREWAI_DISABLE_TELEMETRY=true` and `OTEL_SDK_DISABLED=true` (in `config.py`, before CrewAI imports, and in `run_all.sh`).

## Task 1 - dataset design choices (reproducible)
- `dataset.py`, `random.Random(seed)`; seed = first valid value scanning up from 1 -> **SEED = 1**; N = 60.
- Category weights: Apparel .22, Electronics .14, Home .18, Footwear .20, Beauty .26.
- Status weights: Placed .14, Shipped .18, Delivered .42, Returned .14, Refunded .12.
- Amount ranges (INR): Beauty 199-3,999; Apparel 399-6,999; Footwear 599-8,999; Home 499-9,999; Electronics 999-24,999. *Reasoning:* matches typical basket sizes per category.
- `delayed_shipment`: p = 0.25, never for `Placed` orders. Result: 18.3% (band 10-30%). Counts: see `transcripts/task1_dataset.txt`.

## Task 4 - threshold calibration  (paste from transcripts/part1.txt)
Measured in-scope top-1 similarities: `...`; out-of-scope: `...`; chosen threshold per strategy: `...`.

## Task 5 - chunking recommendation  (paste your numbers)
Mean precision/recall fixed vs sentence: `...`. Recommendation (2-3 sentences citing them): `...`. Then run with `CHUNKING=<winner>`.

## Task 6 - escalation score
`score = (0.55*delayed + 0.45*days_since_created/30) * (1.0 if status in {Placed, Shipped} else 0.3)`, range [0, 1].
Threshold = 80th percentile of my 60 scores (0.2775; top ~21.7% of orders escalate).

## Task 15 - governance notes
- **Least autonomy:** `governance.AUTHORIZED_TOOL_HOLDERS` maps `check_order_status` -> `{"Lookup Agent"}`. `crew.make_agent` calls `assert_tool_allowed` for every tool before an agent is built (wiring it to the Composer raises `PermissionError`, shown in `demo_part4.py`), and `audit_agents` re-checks the finished crew. The Retrieval Agent holds only `rag_lookup`; the Composer holds no tools.
- **Risk level:** Medium (customer-support tickets); see `governance.RISK_JUSTIFICATION`.
- **Budget:** `MAX_REQUEST_TOKENS = 1500` (chars/4 estimate + context + output reserve) rejects oversized requests before any retrieval/LLM work (HTTP 413).

## Notes
- `MOCK_LLM` = `mock_llm.MockLLM(BaseLLM)` (CrewAI) + `review.MockReviewClient` (Autogen) + `evaluate.mock_judge`.
- Name/address are free text and out of masking scope; only fabricated examples are used.
