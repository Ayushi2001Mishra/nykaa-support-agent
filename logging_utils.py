"""Task 12 - JSON-Lines request log. Masking is applied INSIDE the logger so a raw
phone / card last-4 can never reach disk, regardless of the caller."""
import json
import os
import threading
import time
import config
from guardrails import mask_pii

_lock = threading.Lock()


def log_request(trace_id, endpoint, raw_text, status, started_perf, session_id=None, **extra):
    masked, fired = mask_pii(raw_text)
    entry = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "Z",
             "trace_id": trace_id, "endpoint": endpoint, "session_id": session_id,
             "request_text": masked, "pii_masked": fired, "status": status,
             "latency_ms": round((time.perf_counter() - started_perf) * 1000, 2)}
    entry.update(extra)
    os.makedirs(os.path.dirname(config.LOG_PATH) or ".", exist_ok=True)
    with _lock, open(config.LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry
