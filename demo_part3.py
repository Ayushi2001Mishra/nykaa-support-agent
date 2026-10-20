"""Part 3 transcript: FastAPI (HTTP + WebSocket), JSONL logging, evaluation."""
import config  # noqa: F401
import json
import os
from fastapi.testclient import TestClient
import app as app_module
import evaluate

if os.path.exists(config.LOG_PATH):
    os.remove(config.LOG_PATH)
RAW_PHONE = "9876543210"
with TestClient(app_module.app) as c:
    print("== Task 11: HTTP ==")
    print("GET /health ->", c.get("/health").json())
    r = c.post("/ask", json={"question": f"My phone is {RAW_PHONE}. What is the return window for apparel?", "session_id": "s1"})
    print("POST /ask ->", r.status_code, json.dumps(r.json(), ensure_ascii=False)[:400])
    r = c.post("/add-document", json={"doc_id": "gift_wrap", "title": "Gift wrapping policy",
               "text": "Gift wrapping costs 49 rupees per order. It is available for Beauty and Home items only."})
    print("POST /add-document ->", r.status_code, r.json())
    print("POST /ask (new doc) ->", c.post("/ask", json={"question": "How much does gift wrapping cost?"}).json()["response"]["answer"])
    r = c.post("/ask", json={"question": "x" * 15000})
    print("POST /ask (oversized) ->", r.status_code, r.json())

    print("\n== Task 11: WebSocket, client disconnects mid-conversation ==")
    with c.websocket_connect("/ws/chat?session_id=ws1") as ws:
        ws.send_text("What is the return window for beauty products?")
        print("client 1 got:", ws.receive_json()["response"]["answer"])
        ws.send_text("How are COD orders refunded?")          # disconnect without reading the reply
    print("client 1 disconnected; server still alive ->", c.get("/health").json()["status"])
    with c.websocket_connect("/ws/chat?session_id=ws2") as ws:
        ws.send_text("What warranty comes with electronics?")
        print("client 2 got:", ws.receive_json()["response"]["answer"])

print("\n== Task 12: logs ==")
log = open(config.LOG_PATH, encoding="utf-8").read()
for line in log.splitlines()[:3]:
    print(line[:300])
print(f"entries={len(log.splitlines())}  raw phone '{RAW_PHONE}' present on disk: {RAW_PHONE in log}")

print("\n== Task 13: evaluation ==")
evaluate.run()
