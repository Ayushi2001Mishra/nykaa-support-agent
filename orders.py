"""Task 6 - check_order_status with a designed escalation score (pure python)."""
from dataset import ORDERS

ORDER_INDEX = {o["record_id"]: o for o in ORDERS}
W_DELAY, W_AGE = 0.55, 0.45          # weights sum to 1  -> score stays in [0, 1]
OPEN_STATUSES = {"Placed", "Shipped"}
CLOSED_DAMPING = 0.3                  # a Delivered/Returned/Refunded order needs little escalation


def escalation_score(o):
    """score = (0.55*delayed + 0.45*days_since_created/30) * (1.0 if open else 0.3)"""
    age = o["days_since_created"] / 30.0
    factor = 1.0 if o["status"] in OPEN_STATUSES else CLOSED_DAMPING
    return round((W_DELAY * float(o["delayed_shipment"]) + W_AGE * age) * factor, 4)


def percentile(values, p):
    s = sorted(values)
    idx = (len(s) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


SCORES = [escalation_score(o) for o in ORDERS]
# Recommend escalation at the 80th percentile of OUR dataset's scores (top ~20% of orders).
ESCALATION_THRESHOLD = round(percentile(SCORES, 80), 4)


def check_order_status(record_id: str) -> dict:
    o = ORDER_INDEX.get(str(record_id).strip().upper())
    if o is None:
        return {"found": False, "record_id": record_id, "error": "order not found"}
    score = escalation_score(o)
    return {"found": True, "record_id": o["record_id"], "status": o["status"],
            "order_value_inr": o["order_value_inr"], "escalation_score": score,
            "escalation_threshold": ESCALATION_THRESHOLD,
            "escalate": score >= ESCALATION_THRESHOLD}


if __name__ == "__main__":
    print("threshold (80th pct of scores):", ESCALATION_THRESHOLD)
    print("distribution min/median/max:", min(SCORES), percentile(SCORES, 50), max(SCORES))
    print("share >= threshold:", round(100 * sum(s >= ESCALATION_THRESHOLD for s in SCORES) / len(SCORES), 1), "%")
    for o in ORDERS[:4]:
        print(check_order_status(o["record_id"]))
    print(check_order_status("NYK-9999"))
