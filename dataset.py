"""Task 1 - seeded, deterministic order dataset (fabricated data only)."""
import random
from collections import Counter

CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]
CATEGORY_WEIGHTS = [0.22, 0.14, 0.18, 0.20, 0.26]
STATUS_WEIGHTS = [0.14, 0.18, 0.42, 0.14, 0.12]
PRICE_RANGE_INR = {"Beauty": (199, 3999), "Apparel": (399, 6999), "Footwear": (599, 8999),
                   "Home": (499, 9999), "Electronics": (999, 24999)}
PRICE_REASONING = ("Ranges follow typical Nykaa basket sizes: beauty items are small-ticket "
                   "(Rs 199-3,999) while electronics reach Rs 24,999.")
DELAY_PROB = 0.25                       # applied to non-Placed orders only
MIN_DAYS = {"Placed": 0, "Shipped": 1, "Delivered": 4, "Returned": 8, "Refunded": 10}
N_ORDERS = 60
SEED_START = 1


def generate(seed, n=N_ORDERS):
    rng = random.Random(seed)
    orders = []
    for i in range(n):
        cat = rng.choices(CATEGORIES, CATEGORY_WEIGHTS)[0]
        status = rng.choices(STATUSES, STATUS_WEIGHTS)[0]
        lo, hi = PRICE_RANGE_INR[cat]
        delayed_draw = rng.random() < DELAY_PROB
        orders.append({
            "record_id": f"NYK-{1001 + i}",
            "category": cat,
            "status": status,
            "order_value_inr": rng.randint(lo, hi),
            "days_since_created": rng.randint(MIN_DAYS[status], 30),
            "delayed_shipment": bool(delayed_draw and status != "Placed"),
        })
    return orders


def summarize(orders):
    n = len(orders)
    return {"n": n,
            "per_category": dict(Counter(o["category"] for o in orders)),
            "per_status": dict(Counter(o["status"] for o in orders)),
            "delayed_pct": round(100 * sum(o["delayed_shipment"] for o in orders) / n, 1)}


def validate(orders):
    s, problems = summarize(orders), []
    if s["n"] < 40: problems.append("fewer than 40 orders")
    for c in CATEGORIES:
        if s["per_category"].get(c, 0) < 3: problems.append(f"category {c} < 3")
    for st in STATUSES:
        if s["per_status"].get(st, 0) < 1: problems.append(f"status {st} missing")
    if not 10 <= s["delayed_pct"] <= 30: problems.append(f"delayed {s['delayed_pct']}% outside 10-30")
    return problems


def find_seed(start=SEED_START, limit=1000):
    """Deterministically scan seeds upward; never edits records by hand."""
    for seed in range(start, start + limit):
        if not validate(generate(seed)):
            return seed
    raise RuntimeError("no valid seed found - adjust weights")


SEED = find_seed()
ORDERS = generate(SEED)

if __name__ == "__main__":
    print(f"SEED={SEED} (first valid seed scanning up from {SEED_START})")
    print("category weights:", dict(zip(CATEGORIES, CATEGORY_WEIGHTS)))
    print("status weights:  ", dict(zip(STATUSES, STATUS_WEIGHTS)))
    print("price ranges INR:", PRICE_RANGE_INR)
    print("reasoning:", PRICE_REASONING)
    s = summarize(ORDERS)
    print("count per category:", s["per_category"])
    print("count per status:  ", s["per_status"])
    print(f"delayed_shipment=True: {s['delayed_pct']}%   problems: {validate(ORDERS) or 'none'}")
    print("sample:", ORDERS[0])
