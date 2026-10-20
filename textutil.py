"""Pure-python text helpers shared by RAG, guardrails, review and evaluation."""
import re

FALLBACK = ("I don't know based on Nykaa's support knowledge base. "
            "Let me connect you with a human agent.")

STOP = set("""the a an and or of to in on for with is are was were be been it its this that these
those at by from as if then than but not no can could should would will you your my me we our
what when where which who how do does did have has had i about any into over after before""".split())


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _stem(w):
    return w[:-1] if len(w) > 3 and w.endswith("s") else w


def content_words(text):
    return [_stem(w) for w in re.findall(r"[a-z0-9₹]+", text.lower())
            if len(w) > 2 and w not in STOP]


def support_ratio(text, context):
    """Fraction of `text`'s content words that appear in `context` (lexical support)."""
    ctx = set(content_words(context))
    words = content_words(text)
    return 1.0 if not words else sum(w in ctx for w in words) / len(words)


def unsupported_sentences(text, context, min_ratio=0.6):
    return [s for s in split_sentences(text) if support_ratio(s, context) < min_ratio]


def order_sentences(order):
    """The one place order facts are phrased, so composer and reviewer agree."""
    s = (f"Order {order['record_id']} is currently {order['status']} "
         f"with an order value of ₹{order['order_value_inr']}.")
    if order.get("escalate"):
        s += " It has been flagged for escalation to Level 2 support."
    return s
