"""Task 10 - input-side (PII masking, prompt-injection) and output-side (groundedness) guardrails."""
import re
from textutil import unsupported_sentences

# Indian mobile numbers: optional +91 / 91 prefix, then 10 digits starting 6-9 (optional space/hyphen mid-number).
PHONE_RE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
# Card last-4: only masked when a card cue precedes the 4 digits (so order ids / years are safe).
CARD_A = re.compile(r"(?i)\b(card(?:\s+number)?(?:\s+ending)?(?:\s+in)?(?:\s+is)?|ending(?:\s+in)?|"
                    r"last\s*4(?:\s*digits?)?(?:\s+(?:are|is))?)[\s:#-]*(\d{4})\b")
CARD_B = re.compile(r"(?i)([x*]{4}[\s-]?)(\d{4})\b")

INJECTION_PATTERNS = [re.compile(p, re.I) for p in [
    r"ignore\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|earlier)\s+(?:instructions|rules|prompts?)",
    r"disregard\s+(?:all\s+|any\s+|the\s+)?(?:previous|prior|above|your)\s+(?:instructions|rules)",
    r"(?:reveal|show|print|repeat)\s+(?:me\s+)?(?:your|the)\s+(?:system|hidden|initial)\s+(?:prompt|instructions)",
    r"you\s+are\s+now\s+(?:in\s+)?(?:dan|developer\s+mode|an?\s+unrestricted)",
    r"developer\s+mode", r"jailbreak",
    r"pretend\s+(?:you\s+have|there\s+are)\s+no\s+(?:rules|restrictions)",
    r"override\s+(?:your|the)\s+(?:safety|guardrails|instructions)",
]]


def mask_pii(text):
    """Returns (masked_text, fired) where fired lists which PII types were masked."""
    fired = []
    out, n = PHONE_RE.subn("[PHONE]", text)
    if n: fired.append("PHONE")
    out, n1 = CARD_A.subn(lambda m: m.group(0)[: m.start(2) - m.start(0)] + "[CARD_LAST4]", out)
    out, n2 = CARD_B.subn(lambda m: m.group(1) + "[CARD_LAST4]", out)
    if n1 or n2: fired.append("CARD_LAST4")
    return out, fired


def detect_injection(text):
    return [p.pattern for p in INJECTION_PATTERNS if p.search(text)]


def groundedness_guard(response: dict, context_text: str):
    """Output-side check. Refuse unless the answer is supported by retrieved context
    (or by the order record, which is included in context_text). Returns (ok, reason)."""
    if response.get("needs_clarification"):
        return True, "clarification request"
    if not response.get("grounded"):
        return False, "retrieval similarity below calibrated threshold"
    bad = unsupported_sentences(response["answer"], context_text)
    if bad:
        return False, f"unsupported sentence(s): {bad}"
    return True, "supported by context"
