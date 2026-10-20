"""Task 14 - Autogen review stage: 2-agent RoundRobinGroupChat (max_turns=2).
Policy_Compliance_Reviewer checks the draft against the retrieved context; Final_Editor
returns a structured Verdict (approved / final_answer / reason)."""
import asyncio
import json
import re
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import StructuredMessage
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_core.models import ChatCompletionClient, CreateResult, RequestUsage, SystemMessage
from schemas import Verdict
from textutil import FALLBACK, split_sentences, unsupported_sentences


def _text(m):
    c = getattr(m, "content", "")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(x if isinstance(x, str) else str(getattr(x, "content", x)) for x in c)
    return str(c)


class MockReviewClient(ChatCompletionClient):
    """Keyless model client for MOCK_LLM. Role is read from the agent's system message."""
    _INFO = {"vision": False, "function_calling": False, "json_output": True,
             "family": "unknown", "structured_output": True}

    def __init__(self):
        self._total = RequestUsage(prompt_tokens=0, completion_tokens=0)
        self.calls = 0

    async def create(self, messages, **kwargs) -> CreateResult:
        self.calls += 1
        system = " ".join(_text(m) for m in messages if isinstance(m, SystemMessage))
        convo = "\n".join(_text(m) for m in messages if not isinstance(m, SystemMessage))
        content = self._editor(convo) if "ROLE=EDITOR" in system else self._reviewer(convo)
        usage = RequestUsage(prompt_tokens=len(convo) // 4, completion_tokens=len(content) // 4)
        self._total = RequestUsage(self._total.prompt_tokens + usage.prompt_tokens,
                                   self._total.completion_tokens + usage.completion_tokens)
        return CreateResult(finish_reason="stop", content=content, usage=usage, cached=False)

    @staticmethod
    def _parse(convo):
        m = re.search(r"DRAFT_ANSWER:\n(.*?)\nRETRIEVED_CONTEXT:\n(.*)", convo, re.S)
        return (m.group(1).strip(), m.group(2).strip()) if m else ("", "")

    def _reviewer(self, convo):
        draft, ctx = self._parse(convo)
        bad = unsupported_sentences(draft, ctx)
        verdict = "OK: every claim is supported by the retrieved context." if not bad \
            else "ISSUES: some claims are not supported by the retrieved context."
        return f"REVIEW: {verdict}\nFLAGGED::{json.dumps(bad)}"

    def _editor(self, convo):
        draft, _ = self._parse(convo)
        flagged = json.loads((re.findall(r"FLAGGED::(\[.*\])", convo) or ["[]"])[-1])
        if not flagged:
            v = Verdict(approved=True, final_answer=draft, reason="All claims are supported by the retrieved context.")
        else:
            kept = [s for s in split_sentences(draft) if s not in flagged]
            v = Verdict(approved=False, final_answer=" ".join(kept) or FALLBACK,
                        reason=f"Removed {len(flagged)} unsupported claim(s): {flagged}")
        return v.model_dump_json()

    async def create_stream(self, messages, **kwargs):
        result = await self.create(messages, **kwargs)
        yield result.content
        yield result

    async def close(self): pass
    def actual_usage(self): return self._total
    def total_usage(self): return self._total
    def count_tokens(self, messages, **kwargs): return sum(len(_text(m)) // 4 for m in messages)
    def remaining_tokens(self, messages, **kwargs): return 8000 - self.count_tokens(messages)
    @property
    def capabilities(self): return {k: self._INFO[k] for k in ("vision", "function_calling", "json_output")}
    @property
    def model_info(self): return self._INFO


def build_team():
    client = MockReviewClient()      # swap for OpenAIChatCompletionClient(...) to use a real model
    reviewer = AssistantAgent(
        "Policy_Compliance_Reviewer", model_client=client,
        system_message="ROLE=REVIEWER. Check each claim in DRAFT_ANSWER against RETRIEVED_CONTEXT and list unsupported ones.")
    editor = AssistantAgent(
        "Final_Editor", model_client=client, output_content_type=Verdict,
        system_message="ROLE=EDITOR. Approve the draft unchanged, or revise it to remove unsupported claims.")
    return RoundRobinGroupChat([reviewer, editor], max_turns=2,
                               custom_message_types=[StructuredMessage[Verdict]])


async def review_draft(draft: str, context: str) -> Verdict:
    result = await build_team().run(task=f"DRAFT_ANSWER:\n{draft}\nRETRIEVED_CONTEXT:\n{context}")
    for m in reversed(result.messages):
        if isinstance(m, StructuredMessage):
            return m.content
    raise RuntimeError("Final_Editor produced no structured verdict")


def review_draft_sync(draft: str, context: str) -> Verdict:
    return asyncio.run(review_draft(draft, context))    # call from sync code / worker threads
