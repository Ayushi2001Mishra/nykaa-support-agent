"""Deterministic MOCK_LLM for CrewAI (extends crewai.llms.base_llm.BaseLLM, the documented
extension point). Speaks CrewAI's text ReAct protocol:
    Thought / Action / Action Input     -> CrewAI runs the tool and appends 'Observation: ...'
    Thought / Final Answer              -> task finished
Parsing logic lives in mock_parsing.py (see its docstring for the two silent pitfalls)."""
import json
import os
import config
from crewai.llms.base_llm import BaseLLM
import mock_parsing as mp
from textutil import FALLBACK


def _as_messages(messages):
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    out = []
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, list):
            c = " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in c)
        out.append({"role": m.get("role", "user"), "content": c or ""})
    return out


class MockLLM(BaseLLM):
    def __init__(self, model: str = "mock-llm"):
        super().__init__(model=model, temperature=0.0)
        self.call_count = 0

    # --- BaseLLM interface -------------------------------------------------
    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, **kwargs):
        self.call_count += 1
        return self._respond(_as_messages(messages))

    async def acall(self, messages, **kwargs):
        return self.call(messages, **kwargs)

    def supports_function_calling(self) -> bool:
        return False                      # force CrewAI's text ReAct path

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        return 8192

    # --- behaviour -----------------------------------------------------------
    def _respond(self, msgs):
        system = "\n".join(m["content"] for m in msgs if m["role"] == "system")
        rest = [m for m in msgs if m["role"] != "system"]
        task_text = rest[0]["content"] if rest else ""
        tail = "\n".join(m["content"] for m in rest[1:])       # only what happened AFTER the task prompt
        tools = mp.declared_tools(system) or mp.declared_tools(task_text)

        if tools:                                               # Retrieval / Lookup agents
            obs = mp.last_observation(tail)
            if obs is not None:                                 # a tool already ran -> finish
                return f"Thought: I now know the final answer\nFinal Answer: {mp.marker_line(obs) or obs}"
            question, order_id = mp.extract_question(task_text), mp.extract_order_id(task_text)
            for name, args in tools:                            # dispatch by ARGUMENT SCHEMA, not name
                if "record_id" in args:
                    if order_id:
                        return self._action(name, {"record_id": order_id})
                    return ("Thought: no order id was supplied, so no lookup is possible\n"
                            'Final Answer: ORDER_RESULT::{"found": false, "reason": "no order id supplied"}')
                if "query" in args and question:
                    return self._action(name, {"query": question})
            return f"Thought: nothing to do\nFinal Answer: {FALLBACK}"

        text = task_text + "\n" + tail                          # Composer: combine upstream outputs
        resp = mp.compose_response(mp.extract_marker(text, "RAG_RESULT"),
                                   mp.extract_marker(text, "ORDER_RESULT"))
        return "Thought: I now can give a great answer\nFinal Answer: " + json.dumps(resp)

    @staticmethod
    def _action(tool, args):
        return f"Thought: I should use a tool\nAction: {tool}\nAction Input: {json.dumps(args)}"


def get_llm():
    if config.MOCK_LLM:
        return MockLLM()
    from crewai import LLM                  # optional real model behind MOCK_LLM=0
    return LLM(model=os.getenv("REAL_LLM_MODEL", "gpt-4o-mini"))
