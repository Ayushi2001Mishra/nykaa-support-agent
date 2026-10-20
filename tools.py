"""CrewAI tools. NOTE: the RAG tool is deliberately named 'rag_lookup' - the MockLLM must
(and does) dispatch by argument schema, so this name cannot be misclassified."""
import json
from typing import Type
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from governance import cached_grounded_answer
from orders import check_order_status

TOOL_CALLS = []                              # evidence log: every real tool invocation


class RagInput(BaseModel):
    query: str = Field(..., description="The customer's policy question")


class OrderInput(BaseModel):
    record_id: str = Field(..., description="Order id such as NYK-1005")


class RagLookupTool(BaseTool):
    name: str = "rag_lookup"
    description: str = "Search Nykaa's policy knowledge base; returns a grounded answer with sources."
    args_schema: Type[BaseModel] = RagInput

    def _run(self, query: str) -> str:
        TOOL_CALLS.append({"tool": self.name, "args": {"query": query}})
        res = cached_grounded_answer(query)
        keep = {k: res[k] for k in ("grounded", "answer", "sources", "top_sim")}
        return "RAG_RESULT::" + json.dumps(keep)


class CheckOrderStatusTool(BaseTool):
    name: str = "check_order_status"
    description: str = "Look up an order by record_id; returns status, value and escalation score."
    args_schema: Type[BaseModel] = OrderInput

    def _run(self, record_id: str) -> str:
        TOOL_CALLS.append({"tool": self.name, "args": {"record_id": record_id}})
        return "ORDER_RESULT::" + json.dumps(check_order_status(record_id))
