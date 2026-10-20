"""Tasks 7, 9, 15 - the CrewAI crew (Retrieval, Lookup, Composer) with least autonomy."""
import json
import config
from crewai import Agent, Crew, Process, Task
from governance import assert_tool_allowed, audit_agents
from mock_llm import get_llm
from schemas import SupportResponse
from tools import CheckOrderStatusTool, RagLookupTool


def make_agent(role, goal, backstory, tools, llm):
    for t in tools:                                   # Application layer: least autonomy
        assert_tool_allowed(role, t.name)
    return Agent(role=role, goal=goal, backstory=backstory, tools=tools, llm=llm,
                 allow_delegation=False, verbose=False, max_iter=4)


def build_crew(llm=None):
    llm = llm or get_llm()
    retr = make_agent("Retrieval Agent", "Find grounded policy answers",
                      "You search Nykaa's policy knowledge base.", [RagLookupTool()], llm)
    look = make_agent("Lookup Agent", "Look up order records",
                      "You are the only agent allowed to read order records.", [CheckOrderStatusTool()], llm)
    comp = make_agent("Response Composer", "Write one final structured answer",
                      "You merge the other agents' outputs; you have no tools.", [], llm)
    t1 = Task(description="Answer the policy part of the customer question using your tool. "
                          "QUESTION: [[{question}]]",
              expected_output="The raw tool result line.", agent=retr)
    t2 = Task(description="If an order id is given, look the order up with your tool; otherwise do nothing. "
                          "ORDER_ID: [[{order_id}]] QUESTION: [[{question}]]",
              expected_output="The raw tool result line.", agent=look)
    t3 = Task(description="Combine the policy retrieval result and the order lookup result into one "
                          "JSON object matching the SupportResponse schema.",
              expected_output="A JSON SupportResponse.", agent=comp, context=[t1, t2],
              output_pydantic=SupportResponse)
    crew = Crew(agents=[retr, look, comp], tasks=[t1, t2, t3], process=Process.sequential, verbose=False)
    audit_agents(crew.agents)
    return crew


def run_crew(question, order_id=None) -> SupportResponse:
    crew = build_crew()
    out = crew.kickoff(inputs={"question": question, "order_id": order_id or "NONE"})
    data = out.pydantic
    if data is None:                                  # fall back to parsing the raw text
        raw = out.raw
        data = json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    payload = data.model_dump() if hasattr(data, "model_dump") else data
    return SupportResponse.model_validate(payload)    # explicit validation in code (Task 9)
