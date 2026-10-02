# src/supervisor_agent.py
import os
from typing import Annotated, TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed", timeout=60.0)


class SupervisorState(TypedDict):
    messages: Annotated[list, add_messages]
    next_agent: str
    task: str
    use_web: bool
    retriever_output: str
    web_output: str
    synthesis_output: str
    approved: bool

def _msg_content(m):
    return m.content if hasattr(m, "content") else m["content"]


def retriever_node(state: SupervisorState) -> dict:
    """Narrow job: search the corpus, return raw findings. Does not answer
    the user directly — that's the synthesis agent's job."""
    from retrieve import load_index_data, search
    index, chunks, sources, model = load_index_data()
    results = search(state["task"], index, chunks, sources, model, 3)
    findings = "\n\n".join(f"[{r['source']}] {r['chunk'][:300]}" for r in results)
    print(f"[retriever_agent] found {len(results)} chunks")
    return {"retriever_output": findings, "next_agent": "synthesis"}


def synthesis_node(state: SupervisorState) -> dict:
    findings = state.get("retriever_output") or state.get("web_output") or "No information found."
    prompt = f"""Question: {state['task']}

Retrieved information:
{findings}

Write a concise, grounded answer using only the information above."""
    response = client.chat.completions.create(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=300,
    )
    answer = response.choices[0].message.content
    print(f"[synthesis_agent] produced answer")
    return {"synthesis_output": answer, "next_agent": "done"}

def supervisor_node(state: SupervisorState) -> dict:
    if not state.get("retriever_output") and not state.get("web_output"):
        if state.get("use_web"):
            print("[supervisor] routing to web_research")
            return {"next_agent": "web_research"}
        print("[supervisor] routing to retriever")
        return {"next_agent": "retriever"}
    if not state.get("synthesis_output"):
        print("[supervisor] findings available, routing to synthesis")
        return {"next_agent": "synthesis"}
    print("[supervisor] task complete")
    return {"next_agent": "done"}

def route_from_supervisor(state: SupervisorState) -> str:
    return state["next_agent"]

from langgraph.types import interrupt, Command

def approval_node(state: SupervisorState) -> dict:
    """Pauses the graph, surfaces the draft answer to a human, and waits
    for an explicit approve/reject before the system treats it as final."""
    decision = interrupt({
        "question": "Approve this answer?",
        "draft_answer": state["synthesis_output"],
    })
    print(f"[approval_node] human decision: {decision}")
    return {"approved": decision.strip().lower() == "approve"}

def web_research_node(state: SupervisorState) -> dict:
    """Stub for today — simulates a web search result. The point here is
    testing three-way routing, not building a real search integration."""
    print(f"[web_research_agent] (stub) searching web for: {state['task']}")
    stub_result = f"[Simulated web result for: {state['task']}] No real web search wired up yet."
    return {"web_output": stub_result, "next_agent": "synthesis"}

def build_supervisor_graph():
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("web_research", web_research_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("approval", approval_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, {
        "retriever": "retriever", "web_research": "web_research",
        "synthesis": "synthesis", "done": "approval",
    })
    graph.add_edge("retriever", "supervisor")
    graph.add_edge("web_research", "supervisor")
    graph.add_edge("synthesis", "supervisor")
    graph.add_edge("approval", END)

    return graph.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    app = build_supervisor_graph()
    config = {"configurable": {"thread_id": "supervisor-hitl-1"}}

    result = app.invoke(
        {"messages": [], "next_agent": "", "task": "What does the Navigator LLM do in CoLLM-NAS?",
         "use_web": False, "retriever_output": "", "web_output": "",
         "synthesis_output": "", "approved": False},
        config=config,
    )

    if "__interrupt__" in result:
        interrupt_data = result["__interrupt__"][0].value
        print(f"\n--- PAUSED FOR HUMAN APPROVAL ---")
        print(f"Draft answer: {interrupt_data['draft_answer']}")
        human_input = input("Approve? (approve/reject): ").strip().lower()

        final_result = app.invoke(Command(resume=human_input), config=config)

        if final_result.get("approved"):
            print(f"\nApproved: True")
            print(f"Final answer: {final_result['synthesis_output']}")
        else:
            print(f"\nApproved: False")
            print("Final answer: [BLOCKED — answer was not approved by human reviewer, not returned to user]")
    else:
        print(f"\nFINAL (no interrupt triggered): {result}")