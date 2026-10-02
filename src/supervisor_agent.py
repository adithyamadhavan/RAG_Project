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
    retriever_output: str
    synthesis_output: str


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
    """Narrow job: given retrieved findings, write a grounded final answer.
    Does not retrieve anything itself — trusts what retriever_node found."""
    prompt = f"""Question: {state['task']}

Retrieved information:
{state['retriever_output']}

Write a concise, grounded answer using only the information above."""
    response = client.chat.completions.create(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=300,
    )
    answer = response.choices[0].message.content
    print(f"[synthesis_agent] produced answer")
    return {"synthesis_output": answer, "next_agent": "done"}

def supervisor_node(state: SupervisorState) -> dict:
    """Decides which specialist acts next, given current state. This is
    the ONLY node with routing authority — sub-agents never decide for
    themselves what happens next."""
    if state.get("retriever_output") and not state.get("synthesis_output"):
        print("[supervisor] retrieval done, routing to synthesis")
        return {"next_agent": "synthesis"}
    if not state.get("retriever_output"):
        print("[supervisor] routing to retriever")
        return {"next_agent": "retriever"}
    print("[supervisor] task complete")
    return {"next_agent": "done"}


def route_from_supervisor(state: SupervisorState) -> str:
    return state["next_agent"]


def build_supervisor_graph():
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("synthesis", synthesis_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, {
        "retriever": "retriever", "synthesis": "synthesis", "done": END,
    })
    graph.add_edge("retriever", "supervisor")
    graph.add_edge("synthesis", "supervisor")

    return graph.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    app = build_supervisor_graph()
    result = app.invoke(
        {"messages": [], "next_agent": "", "task": "What does the Navigator LLM do in CoLLM-NAS?",
         "retriever_output": "", "synthesis_output": ""},
        config={"configurable": {"thread_id": "supervisor-test-1"}},
    )
    print(f"\nFINAL ANSWER: {result['synthesis_output']}")  