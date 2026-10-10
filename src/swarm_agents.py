# src/swarm_agents.py
import os
import itertools
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from openai import OpenAI

from bench_utils import counted_chat, benchmark
from prompts import build_prompt

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed", timeout=60.0)

# Index is loaded once and reused, so load time is not counted in latency.
_INDEX = None


def _get_index():
    global _INDEX
    if _INDEX is None:
        from retrieve import load_index_data
        _INDEX = load_index_data()
    return _INDEX


_run_ids = itertools.count()


class SwarmState(TypedDict):
    messages: Annotated[list, add_messages]
    task: str
    findings: str
    answer: str
    handoff_to: str


def retriever_agent(state: SwarmState) -> dict:
    """In a swarm, THIS agent decides whether to hand off — no central
    supervisor makes that call."""
    from retrieve import search
    index, chunks, sources, model = _get_index()
    results = search(state["task"], index, chunks, sources, model, 3)
    findings = "\n\n".join(f"[{r['source']}] {r['chunk'][:300]}" for r in results)
    print(f"[retriever_agent] found {len(results)} chunks, handing off to synthesis_agent")
    if os.environ.get("SHOW_CHUNKS"):
        print("CHUNKS:\n" + findings)
    # Self-directed handoff: this agent decides where control goes next,
    # rather than reporting back to a router.
    return {"findings": findings, "handoff_to": "synthesis_agent"}


def synthesis_agent(state: SwarmState) -> dict:
    """Receives direct handoff from retriever_agent, no supervisor involved."""
    prompt = build_prompt(state['task'], state['findings'])
    response = counted_chat(
        client,
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=300,
    )
    answer = response.choices[0].message.content
    print("[synthesis_agent] produced answer, handing off to END")
    return {"answer": answer, "handoff_to": "end"}


def route_by_handoff(state: SwarmState) -> str:
    """The ONLY thing resembling central routing — and notice it's just
    reading each agent's own decision, not making one itself."""
    return state["handoff_to"]


def build_swarm_graph():
    graph = StateGraph(SwarmState)
    graph.add_node("retriever_agent", retriever_agent)
    graph.add_node("synthesis_agent", synthesis_agent)

    graph.add_edge(START, "retriever_agent")
    graph.add_conditional_edges("retriever_agent", route_by_handoff, {
        "synthesis_agent": "synthesis_agent",
    })
    graph.add_conditional_edges("synthesis_agent", route_by_handoff, {
        "end": END,
    })

    return graph.compile(checkpointer=InMemorySaver())


QUERY = "What does the Navigator LLM do in CoLLM-NAS?"


def _initial_state(task=QUERY):
    return {"messages": [], "task": task, "findings": "",
            "answer": "", "handoff_to": ""}


if __name__ == "__main__":
    app = build_swarm_graph()

    def run_once():
        cfg = {"configurable": {"thread_id": f"bench-{next(_run_ids)}"}}
        return app.invoke(_initial_state(), config=cfg)["answer"]

    if os.environ.get("BENCH"):
        _get_index()  # load before timing
        benchmark(run_once)
    else:
        print(f"\nFINAL ANSWER: {run_once()}")