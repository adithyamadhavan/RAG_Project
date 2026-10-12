# src/supervisor_llm.py
# Same graph as supervisor_agent.py, but the routing decision is made by an
# LLM call instead of if/else rules. Retriever, synthesis and approval nodes
# are imported unchanged, so any difference in cost, latency or reliability
# comes from the routing alone.
#
#   python src/supervisor_llm.py              -> one interactive run
#   BENCH=1 python src/supervisor_llm.py      -> 1 warm-up + 3 timed runs
#   ROUTER_GUARD=0 BENCH=1 python ...         -> let the model's choice stand
#                                                (exposes real misrouting)
import os
import itertools
from collections import Counter

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError
from langgraph.types import Command

from bench_utils import counted_chat, benchmark
from supervisor_agent import (
    SupervisorState, retriever_node, synthesis_node, approval_node,
    client, MODEL_NAME, _get_index, _initial_state,
)

GUARD = os.environ.get("ROUTER_GUARD", "1") != "0"
ROUTE_OPTIONS = ("retriever", "synthesis", "finish")

# Routing reliability, accumulated across all runs in this process.
ROUTE_STATS = Counter()
_run_ids = itertools.count()


def _expected_next(state):
    """What a correct router should pick. Used for scoring (and as the guard)."""
    if not state.get("retriever_output"):
        return "retriever"
    if not state.get("synthesis_output"):
        return "synthesis"
    return "finish"


def llm_supervisor_node(state: SupervisorState) -> dict:
    expected = _expected_next(state)
    prompt = f"""You are a supervisor routing work between agents.

Agents:
- retriever: searches the document corpus. Use it when no information has been retrieved yet.
- synthesis: writes the answer from retrieved information. Use it when information has been retrieved but no answer exists yet.
- finish: use it when an answer already exists.

Current state:
- Information retrieved: {'yes' if state.get('retriever_output') else 'no'}
- Answer written: {'yes' if state.get('synthesis_output') else 'no'}

Reply with exactly one word: retriever, synthesis, or finish."""
    resp = counted_chat(
        client, model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=6,
    )
    text = (resp.choices[0].message.content or "").strip().lower()
    choice = next((o for o in ROUTE_OPTIONS if o in text), None)

    ROUTE_STATS["decisions"] += 1
    if choice == expected:
        ROUTE_STATS["correct"] += 1
    if GUARD and choice != expected:
        ROUTE_STATS["overridden"] += 1
        choice = expected
    elif choice is None:
        choice = "finish"

    print(f"[llm_supervisor] model said {text!r} -> routing to {choice}")
    return {"next_agent": "done" if choice == "finish" else choice}


def route_from_supervisor(state: SupervisorState) -> str:
    return state["next_agent"]


def build_llm_supervisor_graph():
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", llm_supervisor_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("synthesis", synthesis_node)
    graph.add_node("approval", approval_node)

    graph.add_edge(START, "supervisor")
    graph.add_conditional_edges("supervisor", route_from_supervisor, {
        "retriever": "retriever", "synthesis": "synthesis", "done": "approval",
    })
    graph.add_edge("retriever", "supervisor")
    graph.add_edge("synthesis", "supervisor")
    graph.add_edge("approval", END)
    return graph.compile(checkpointer=InMemorySaver())


def print_route_stats():
    d = ROUTE_STATS["decisions"]
    if not d:
        return
    print(f"ROUTER: {ROUTE_STATS['correct']}/{d} decisions correct "
          f"({100 * ROUTE_STATS['correct'] / d:.0f}%), "
          f"{ROUTE_STATS['overridden']} overridden by guard, "
          f"{ROUTE_STATS['failed_runs']} failed runs "
          f"(guard {'on' if GUARD else 'OFF'})")


if __name__ == "__main__":
    app = build_llm_supervisor_graph()

    def run_once():
        cfg = {"configurable": {"thread_id": f"llm-{next(_run_ids)}"},
               "recursion_limit": 12}
        try:
            r = app.invoke(_initial_state(), config=cfg)
            if "__interrupt__" in r:  # auto-approve, no input() in timed path
                r = app.invoke(Command(resume="approve"), config=cfg)
            return r.get("synthesis_output", "")
        except GraphRecursionError:
            ROUTE_STATS["failed_runs"] += 1
            return ""

    if os.environ.get("BENCH"):
        _get_index()  # load before timing
        benchmark(run_once)
        print_route_stats()
    else:
        print(f"\nFINAL ANSWER: {run_once()}")
        print_route_stats()