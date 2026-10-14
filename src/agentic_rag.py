# src/agentic_rag.py
# One LangGraph that merges the pieces built earlier:
#
#   input_guard -> (math?) tool_node ---------------------------+
#               -> retrieve -> grade -+-> generate --------------+-> output_guard -> approval -> END
#                    ^                +-> reformulate (once) -+  |
#                    +----------------+                         |
#                                     +-> abstain --------------+
#
#   input_guard : regex PII redaction on the query (guardrails.detect_pii/redact_pii)
#                 optional LLM content check (CONTENT_GUARD=1)
#   retrieve    : same index + search() as the other modes, top 3
#   grade       : (optional) distance cutoff, then a forced-reasoning relevance
#                 check (evidence first, then VERDICT:). GRADER=crag uses your
#                 corrective_rag.grade_context_sufficiency instead.
#   reformulate : corrective_rag.reformulate_query, at most once
#   abstain     : no generate call; returns a NOT_IN_CORPUS answer (the fix for
#                 the 0/2 out-of-corpus result)
#   tool_node   : arithmetic goes to react_loop.execute_tool_with_retry("calculator")
#   output_guard: regex PII redaction on the answer (+ Presidio if OUTPUT_PRESIDIO=1)
#   approval    : human-in-the-loop interrupt, same as the supervisor
#
#   python src/agentic_rag.py               -> one interactive run
#   BENCH=1 python src/agentic_rag.py       -> 1 warm-up + 3 timed runs
#   SHOW_SCORES=1 ...                       -> print retrieval distances (to tune MAX_DIST)
#   MAX_DIST=1.1 ...                        -> abstain when best L2 distance is above this
import os
import re
import itertools
from typing import Annotated, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from openai import OpenAI

from bench_utils import counted_chat, benchmark
from prompts import build_prompt, NOT_FOUND

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed", timeout=60.0)

# LangSmith tracing (optional). With LANGSMITH_TRACING=true and LANGSMITH_API_KEY set,
# every LangGraph node is traced automatically, and wrapping the client below adds each
# LLM call (prompt, output, latency) as a child span. Off by default: nothing is sent.
if os.environ.get("LANGSMITH_TRACING", "").lower() == "true" or \
   os.environ.get("LANGCHAIN_TRACING_V2", "").lower() == "true":
    try:
        from langsmith.wrappers import wrap_openai
        client = wrap_openai(client)
        print("[langsmith] tracing on, project:",
              os.environ.get("LANGSMITH_PROJECT") or os.environ.get("LANGCHAIN_PROJECT") or "default")
    except ImportError:
        print("[langsmith] tracing requested but 'langsmith' is not installed: pip install langsmith")

# Retrieval-score gate. None = off. Tune with SHOW_SCORES=1 (lower L2 = closer).
_md = os.environ.get("MAX_DIST")
MAX_DIST = float(_md) if _md else 0.8             # calibrated: in-corpus <= 0.52, France 1.08
GRADER = os.environ.get("GRADER", "score")        # "score" | "crag" | "own" (1.5B LLM grader, unreliable)
CONTENT_GUARD = os.environ.get("CONTENT_GUARD") == "1"
OUTPUT_PRESIDIO = os.environ.get("OUTPUT_PRESIDIO") == "1"
SHOW_SCORES = bool(os.environ.get("SHOW_SCORES"))
USE_LTM = os.environ.get("USE_LTM") == "1"        # long-term memory, default off

ABSTAIN_TEXT = (f"{NOT_FOUND}: the indexed papers do not contain information "
                "that answers this question.")
MATH_RE = re.compile(r"^[\s\d+\-*/().%]+$")
MATH_IN_Q = re.compile(r"(\d[\d\s+\-*/().]*[+\-*/][\d\s+\-*/().]*\d)")

_INDEX = None
_run_ids = itertools.count()


def _get_index():
    global _INDEX
    if _INDEX is None:
        from retrieve import load_index_data
        _INDEX = load_index_data()
    return _INDEX


def _keep_recent(old, new):
    """Reducer for conversation history: append, keep the last 6 turns."""
    return ((old or []) + (new or []))[-6:]


class AgenticState(TypedDict):
    task: str          # original user question (after input redaction)
    query: str         # query used for retrieval (may be reformulated)
    attempts: int      # reformulations used
    findings: str
    best_dist: float   # -1.0 when unknown
    verdict: str       # "relevant" | "irrelevant"
    answer: str
    blocked: bool
    route: str         # "tool" | "rag"
    path: str          # trace of nodes visited, for evals / README
    approved: bool
    cache_hit: bool
    cache_sim: float
    # Conversation memory. Persisted by the checkpointer per thread_id, so it is
    # NOT reset in _initial_state: a new turn on the same thread sees it.
    history: Annotated[list, _keep_recent]


def _initial_state(task):
    return {"task": task, "query": task, "attempts": 0, "findings": "",
            "best_dist": -1.0, "verdict": "", "answer": "", "blocked": False,
            "route": "", "path": "", "approved": False,
            "cache_hit": False, "cache_sim": -1.0}


def _p(state, node):
    return (state.get("path") or "") + node + ">"


# ---------------------------------------------------------------- guards
def input_guard(state: AgenticState) -> dict:
    from guardrails import detect_pii, redact_pii
    task = state["task"]
    if detect_pii(task):
        task = redact_pii(task)
        print("[input_guard] PII redacted from query")
    if CONTENT_GUARD:
        from guardrails import check_content_appropriateness
        ok, reason = check_content_appropriateness(task)
        # ASSUMPTION: the bool is True when the text is acceptable.
        # If yours returns True for "flagged", flip the condition below.
        if not ok:
            print(f"[input_guard] blocked: {reason}")
            return {"task": task, "query": task, "blocked": True,
                    "answer": "Request blocked by content guardrail.",
                    "path": _p(state, "input_guard(blocked)")}
    route = "tool" if MATH_IN_Q.search(task) and len(task) < 80 else "rag"
    return {"task": task, "query": task, "route": route,
            "path": _p(state, "input_guard")}


def route_after_input(state: AgenticState) -> str:
    if state["blocked"]:
        return "approval"
    return "tool_node" if state["route"] == "tool" else "contextualize"


# ---------------------------------------------------------------- conversation memory
def contextualize(state: AgenticState) -> dict:
    """Turn a follow-up ("what about its limitations?") into a standalone
    question using the last two turns. No history -> no LLM call."""
    hist = state.get("history") or []
    if not hist:
        return {"path": _p(state, "contextualize(none)")}
    turns = "\n".join(f"Q: {t['q']}\nA: {t['a'][:200]}" for t in hist[-2:])
    prompt = f"""Conversation so far:
{turns}

Follow-up question: {state['task']}

Rewrite the follow-up as a complete standalone question that names the thing it refers to.
If it is already standalone, repeat it unchanged. Reply with the question only."""
    resp = counted_chat(client, model=MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.0, max_tokens=60)
    new_q = (resp.choices[0].message.content or "").strip().strip('"').split("\n")[0]
    if not new_q or len(new_q) > 300:
        new_q = state["task"]
    print(f"[contextualize] {state['task']!r} -> {new_q!r}")
    return {"task": new_q, "query": new_q, "path": _p(state, "contextualize")}


def remember(state: AgenticState) -> dict:
    """Store the finished turn in history (approved, non-abstain answers only)."""
    if (state["approved"] and not state["blocked"]
            and NOT_FOUND not in state["answer"]):
        if USE_LTM and not os.environ.get("EVAL_RUN"):   # evals auto-approve: never memorise them
            from agent_memory import store_turn
            outcome = store_turn(state["task"], state["answer"], model=_get_index()[3])
            print(f"[memory] long-term: {outcome}")
        return {"history": [{"q": state["task"], "a": state["answer"]}],
                "path": _p(state, "remember")}
    return {}


# ---------------------------------------------------------------- semantic cache
def _use_cache():
    return os.environ.get("USE_CACHE") == "1"


def cache_lookup(state: AgenticState) -> dict:
    if not _use_cache():
        return {"path": _p(state, "cache_lookup(off)")}
    from semantic_cache import CACHE
    answer, sim = CACHE.lookup(_get_index()[3], state["task"])
    if answer is not None:
        print(f"[cache] HIT (similarity {sim:.3f})")
        return {"answer": answer, "cache_hit": True, "cache_sim": sim,
                "approved": True, "path": _p(state, "cache_hit")}
    print(f"[cache] miss (best similarity {sim:.3f})")
    return {"cache_sim": sim, "path": _p(state, "cache_lookup")}


def route_after_cache(state: AgenticState) -> str:
    return "output_guard" if state["cache_hit"] else "retrieve"


def route_after_output(state: AgenticState) -> str:
    return "remember" if state["cache_hit"] else "approval"


def cache_write(state: AgenticState) -> dict:
    """Only approved, grounded RAG answers are cached."""
    if (_use_cache() and state["approved"] and state["route"] == "rag"
            and not state["blocked"] and not state["cache_hit"]
            and NOT_FOUND not in state["answer"]):
        from semantic_cache import CACHE
        CACHE.put(_get_index()[3], state["task"], state["answer"])
        print("[cache] stored")
        return {"path": _p(state, "cache_write")}
    return {}


def output_guard(state: AgenticState) -> dict:
    from guardrails import detect_pii, redact_pii
    ans = state["answer"]
    if detect_pii(ans):
        ans = redact_pii(ans)
        print("[output_guard] PII redacted from answer")
    if OUTPUT_PRESIDIO:
        from guardrails_presi import check_output_for_pii_leakage
        print(f"[output_guard] presidio: {check_output_for_pii_leakage(ans)}")
    return {"answer": ans, "path": _p(state, "output_guard")}


# ---------------------------------------------------------------- tool branch
def tool_node(state: AgenticState) -> dict:
    from react_loop import execute_tool_with_retry
    expr = MATH_IN_Q.search(state["task"]).group(1).strip()
    result = execute_tool_with_retry("calculator", expr)
    print(f"[tool_node] calculator({expr!r}) -> {result}")
    return {"answer": f"{expr} = {result}", "path": _p(state, "tool_node")}


# ---------------------------------------------------------------- retrieval + grading
def _dist(r):
    for k in ("distance", "dist", "score", "l2", "similarity"):
        if k in r and isinstance(r[k], (int, float)):
            return float(r[k])
    return None


def retrieve_node(state: AgenticState) -> dict:
    from retrieve import search
    index, chunks, sources, model = _get_index()
    results = search(state["query"], index, chunks, sources, model, 3)
    findings = "\n\n".join(f"[{r['source']}] {r['chunk'][:300]}" for r in results)
    ds = [d for d in (_dist(r) for r in results) if d is not None]
    best = min(ds) if ds else -1.0
    if SHOW_SCORES:
        print(f"[retrieve] keys={list(results[0].keys()) if results else []} "
              f"dists={ds} query={state['query']!r}")
    print(f"[retrieve] {len(results)} chunks (attempt {state['attempts'] + 1})")
    return {"findings": findings, "best_dist": best, "path": _p(state, "retrieve")}


def _own_grade(question, findings) -> bool:
    prompt = f"""Question: {question}

Retrieved passages:
{findings}

Are these passages about the topic of the question, such that they could help answer it?
First quote the most relevant phrase from the passages (or write "none").
Then on a new line write VERDICT: YES or VERDICT: NO."""
    resp = counted_chat(
        client, model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=120,
    )
    text = resp.choices[0].message.content or ""
    if SHOW_SCORES:
        print(f"[grade raw] {text!r}")
    m = re.findall(r"VERDICT:\s*\W*(YES|NO)", text.upper())
    if m:
        return m[-1] == "YES"
    return False


def grade_node(state: AgenticState) -> dict:
    # 1) score gate: nothing close enough -> skip the LLM entirely
    if MAX_DIST is not None and state["best_dist"] >= 0 and state["best_dist"] > MAX_DIST:
        print(f"[grade] best distance {state['best_dist']:.3f} > {MAX_DIST} -> irrelevant")
        return {"verdict": "irrelevant", "path": _p(state, "grade(score)")}
    # 2) relevance check
    if GRADER == "score":
        if MAX_DIST is None or state["best_dist"] < 0:
            raise RuntimeError("GRADER=score needs MAX_DIST set and a distance key")
        ok = state["best_dist"] <= MAX_DIST
    elif GRADER == "crag":
        from corrective_rag import grade_context_sufficiency
        ok = bool(grade_context_sufficiency(
            state["query"], [c for c in state["findings"].split("\n\n") if c]))
    else:
        ok = _own_grade(state["query"], state["findings"])
    print(f"[grade] {'relevant' if ok else 'irrelevant'}")
    return {"verdict": "relevant" if ok else "irrelevant", "path": _p(state, "grade")}


def route_after_grade(state: AgenticState) -> str:
    if state["verdict"] == "relevant":
        return "generate"
    return "reformulate" if state["attempts"] < 1 else "abstain"


def reformulate_node(state: AgenticState) -> dict:
    from corrective_rag import reformulate_query
    new_q = (reformulate_query(state["task"]) or state["task"]).strip()
    print(f"[reformulate] {new_q!r}")
    return {"query": new_q, "attempts": state["attempts"] + 1,
            "path": _p(state, "reformulate")}


def abstain_node(state: AgenticState) -> dict:
    print("[abstain] no relevant evidence -> not answering from model knowledge")
    return {"answer": ABSTAIN_TEXT, "path": _p(state, "abstain")}


def generate_node(state: AgenticState) -> dict:
    prompt = build_prompt(state["task"], state["findings"])
    if USE_LTM:
        from agent_memory import recall
        past = recall(state["task"], model=_get_index()[3])
        if past:
            print(f"[memory] {len(past)} relevant past interaction(s)")
            prompt += ("\n\nEarlier conversations with this user (background only; "
                       "they are NOT evidence, answer only from the retrieved "
                       "information above):\n" + "\n".join(f"- {t}" for t in past))
    resp = counted_chat(
        client, model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1, max_tokens=300,
    )
    return {"answer": resp.choices[0].message.content, "path": _p(state, "generate")}


# ---------------------------------------------------------------- HITL
def approval_node(state: AgenticState) -> dict:
    if state["blocked"]:
        return {"approved": False, "path": _p(state, "approval(skipped)")}
    decision = interrupt({"question": "Approve this answer?",
                          "draft_answer": state["answer"]})
    print(f"[approval_node] human decision: {decision}")
    return {"approved": str(decision).strip().lower() == "approve",
            "path": _p(state, "approval")}


def build_agentic_graph():
    g = StateGraph(AgenticState)
    g.add_node("input_guard", input_guard)
    g.add_node("tool_node", tool_node)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("reformulate", reformulate_node)
    g.add_node("abstain", abstain_node)
    g.add_node("generate", generate_node)
    g.add_node("output_guard", output_guard)
    g.add_node("approval", approval_node)
    g.add_node("cache_lookup", cache_lookup)
    g.add_node("cache_write", cache_write)
    g.add_node("contextualize", contextualize)
    g.add_node("remember", remember)

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_input, {
        "tool_node": "tool_node", "contextualize": "contextualize", "approval": "approval"})
    g.add_edge("contextualize", "cache_lookup")
    g.add_conditional_edges("cache_lookup", route_after_cache, {
        "output_guard": "output_guard", "retrieve": "retrieve"})
    g.add_edge("tool_node", "output_guard")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", route_after_grade, {
        "generate": "generate", "reformulate": "reformulate", "abstain": "abstain"})
    g.add_edge("reformulate", "retrieve")
    g.add_edge("generate", "output_guard")
    g.add_edge("abstain", "output_guard")
    g.add_conditional_edges("output_guard", route_after_output, {
        "remember": "remember", "approval": "approval"})
    g.add_edge("approval", "cache_write")
    g.add_edge("cache_write", "remember")
    g.add_edge("remember", END)
    return g.compile(checkpointer=InMemorySaver())


QUERY = "What does the Navigator LLM do in CoLLM-NAS?"

if __name__ == "__main__":
    app = build_agentic_graph()

    def run_once(q=QUERY):
        cfg = {"configurable": {"thread_id": f"agentic-{next(_run_ids)}"}}
        r = app.invoke(_initial_state(q), config=cfg)
        if "__interrupt__" in r:
            r = app.invoke(Command(resume="approve"), config=cfg)
        return r

    if os.environ.get("BENCH"):
        _get_index()
        benchmark(lambda: run_once())
        raise SystemExit

    import sys
    if sys.argv[1:] == ["calibrate"]:
        # No LLM calls: best retrieval distance per eval question.
        import json
        from retrieve import search
        index, chunks, sources, model = _get_index()
        items = json.load(open(os.environ.get("EVAL_SET", "evals/agent_eval_set.json")))
        for it in items:
            res = search(it["question"], index, chunks, sources, model, 3)
            ds = [round(_dist(r), 3) for r in res if _dist(r) is not None]
            print(f"{it['type']:<4} {it['id']:<14} best={min(ds) if ds else None}  all={ds}  {it['question'][:50]}")
        raise SystemExit

    q = os.environ.get("Q", QUERY)
    cfg = {"configurable": {"thread_id": "agentic-hitl-1"}}
    r = app.invoke(_initial_state(q), config=cfg)
    if "__interrupt__" in r:
        d = r["__interrupt__"][0].value
        print(f"\n--- PAUSED FOR HUMAN APPROVAL ---\nDraft answer: {d['draft_answer']}")
        human = input("Approve? (approve/reject): ").strip().lower()
        r = app.invoke(Command(resume=human), config=cfg)
    if r.get("approved"):
        print(f"\nApproved: True\nFinal answer: {r['answer']}")
    else:
        print("\nApproved: False\nFinal answer: [BLOCKED - not approved, not returned to user]")
    print(f"Path: {r['path']}")