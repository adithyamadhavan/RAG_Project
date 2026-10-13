# src/workflows.py
# Two workflow patterns on top of the same retrieval and model as the other modes.
#
# 1) EVALUATOR-OPTIMIZER   (generate -> evaluate -> revise, capped)
#    The evaluator is DETERMINISTIC, not an LLM: the 1.5B model cannot judge
#    reliably (it said NO to everything as a grader). It splits the draft into
#    sentences and flags any sentence whose content words, or any number, are
#    not found in the retrieved text. Flagged sentences go back to the model
#    as feedback; after MAX_REVISIONS the unsupported sentences are removed,
#    and if nothing is left the answer becomes NOT_IN_CORPUS.
#
# 2) ROUTING   (classify the question, send it to a specialised handler)
#      math     -> calculator tool
#      compare  -> retrieve for EACH side of "X vs Y" separately, then answer
#      qa       -> evaluator-optimizer above
#    Routing is rule-based (regex): same lesson as the LLM-routed supervisor,
#    where the 1.5B model misrouted 1 decision in 3.
#
#   python src/workflows.py "your question"      -> one run, prints the path
#   BENCH=1 python src/workflows.py              -> 1 warm-up + 3 timed runs
#   Env: GROUND_MIN (default 0.4) share of a sentence's content words that must
#        appear in the sources; MAX_REVISIONS (default 2)
import os
import re
import sys
from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from bench_utils import counted_chat, benchmark
from prompts import build_prompt, NOT_FOUND
import agentic_rag as ag

GROUND_MIN = float(os.environ.get("GROUND_MIN", "0.4"))
MAX_REVISIONS = int(os.environ.get("MAX_REVISIONS", "2"))

STOP = set("""this that with from have been were will would could should their there which about
also into than then them they such these those other more most some only over very what when
where while does doing each both used using use uses based make made allow allows within across
between through during after before""".split())
TOKEN = re.compile(r"[a-z0-9]+(?:[-.][a-z0-9]+)*")


# ---------------------------------------------------------------- evaluator
def _tokens(text):
    return set(TOKEN.findall(text.lower()))


def _stem(w):
    return w[:-1] if len(w) > 4 and w.endswith("s") else w


def _sentences(answer: str) -> list:
    """Split into checkable sentences. List markers ("1.", "-", "*") and markdown
    bold are removed first, so "2." is never mistaken for an invented figure."""
    clean = re.sub(r"(?m)^\s*(?:\d+[.)]|[-*\u2022])\s+", "", answer.strip())
    clean = clean.replace("**", "")
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", clean) if x.strip()]


def unsupported_sentences(answer: str, findings: str) -> list:
    """Sentences of `answer` that the retrieved text does not support."""
    src = {_stem(t) for t in _tokens(findings)}
    bad = []
    for sent in _sentences(answer):
        toks = TOKEN.findall(sent.lower())
        numbers = [t for t in toks if any(c.isdigit() for c in t)]
        if any(_stem(n) not in src for n in numbers):      # invented figure
            if os.environ.get("SHOW_SCORES"):
                print(f"[ground] NUMBER not in sources: {sent[:70]!r}")
            bad.append(sent)
            continue
        words = [_stem(t) for t in toks if len(t) >= 4 and t not in STOP]
        if len(words) < 2:                                   # too short to judge
            continue
        ratio = sum(w in src for w in words) / len(words)
        if os.environ.get("SHOW_SCORES"):
            print(f"[ground] {ratio:.2f} {'OK ' if ratio >= GROUND_MIN else 'BAD'} {sent[:70]!r}")
        if ratio < GROUND_MIN:
            bad.append(sent)
    return bad


# ---------------------------------------------------------------- evaluator-optimizer graph
class EOState(TypedDict):
    question: str
    findings: str
    best_dist: float
    answer: str
    unsupported: list
    revisions: int
    path: str


def _p(s, node):
    return (s.get("path") or "") + node + ">"


def eo_retrieve(s: EOState) -> dict:
    from retrieve import search
    index, chunks, sources, model = ag._get_index()
    res = search(s["question"], index, chunks, sources, model, 3)
    findings = "\n\n".join(f"[{r['source']}] {r['chunk'][:300]}" for r in res)
    ds = [ag._dist(r) for r in res if ag._dist(r) is not None]
    return {"findings": findings, "best_dist": min(ds) if ds else -1.0,
            "path": _p(s, "retrieve")}


def eo_after_retrieve(s: EOState) -> str:
    far = s["best_dist"] >= 0 and ag.MAX_DIST is not None and s["best_dist"] > ag.MAX_DIST
    return "abstain" if far else "generate"


def eo_abstain(s: EOState) -> dict:
    return {"answer": ag.ABSTAIN_TEXT, "path": _p(s, "abstain")}


def eo_generate(s: EOState) -> dict:
    resp = counted_chat(ag.client, model=ag.MODEL_NAME,
                        messages=[{"role": "user",
                                   "content": build_prompt(s["question"], s["findings"])}],
                        temperature=0.1, max_tokens=300)
    return {"answer": resp.choices[0].message.content or "", "path": _p(s, "generate")}


def eo_evaluate(s: EOState) -> dict:
    if NOT_FOUND in s["answer"]:
        bad = []
    else:
        bad = unsupported_sentences(s["answer"], s["findings"])
    if bad:
        print(f"[evaluate] {len(bad)} unsupported sentence(s)")
    return {"unsupported": bad, "path": _p(s, "evaluate")}


def eo_route(s: EOState) -> str:
    if not s["unsupported"]:
        return "done"
    return "revise" if s["revisions"] < MAX_REVISIONS else "trim"


def eo_revise(s: EOState) -> dict:
    listing = "\n".join(f"- {x}" for x in s["unsupported"])
    prompt = f"""Question: {s['question']}

Retrieved information:
{s['findings']}

Draft answer:
{s['answer']}

These statements in the draft are NOT supported by the retrieved information:
{listing}

Rewrite the answer using only the retrieved information. Remove or correct the unsupported statements.
If nothing is supported, reply with exactly: {NOT_FOUND}"""
    resp = counted_chat(ag.client, model=ag.MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1, max_tokens=300)
    return {"answer": resp.choices[0].message.content or "",
            "revisions": s["revisions"] + 1, "path": _p(s, "revise")}


def eo_trim(s: EOState) -> dict:
    """Out of revisions: keep only the supported sentences."""
    bad = set(s["unsupported"])
    kept = [x for x in _sentences(s["answer"]) if x not in bad]
    answer = " ".join(kept).strip() or ag.ABSTAIN_TEXT
    return {"answer": answer, "unsupported": [], "path": _p(s, "trim")}


def build_eo_graph():
    g = StateGraph(EOState)
    for name, fn in [("retrieve", eo_retrieve), ("abstain", eo_abstain),
                     ("generate", eo_generate), ("evaluate", eo_evaluate),
                     ("revise", eo_revise), ("trim", eo_trim)]:
        g.add_node(name, fn)
    g.add_edge(START, "retrieve")
    g.add_conditional_edges("retrieve", eo_after_retrieve,
                            {"abstain": "abstain", "generate": "generate"})
    g.add_edge("abstain", END)
    g.add_edge("generate", "evaluate")
    g.add_conditional_edges("evaluate", eo_route,
                            {"done": END, "revise": "revise", "trim": "trim"})
    g.add_edge("revise", "evaluate")
    g.add_edge("trim", END)
    return g.compile()


def eo_initial(question):
    return {"question": question, "findings": "", "best_dist": -1.0, "answer": "",
            "unsupported": [], "revisions": 0, "path": ""}


# ---------------------------------------------------------------- routing graph
COMPARE_PATTERNS = [
    re.compile(r"(?:difference between|differences between|compare|comparison of)\s+"
               r"(.+?)\s+(?:and|with|vs\.?|versus)\s+(.+?)[?.\s]*$", re.I),
    re.compile(r"^(.+?)\s+(?:vs\.?|versus)\s+(.+?)[?.\s]*$", re.I),
]


class RouteState(TypedDict):
    question: str
    route: str
    answer: str
    findings: str
    path: str


def classify(question: str) -> str:
    if ag.MATH_IN_Q.search(question) and len(question) < 80:
        return "math"
    if any(p.search(question) for p in COMPARE_PATTERNS):
        return "compare"
    return "qa"


def r_route(s: RouteState) -> dict:
    r = classify(s["question"])
    print(f"[router] -> {r}")
    return {"route": r, "path": _p(s, f"router({r})")}


def r_math(s: RouteState) -> dict:
    from react_loop import execute_tool_with_retry
    expr = ag.MATH_IN_Q.search(s["question"]).group(1).strip()
    return {"answer": f"{expr} = {execute_tool_with_retry('calculator', expr)}",
            "findings": "", "path": _p(s, "math")}


def _compare_sides(question):
    for p in COMPARE_PATTERNS:
        m = p.search(question)
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return None


def r_compare(s: RouteState) -> dict:
    from retrieve import search
    index, chunks, sources, model = ag._get_index()
    left, right = _compare_sides(s["question"])
    blocks, far = [], False
    for side in (left, right):
        res = search(side, index, chunks, sources, model, 2)
        ds = [ag._dist(r) for r in res if ag._dist(r) is not None]
        if ag.MAX_DIST is not None and ds and min(ds) > ag.MAX_DIST:
            far = True
        blocks.append(f"About {side}:\n" + "\n".join(
            f"[{r['source']}] {r['chunk'][:300]}" for r in res))
    findings = "\n\n".join(blocks)
    if far:
        return {"answer": ag.ABSTAIN_TEXT, "findings": findings,
                "path": _p(s, "compare(abstain)")}
    resp = counted_chat(ag.client, model=ag.MODEL_NAME,
                        messages=[{"role": "user",
                                   "content": build_prompt(s["question"], findings)}],
                        temperature=0.1, max_tokens=350)
    return {"answer": resp.choices[0].message.content or "", "findings": findings,
            "path": _p(s, "compare")}


_EO = None


def r_qa(s: RouteState) -> dict:
    global _EO
    if _EO is None:
        _EO = build_eo_graph()
    r = _EO.invoke(eo_initial(s["question"]))
    return {"answer": r["answer"], "findings": r["findings"],
            "path": _p(s, "qa[" + r["path"].rstrip(">") + "]")}


def build_workflow_graph():
    g = StateGraph(RouteState)
    g.add_node("router", r_route)
    g.add_node("math", r_math)
    g.add_node("compare", r_compare)
    g.add_node("qa", r_qa)
    g.add_edge(START, "router")
    g.add_conditional_edges("router", lambda s: s["route"],
                            {"math": "math", "compare": "compare", "qa": "qa"})
    for n in ("math", "compare", "qa"):
        g.add_edge(n, END)
    return g.compile()


def initial(question):
    return {"question": question, "route": "", "answer": "", "findings": "", "path": ""}


if __name__ == "__main__":
    app = build_workflow_graph()
    ag._get_index()
    q = " ".join(sys.argv[1:]) or "What does the Navigator LLM do in CoLLM-NAS?"
    if os.environ.get("BENCH"):
        benchmark(lambda: app.invoke(initial(q)))
        raise SystemExit
    r = app.invoke(initial(q))
    print(f"\nANSWER: {r['answer']}\nPATH: {r['path']}")