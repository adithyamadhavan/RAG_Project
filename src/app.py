# app.py  (put in the RAG_Project root, run with: streamlit run app.py)
# Thin Streamlit shell. Backend features plug in later by adding keys to the
# dict returned by run_supervisor() / run_swarm().
import os, sys, time, uuid
import streamlit as st
from langgraph.types import Command

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
import supervisor_agent as sup
import swarm_agents as sw
from retrieve import search
from bench_utils import counted_chat
from prompts import build_prompt

from bench_utils import CALLS

st.set_page_config(page_title="RAG + Agents", layout="wide")


@st.cache_resource
def get_supervisor_app():
    # Load the index once so it is not paid on every query.
    if hasattr(sup, "_get_index"):
        sup._get_index()
    return sup.build_supervisor_graph()


def _initial_state(query):
    return {"messages": [], "next_agent": "", "task": query, "use_web": False,
            "retriever_output": "", "web_output": "",
            "synthesis_output": "", "approved": False}


def run_supervisor(query):
    """Run until the approval interrupt. Returns a dict for the UI."""
    app = get_supervisor_app()
    thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": thread_id}}
    if CALLS is not None:
        CALLS.clear()
    t0 = time.perf_counter()
    result = app.invoke(_initial_state(query), config=cfg)
    elapsed = time.perf_counter() - t0
    out = {"thread_id": thread_id, "latency": elapsed,
           "llm_calls": CALLS["llm"] if CALLS is not None else None,
           "sources": result.get("retriever_output", ""),
           "cache_hit": None}                       # filled in on Oct 12
    if "__interrupt__" in result:
        out["draft"] = result["__interrupt__"][0].value["draft_answer"]
        out["needs_approval"] = True
    else:
        out["draft"] = result.get("synthesis_output", "")
        out["needs_approval"] = False
    return out


def resume_supervisor(thread_id, decision):
    app = get_supervisor_app()
    cfg = {"configurable": {"thread_id": thread_id}}
    t0 = time.perf_counter()                        # human wait is NOT included
    final = app.invoke(Command(resume=decision), config=cfg)
    return final.get("approved", False), final.get("synthesis_output", ""), time.perf_counter() - t0


@st.cache_resource
def get_swarm_app():
    if hasattr(sw, "_get_index"):
        sw._get_index()
    return sw.build_swarm_graph()


def run_swarm(query):
    app = get_swarm_app()
    thread_id = f"ui-{uuid.uuid4().hex[:8]}"
    cfg = {"configurable": {"thread_id": thread_id}}
    if CALLS is not None:
        CALLS.clear()
    t0 = time.perf_counter()
    r = app.invoke({"messages": [], "task": query, "findings": "",
                    "answer": "", "handoff_to": ""}, config=cfg)
    return {"thread_id": thread_id, "latency": time.perf_counter() - t0,
            "llm_calls": CALLS["llm"] if CALLS is not None else None,
            "sources": r.get("findings", ""), "cache_hit": None,
            "draft": r["answer"], "needs_approval": False}


def run_rag(query):
    """Plain RAG: retrieve, then one LLM call. No graph, routing or approval."""
    index, chunks, sources, model = sup._get_index()   # loaded once
    CALLS.clear()
    t0 = time.perf_counter()
    results = search(query, index, chunks, sources, model, 3)
    findings = "\n\n".join(f"[{r['source']}] {r['chunk'][:300]}" for r in results)
    prompt = build_prompt(query, findings)
    resp = counted_chat(sup.client, model=sup.MODEL_NAME,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.1, max_tokens=300)
    return {"thread_id": None, "latency": time.perf_counter() - t0,
            "llm_calls": CALLS["llm"], "sources": findings, "cache_hit": None,
            "draft": resp.choices[0].message.content, "needs_approval": False}


# ---------------- UI ----------------
st.title("RAG + Agents demo")
mode = st.sidebar.radio("Mode", ["Plain RAG", "Supervisor", "Swarm"])
st.sidebar.caption("Plain RAG: retrieve + one LLM call. Supervisor adds a human approval step.")

if "last" not in st.session_state:
    st.session_state.last = None
    st.session_state.final = None

query = st.text_input("Ask about the NAS papers",
                      value="What does the Navigator LLM do in CoLLM-NAS?")
if st.button("Run", type="primary"):
    st.session_state.final = None
    with st.spinner("Running..."):
        runner = {"Plain RAG": run_rag, "Supervisor": run_supervisor, "Swarm": run_swarm}[mode]
        st.session_state.last = runner(query)

res = st.session_state.last
if res:
    c1, c2, c3 = st.columns(3)
    c1.metric("Latency", f"{res['latency']:.1f}s")
    c2.metric("LLM calls", res["llm_calls"] if res["llm_calls"] is not None else "n/a")
    c3.metric("Cache", {True: "hit", False: "miss", None: "off"}[res["cache_hit"]])

    if res["needs_approval"] and st.session_state.final is None:
        st.subheader("Draft answer (awaiting approval)")
        st.write(res["draft"])
        a, r = st.columns(2)
        if a.button("Approve"):
            st.session_state.final = resume_supervisor(res["thread_id"], "approve")
            st.rerun()
        if r.button("Reject"):
            st.session_state.final = resume_supervisor(res["thread_id"], "reject")
            st.rerun()
    elif st.session_state.final is not None:
        approved, answer, _ = st.session_state.final
        if approved:
            st.success("Approved")
            st.write(answer)
        else:
            st.error("Blocked: answer was not approved by the reviewer.")
    else:
        st.write(res["draft"])

    if res["sources"]:
        with st.expander("Retrieved sources"):
            st.text(res["sources"])