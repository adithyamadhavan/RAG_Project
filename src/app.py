# src/app.py
# Streamlit chat front end. It does NOT import the graph: it only talks to the
# FastAPI service over HTTP (POST /ask, POST /approve), so the UI and the
# backend are separate programs.
#
# Three terminals:
#   1) the model server (vllm-mlx on :8000)
#   2) uvicorn api:app --app-dir src --port 8001
#   3) streamlit run src/app.py
#
#   pip install streamlit requests
import requests
import streamlit as st

st.set_page_config(page_title="Agentic RAG", page_icon="🔎", layout="centered")

# ---------------------------------------------------------------- state
# Streamlit re-runs this whole script on every click, so anything that must
# survive lives in st.session_state.
ss = st.session_state
ss.setdefault("messages", [])      # [{role, content, meta}]
ss.setdefault("thread_id", None)   # same id = same conversation (memory)
ss.setdefault("pending", None)     # {"draft": str} while waiting for approval
ss.setdefault("error", None)

EXAMPLES = [
    "What does the Navigator LLM do in CoLLM-NAS?",
    "What are its key components?",
    "What is the difference between the Navigator LLM and the Generator LLM?",
    "What is 15 * 23 + 7?",
    "What is the capital of France?",
]


# ---------------------------------------------------------------- API calls
def call_api(path: str, payload: dict):
    """POST to the backend; returns the JSON or sets an error message."""
    try:
        r = requests.post(f"{ss.api_url}{path}", json=payload, timeout=180)
    except requests.exceptions.ConnectionError:
        ss.error = f"Cannot reach the API at {ss.api_url}. Is uvicorn running?"
        return None
    except requests.exceptions.Timeout:
        ss.error = "The request timed out (the model may be busy)."
        return None
    if r.status_code == 404 and path == "/approve":
        ss.pending, ss.thread_id = None, None
        ss.error = "That conversation was lost (the server restarted). Ask again."
        return None
    if r.status_code != 200:
        ss.error = f"API error {r.status_code}: {r.text[:200]}"
        return None
    return r.json()


def tags(resp: dict) -> list:
    """Short labels explaining how the answer was produced."""
    p = resp.get("path", "")
    out = []
    if resp.get("cache_hit"):
        out.append("⚡ cache hit")
    if "tool_node" in p:
        out.append("🧮 calculator")
    if "reformulate" in p:
        out.append("↻ query rewritten")
    if "contextualize>" in p:
        out.append("💬 used chat context")
    if "abstain" in p:
        out.append("🚫 not in corpus")
    out.append(f"{resp.get('latency_ms', 0) / 1000:.1f}s")
    return out


def handle(resp: dict):
    """Turn an API response into chat messages / a pending approval."""
    ss.thread_id = resp["thread_id"]
    status = resp["status"]
    if status == "pending_approval":
        ss.pending = {"draft": resp["draft"], "meta": resp}
        return
    ss.pending = None
    if status == "done":
        text = resp["answer"]
    elif status == "blocked":
        text = f"🛑 {resp['answer']}"
    else:
        text = "❌ The reviewer rejected this answer, so it was not returned."
    ss.messages.append({"role": "assistant", "content": text, "meta": resp})


def ask(question: str):
    ss.error = None
    ss.messages.append({"role": "user", "content": question})
    payload = {"question": question}
    if ss.thread_id:
        payload["thread_id"] = ss.thread_id
    resp = call_api("/ask", payload)
    if resp:
        handle(resp)


def decide(decision: str):
    ss.error = None
    resp = call_api("/approve", {"thread_id": ss.thread_id, "decision": decision})
    if resp:
        handle(resp)


def new_conversation():
    ss.messages, ss.thread_id, ss.pending, ss.error = [], None, None, None


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    st.text_input("API URL", value=  "http://127.0.0.1:8001", key="api_url")
    if st.button("Check API"):
        try:
            h = requests.get(f"{ss.api_url}/health", timeout=5).json()
            st.success(f"API up · index loaded: {h['index_loaded']}")
        except Exception as e:
            st.error(f"API not reachable: {type(e).__name__}")
    st.button("New conversation", on_click=new_conversation)
    st.caption(f"Thread: `{ss.thread_id[:8]}…`" if ss.thread_id else "Thread: none yet")
    st.divider()
    st.subheader("Try these")
    for ex in EXAMPLES:
        if st.button(ex, key=f"ex-{ex}", disabled=ss.pending is not None):
            ask(ex)
            st.rerun()

# ---------------------------------------------------------------- main page
st.title("🔎 Agentic RAG")
st.caption("Answers come only from the indexed NAS papers. Every answer waits for "
           "your approval before it is returned.")

for m in ss.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        meta = m.get("meta")
        if meta:
            st.caption(" · ".join(tags(meta)))
            with st.expander("How this answer was produced"):
                st.write(" → ".join(x for x in meta["path"].split(">") if x))

if ss.pending:
    with st.chat_message("assistant"):
        st.markdown("**Draft answer, waiting for your approval**")
        st.markdown(ss.pending["draft"])
        st.caption(" · ".join(tags(ss.pending["meta"])))
        c1, c2, _ = st.columns([1, 1, 3])
        c1.button("✅ Approve", on_click=decide, args=("approve",), type="primary")
        c2.button("❌ Reject", on_click=decide, args=("reject",))

if ss.error:
    st.error(ss.error)

question = st.chat_input("Ask about the papers…", disabled=ss.pending is not None)
if question:
    ask(question)
    st.rerun()