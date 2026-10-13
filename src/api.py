# src/api.py
# FastAPI wrapper around the agentic RAG graph.
#
#   pip install fastapi uvicorn
#   uvicorn api:app --reload --app-dir src        (run from the project root)
#   open http://127.0.0.1:8001/docs  (add --port 8001; vllm already uses 8000)
#
# Endpoints
#   GET  /health            liveness + whether the index is loaded
#   POST /ask               run the graph. Returns either a final answer
#                           (status "done") or a draft waiting for a human
#                           (status "pending_approval", with a thread_id)
#   POST /approve           resume a paused run with "approve" or "reject"
#
# The human-in-the-loop step maps onto HTTP like this: the graph pauses at
# interrupt(); /ask returns the draft + thread_id; the client later calls
# /approve with that thread_id, which resumes the SAME run from the
# checkpointer. That is why thread_id matters.
#
# Notes
#   - Endpoints are plain `def`, so FastAPI runs them in a thread pool and the
#     blocking LangGraph / LLM calls do not freeze the server.
#   - vllm-mlx serves one generation at a time; simultaneous requests are
#     retried by counted_chat (see bench_utils).
#   - The checkpointer is in memory: pending approvals are lost on restart.
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from langgraph.types import Command
from pydantic import BaseModel, Field

import agentic_rag as ag


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup: load the index and build the graph one time,
    # not on every request.
    ag._get_index()
    app.state.graph = ag.build_agentic_graph()
    yield


app = FastAPI(title="Agentic RAG API", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


class ApproveRequest(BaseModel):
    thread_id: str
    decision: Literal["approve", "reject"]


class AskResponse(BaseModel):
    status: Literal["done", "pending_approval", "blocked", "rejected"]
    thread_id: str
    answer: Optional[str] = None       # final answer (done)
    draft: Optional[str] = None        # waiting for approval
    cache_hit: bool = False
    path: str = ""                     # nodes visited, handy for debugging
    latency_ms: int = 0


def _cfg(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _to_response(result: dict, thread_id: str, t0: float) -> AskResponse:
    ms = int((time.perf_counter() - t0) * 1000)
    base = dict(thread_id=thread_id, path=result.get("path", ""),
                cache_hit=result.get("cache_hit", False), latency_ms=ms)
    if "__interrupt__" in result:
        draft = result["__interrupt__"][0].value["draft_answer"]
        return AskResponse(status="pending_approval", draft=draft, **base)
    if result.get("blocked"):
        return AskResponse(status="blocked", answer=result["answer"], **base)
    if result.get("cache_hit") or result.get("approved"):
        return AskResponse(status="done", answer=result["answer"], **base)
    return AskResponse(status="rejected", **base)


@app.get("/health")
def health():
    return {"status": "ok", "index_loaded": ag._INDEX is not None}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    thread_id = uuid.uuid4().hex
    t0 = time.perf_counter()
    result = app.state.graph.invoke(ag._initial_state(req.question),
                                    config=_cfg(thread_id))
    return _to_response(result, thread_id, t0)


@app.post("/approve", response_model=AskResponse)
def approve(req: ApproveRequest):
    graph = app.state.graph
    state = graph.get_state(_cfg(req.thread_id))
    if not state.next:      # unknown thread, or nothing is waiting
        raise HTTPException(status_code=404,
                            detail="No pending approval for this thread_id")
    t0 = time.perf_counter()
    result = graph.invoke(Command(resume=req.decision), config=_cfg(req.thread_id))
    return _to_response(result, req.thread_id, t0)