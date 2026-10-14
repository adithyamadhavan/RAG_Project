# src/mcp_server.py
# MCP server exposing the RAG project as two READ-ONLY tools to any MCP client
# (Claude Code, Claude Desktop, MCP Inspector):
#   search_corpus(query, k)    -> top-k chunks with source and distance (no LLM)
#   answer_question(question)  -> full agentic RAG answer (guards, abstain, grounded)
#
#   pip install "mcp<2"        # uses mcp.server.fastmcp.FastMCP
#   (mcp 2.x renamed it to MCPServer; the import below handles both)
#
# Run from the project root so relative paths (faiss_index.bin etc.) resolve:
#   python src/mcp_server.py                      # speaks MCP over stdio
#   npx @modelcontextprotocol/inspector python src/mcp_server.py   # try it in a browser
#
# IMPORTANT: over stdio, stdout IS the protocol channel. Any stray print() there
# corrupts it, so imports and tool bodies run with stdout redirected to stderr.
import contextlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)                               # the client may launch us from anywhere
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("EVAL_RUN", "1")       # read-only: never write to long-term memory

try:
    from mcp.server.fastmcp import FastMCP as _Server
except ImportError:
    from mcp.server.mcpserver import MCPServer as _Server

with contextlib.redirect_stdout(sys.stderr):
    import agentic_rag as ag
    from langgraph.types import Command
    from retrieve import search

mcp = _Server("nas-papers-rag")
_APP = None


def _app():
    global _APP
    if _APP is None:
        with contextlib.redirect_stdout(sys.stderr):
            ag._get_index()
            _APP = ag.build_agentic_graph()
    return _APP


@mcp.tool()
def search_corpus(query: str, k: int = 3) -> str:
    """Search the indexed neural-architecture-search papers and return the top
    matching passages with their source PDF and distance (lower = closer).
    Use this to look at raw evidence. Does not call a language model."""
    query = (query or "").strip()
    if not 3 <= len(query) <= 500:
        return "Error: query must be 3-500 characters."
    k = max(1, min(int(k), 8))
    with contextlib.redirect_stdout(sys.stderr):
        index, chunks, sources, model = ag._get_index()
        res = search(query, index, chunks, sources, model, k)
    out = []
    for r in res:
        d = ag._dist(r)
        out.append(f"[{r['rank']}] {r['source']} (distance {d:.3f})\n{r['chunk'][:700]}")
    return "\n\n".join(out) or "No results."


@mcp.tool()
def answer_question(question: str) -> str:
    """Answer a question about the indexed NAS papers (CoLLM-NAS, RZ-NAS, RegNet,
    DiffusionNAG, ...). Answers only from the papers; replies NOT_IN_CORPUS when
    the papers do not contain the answer. Slow on a small local model (a few seconds)."""
    question = (question or "").strip()
    if not 3 <= len(question) <= 500:
        return "Error: question must be 3-500 characters."
    app = _app()
    cfg = {"configurable": {"thread_id": f"mcp-{os.urandom(4).hex()}"}}
    with contextlib.redirect_stdout(sys.stderr):
        r = app.invoke(ag._initial_state(question), config=cfg)
        if "__interrupt__" in r:                     # no human in the loop over MCP
            r = app.invoke(Command(resume="approve"), config=cfg)
    return r.get("answer") or "No answer produced."


if __name__ == "__main__":
    mcp.run()