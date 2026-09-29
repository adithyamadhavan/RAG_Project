import inspect
from functools import lru_cache
from typing import Callable, Dict, get_type_hints


def tool(func: Callable) -> Callable:
    """Attaches a tool schema (name, description, parameters) built from the
    function's signature and docstring. The docstring is effectively a prompt,
    because it is all the LLM ever sees of this tool."""
    sig = inspect.signature(func)
    hints = get_type_hints(func)
    func.tool_schema = {
        "name": func.__name__,
        "description": inspect.getdoc(func) or "No description provided.",
        "parameters": {
            name: hints.get(name, str).__name__ for name in sig.parameters
        },
    }
    return func


@tool
def calculator(expression: str) -> str:
    """Evaluates an arithmetic expression such as '15 * 23 + 7'. Use for any math."""
    allowed = set("0123456789+-*/(). ")
    if not expression or not all(c in allowed for c in expression):
        return "Error: expression contains disallowed characters."
    try:
        # Whitelist check above makes eval acceptable for a demo.
        # In production, use a real expression parser instead of eval.
        return str(eval(expression))
    except Exception as e:
        return f"Error evaluating expression: {e}"


@lru_cache(maxsize=1)
def _load_retriever():
    """Load the FAISS index and embedding model once, not on every tool call."""
    from retrieve import load_index_data
    return load_index_data()


@tool
def search_nas_corpus(query: str) -> str:
    """Searches the neural architecture search paper corpus (CoLLM-NAS, RZ-NAS,
    RegNet, DiffusionNAG and others). Use for questions about those papers."""
    from retrieve import search
    index, chunks, sources, model = _load_retriever()
    results = search(query, index, chunks, sources, model, 3)  # positional: avoids k/top_k naming drift
    if not results:
        return "No relevant information found in the corpus."
    return "\n\n".join(f"[{r['source']}] {r['chunk'][:300]}" for r in results)


AVAILABLE_TOOLS: Dict[str, Callable] = {
    "calculator": calculator,
    "search_nas_corpus": search_nas_corpus,
}


if __name__ == "__main__":
    print("--- Schemas the LLM will see ---")
    for name, fn in AVAILABLE_TOOLS.items():
        print(name, "->", fn.tool_schema)

    print("\n--- Direct tool calls, no LLM ---")
    print("calculator('15 * 23 + 7') =", calculator("15 * 23 + 7"))
    print("calculator('import os') =", calculator("import os"))
    print("\nsearch_nas_corpus('Navigator LLM'):")
    print(search_nas_corpus("Navigator LLM"))