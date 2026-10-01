from dotenv import load_dotenv
load_dotenv()

from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from react_loop import execute_tool_with_retry
from agents_tools import AVAILABLE_TOOLS
class AgentState(TypedDict):
    # add_messages is a reducer: instead of overwriting the list on each
    # update, it APPENDS to it — this is what your messages.append(...)
    # calls were doing by hand in react_loop.py.
    messages: Annotated[list, add_messages]
    iterations: int
    last_call: tuple | None
import re
import os
from openai import OpenAI
from agent_memory import LongTermMemory


VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed", timeout=60.0)
memory = LongTermMemory()
RELEVANCE_THRESHOLD = 0.5
MAX_ITERATIONS = 5

# def build_system_prompt() -> str:
#     tool_lines = "\n".join(
#         f"- {name}({', '.join(fn.tool_schema['parameters'])}): {fn.tool_schema['description']}"
#         for name, fn in AVAILABLE_TOOLS.items()
#     )
#     return f"""You are an agent that can use tools to answer questions.

# Available tools:
# {tool_lines}

# Respond with EXACTLY ONE line, in one of these two forms. Replace the
# placeholder with a real value.

# TOOL_CALL: calculator(15 * 23 + 7)
# FINAL_ANSWER: 352

# Write ONLY ONE line per turn. Do not write both on the same turn."""

def build_system_prompt() -> str:
    tool_lines = "\n".join(
        f"- {name}({', '.join(fn.tool_schema['parameters'])}): {fn.tool_schema['description']}"
        for name, fn in AVAILABLE_TOOLS.items()
    )
    return f"""You are an agent that can use tools to answer questions.

Available tools:
{tool_lines}

Before every reply, first check: does the conversation already contain a
tool result that answers the question? If yes, you MUST write FINAL_ANSWER
using that result — do not call any tool again.

Only if the answer is not yet in the conversation, respond with:
TOOL_CALL: calculator(15 * 23 + 7)

Otherwise respond with:
FINAL_ANSWER: 352

Write ONLY ONE line per turn. Do not write both on the same turn."""


def parse_tool_call(text: str):
    match = re.search(r'TOOL_CALL:\s*(\w+)\(([^)]*)\)', text)
    if match:
        return match.group(1), match.group(2).strip().strip('"\'')
    return None, None


# def call_model_node(state: AgentState) -> dict:
#     """Node 1: ask the LLM for the next step, given everything so far."""
#     response = client.chat.completions.create(
#         model=MODEL_NAME,
#         messages=state["messages"],
#         temperature=0.1,
#         max_tokens=300,
#     )
#     content = response.choices[0].message.content
#     print(f"\n[iteration {state['iterations'] + 1}] raw model output:\n{content}")

#     # Strip any trailing FINAL_ANSWER that rode along with a tool call —
#     # same fix from react_loop.py, same reason: don't let the model's own
#     # history claim it already answered when it hasn't seen a result yet.
#     tool_name, _ = parse_tool_call(content)
#     if tool_name:
#         content = content.split("FINAL_ANSWER:")[0].strip()

#     return {
#         "messages": [{"role": "assistant", "content": content}],
#         "iterations": state["iterations"] + 1,
#     }
def call_model_node(state: AgentState) -> dict:
    openai_messages = []
    for m in state["messages"]:
        if isinstance(m, dict):
            openai_messages.append(m)
        else:
            role = "assistant" if m.type == "ai" else ("system" if m.type == "system" else "user")
            openai_messages.append({"role": role, "content": m.content})

    response = client.chat.completions.create(
        model=MODEL_NAME, messages=openai_messages, temperature=0.1, max_tokens=300,
    )
    content = response.choices[0].message.content
    print(f"\n[iteration {state['iterations'] + 1}] raw model output:\n{content}")

    tool_name, tool_arg = parse_tool_call(content)
    if tool_name:
        content = content.split("FINAL_ANSWER:")[0].strip()
        # Repeat-call guard: if this is the SAME call as last time, the
        # model is stuck. Force it to answer from what it already has
        # instead of letting it loop to the iteration cap.
        if (tool_name, tool_arg) == state.get("last_call"):
            last_tool_msg = next(
                (m for m in reversed(state["messages"])
                 if (m.get("content") if isinstance(m, dict) else m.content).startswith("Tool result:")),
                None,
            )
            observation = (last_tool_msg.get("content") if isinstance(last_tool_msg, dict)
                            else last_tool_msg.content) if last_tool_msg else "no prior result available"
            print(f"[iteration {state['iterations'] + 1}] repeated identical call detected — forcing synthesis")
            content = f"FINAL_ANSWER: Based on the available information: {observation[:300]}"

    return {
        "messages": [{"role": "assistant", "content": content}],
        "iterations": state["iterations"] + 1,
        "last_call": (tool_name, tool_arg) if tool_name else state.get("last_call"),
    }


# def call_tool_node(state: AgentState) -> dict:
#     """Node 2: parse the last message, run the tool, return the observation."""
#     last_message = state["messages"][-1]["content"]
#     tool_name, tool_arg = parse_tool_call(last_message)
#     print(f"[iteration {state['iterations']}] calling {tool_name}({tool_arg!r})")
#     result = execute_tool_with_retry(tool_name, tool_arg)
#     print(f"[iteration {state['iterations']}] tool result: {result[:150]}")
#     return {"messages": [{"role": "user", "content": f"Tool result: {result}"}]}


def call_tool_node(state: AgentState) -> dict:
    last = state["messages"][-1]
    last_content = last.content if hasattr(last, "content") else last["content"]
    tool_name, tool_arg = parse_tool_call(last_content)
    print(f"[iteration {state['iterations']}] calling {tool_name}({tool_arg!r})")
    result = execute_tool_with_retry(tool_name, tool_arg)
    print(f"[iteration {state['iterations']}] tool result: {result[:150]}")
    return {"messages": [{"role": "user", "content": f"Tool result: {result}"}]}
# def route_after_model(state: AgentState) -> str:
#     """This IS your old if/elif/else — just returning a node NAME instead
#     of executing a branch directly."""
#     if state["iterations"] >= MAX_ITERATIONS:
#         return END
#     last_message = state["messages"][-1]["content"]
#     tool_name, _ = parse_tool_call(last_message)
#     if tool_name:
#         return "call_tool"
#     if "FINAL_ANSWER:" in last_message:
#         return END
#     return "call_model"  # malformed reply — loop back and try again

def route_after_model(state: AgentState) -> str:
    if state["iterations"] >= MAX_ITERATIONS:
        return END
    last = state["messages"][-1]
    last_content = last.content if hasattr(last, "content") else last["content"]
    tool_name, _ = parse_tool_call(last_content)
    if tool_name:
        return "call_tool"
    if "FINAL_ANSWER:" in last_content:
        return END
    return "call_model"

def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("call_model", call_model_node)
    graph.add_node("call_tool", call_tool_node)

    graph.add_edge(START, "call_model")
    graph.add_conditional_edges("call_model", route_after_model)
    graph.add_edge("call_tool", "call_model")  # after a tool runs, always go back to the model

    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer) 


# if __name__ == "__main__":
#     app = build_graph()

#     for q in ["What is 15 * 23 + 7?"]:
#         print(f"\n{'=' * 70}\nQuery: {q}\n{'=' * 70}")
#         result = app.invoke(
#             {"messages": [
#                 {"role": "system", "content": build_system_prompt()},
#                 {"role": "user", "content": q},
#             ], "iterations": 0},
#             config={"configurable": {"thread_id": "test-1"}},
#         )
#         last = result["messages"][-1]
#         last_content = last.content if hasattr(last, "content") else last["content"]
#         print(f"\nFINAL (last message): {last_content}")
if __name__ == "__main__":
    import os
    tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2") == "true"
    print(f"[langsmith] tracing enabled: {tracing_enabled}")
    if not tracing_enabled:
        print("[langsmith] set LANGCHAIN_TRACING_V2=true and LANGCHAIN_API_KEY to enable full dashboard tracing")

    app = build_graph()

    for q in [
        "What is 15 * 23 + 7?",
        "According to the corpus, what does the Navigator LLM do in CoLLM-NAS?",
    ]:
        print(f"\n{'=' * 70}\nQuery: {q}\n{'=' * 70}")

        scored = memory.retrieve_with_scores(q, k=2)
        relevant = [m["memory"] for m in scored if m["distance"] < RELEVANCE_THRESHOLD]
        # memory_context = ""
        # if relevant:
        #     memory_context = "\n\nRelevant past interactions:\n" + "\n".join(
        #         f"- {m['text']}" for m in relevant
        #     )
        #     print(f"[memory] found {len(relevant)} relevant past interaction(s) under threshold")
        memory_context = ""
        if relevant:
            memory_context = (
                "\n\nRelevant past interactions (for context only — you must "
                "still respond using TOOL_CALL or FINAL_ANSWER format):\n"
                + "\n".join(f"- {m['text']}" for m in relevant)
            )
        result = app.invoke(
            {"messages": [
                {"role": "system", "content": build_system_prompt() + memory_context},
                {"role": "user", "content": q},
            ], "iterations": 0, "last_call": None},
            config={"configurable": {"thread_id": f"test-{q[:10]}"}},
        )
        last = result["messages"][-1]
        last_content = last.content if hasattr(last, "content") else last["content"]
        print(f"\nFINAL: {last_content}")
        