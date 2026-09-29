import re
import time
import os
from openai import OpenAI
from agents_tools import AVAILABLE_TOOLS

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
MODEL_NAME = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
client = OpenAI(base_url=VLLM_BASE_URL, api_key="not-needed", timeout=60.0)

MAX_ITERATIONS = 5
MAX_TOOL_RETRIES = 2


def build_system_prompt() -> str: 
    tool_lines = "\n".join(
        f"- {name}({', '.join(fn.tool_schema['parameters'])}): {fn.tool_schema['description']}"
        for name, fn in AVAILABLE_TOOLS.items()
    )
    return f"""You are an agent that can use tools to answer questions.

Available tools:
{tool_lines}

Respond with EXACTLY ONE line, in one of these two forms. Replace the
placeholder with a real value — never leave the word "argument" or a
parameter name in your reply.

TOOL_CALL: calculator(15 * 23 + 7)
FINAL_ANSWER: 352

Do not write both on the same turn. Do not repeat the line. Only call
FINAL_ANSWER once you have seen a real tool result in this conversation."""

def parse_tool_call(text: str):
    match = re.search(r'TOOL_CALL:\s*(\w+)\(([^)]*)\)', text)
    if match:
        return match.group(1), match.group(2).strip().strip('"\'')
    return None, None

def execute_tool_with_retry(name: str, arg: str) -> str:
    if name not in AVAILABLE_TOOLS:
        return f"Error: unknown tool '{name}'. Available: {list(AVAILABLE_TOOLS.keys())}"
    for attempt in range(MAX_TOOL_RETRIES + 1):
        try:
            return AVAILABLE_TOOLS[name](arg)
        except Exception as e:
            if attempt < MAX_TOOL_RETRIES:
                print(f"    [retry {attempt + 1}] '{name}' failed: {e}")
                time.sleep(1)
            else:
                return f"Error: '{name}' failed after {MAX_TOOL_RETRIES} retries: {e}"


def run_agent(user_query: str, verbose: bool = True) -> dict:
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": user_query},
    ]
    trace = []

    for i in range(1, MAX_ITERATIONS + 1):
        response = client.chat.completions.create(
            model=MODEL_NAME, messages=messages, temperature=0.1, max_tokens=300,
        )
        content = response.choices[0].message.content
        if verbose:
            print(f"\n[iteration {i}] raw model output:\n{content}")
        trace.append({"iteration": i, "model_output": content})

        tool_name, tool_arg = parse_tool_call(content)
        if tool_name:
            if verbose:
                print(f"[iteration {i}] calling {tool_name}({tool_arg!r})")
            result = execute_tool_with_retry(tool_name, tool_arg)
            if verbose:
                print(f"[iteration {i}] tool result: {result[:200]}")
            trace.append({"iteration": i, "tool": tool_name, "arg": tool_arg, "result": result})
            # Only keep the TOOL_CALL line in history — strip any FINAL_ANSWER
            # that rode along with it, so the model doesn't "remember" answering
            # a question it hasn't actually seen the observation for yet.
            clean_reply = content.split("FINAL_ANSWER:")[0].strip()
            messages.append({"role": "assistant", "content": clean_reply})
            messages.append({"role": "user", "content": f"Tool result: {result}"})
        elif "FINAL_ANSWER:" in content:
            answer = content.split("FINAL_ANSWER:")[-1].strip()
            return {"query": user_query, "answer": answer, "trace": trace, "iterations": i}
        else:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "Reply with either TOOL_CALL: or FINAL_ANSWER:, nothing else."})
    return {
        "query": user_query,
        "answer": "Gave up: hit the iteration cap without a final answer.",
        "trace": trace,
        "iterations": MAX_ITERATIONS,
        "hit_cap": True,
    }


if __name__ == "__main__":
    for q in [
        "What is 15 * 23 + 7?",
        "According to the corpus, what does the Navigator LLM do in CoLLM-NAS?",
    ]:
        print(f"\n{'=' * 70}\nQuery: {q}\n{'=' * 70}")
        result = run_agent(q)
        print(f"\nFINAL: {result['answer']}")
        print(f"Iterations used: {result['iterations']}")